"""
refiner.py
【第2工程：頭脳】LLM（ELYZA 等）による文脈校正を担当するモジュール。

責務:
  - 外部プロンプトファイルの読み込み
  - Whisper セグメントの LLM バッチ校正（前後文脈ベース）
  - タイムスタンプの健全性チェック
  - LLM 出力の複数行パース・ID マッピング

seg["text"] のみを更新します。
formatter.py は seg["text"] を BudouX で文節分割して SRT に書き出します。
words の生タイムスタンプは一切触らず、時間情報を保護します。

ファイルの読み書きは src/utils.py の関数を経由します。
例外は src/exceptions.py で定義したカスタム例外として送出します。
"""

import time    # LLM 全体の処理時間を計測するためのライブラリ
import ollama  # ローカル LLM 実行エンジン「Ollama」と通信するためのライブラリ
from tqdm import tqdm  # tqdm.write を使って進捗バーを破壊せずにログを出力するためのライブラリ

from src.config import BATCH_SIZE_LLM, LLM_MODEL_NAME
from src.exceptions import (
    InvalidConfigError,
    InvalidTimestampError,
    LlmApiError,
    PromptFileNotFoundError,
)
from src.utils import read_text_file

# ------------------------------------------------------------------
# 出力フォーマットの定義（Python 側で管理）
# ------------------------------------------------------------------

# LLM への出力フォーマット指示と【補正対象データ】見出しを一体で定義する。
# プロンプトファイル（prompt.txt）の末尾にこの文字列を結合して完全なプロンプトを組み立てます。
# フォーマット指示を Python 側に置くことで、プロンプトファイルを編集しても
# 出力フォーマットが崩れるリスクを防いでいます。
_OUTPUT_FORMAT_BLOCK = """
【出力フォーマット】（このフォーマットを厳守してください）
ID: 番号 | TEXT: 校正後のテキスト
ID: 番号 | TEXT: 校正後のテキスト

【補正対象データ】
"""


# ------------------------------------------------------------------
# プロンプトの読み込み
# ------------------------------------------------------------------

def load_prompt_template(file_path: str) -> str:
    """外部テキストファイルからプロンプトテンプレートを読み込む関数。

    プロンプトは LLM 校正の動作に必須のため、ファイルが存在しない・空の場合は
    フォールバックせず PromptFileNotFoundError を送出してプログラムを停止させます。
    これにより「意図しない空プロンプトでの処理続行」を確実に防ぎます。

    Args:
        file_path: 読み込みたいプロンプトファイルのパス。

    Returns:
        プロンプトファイルの内容（文字列）。

    Raises:
        PromptFileNotFoundError: ファイルが存在しない、または中身が空の場合。
    """
    try:
        template = read_text_file(file_path)
    except Exception as e:
        raise PromptFileNotFoundError(
            f"プロンプトファイルの読み込みに失敗しました: {file_path} / 原因: {e}"
        ) from e

    if not template:
        raise PromptFileNotFoundError(
            f"プロンプトファイルの内容が空です: {file_path}"
        )

    prompt_filename = file_path.split("/")[-1]
    tqdm.write(f"[*] プロンプトテンプレート [{prompt_filename}] を読み込みました")
    return template


# ------------------------------------------------------------------
# タイムスタンプの健全性チェック
# ------------------------------------------------------------------

def _validate_timestamps(batch_seg: dict) -> None:
    """セグメント内の単語タイムスタンプが正常な値かどうかを検証する関数。

    タイムスタンプが不正な場合は InvalidTimestampError を送出します。
    呼び出し元でこの例外をキャッチし、LLM 校正をスキップして生データを維持します。

    検証内容:
      - start/end が None または空文字でないこと
      - start/end が float に変換できる数値であること

    Args:
        batch_seg: 検証対象のセグメント辞書（"words" キーを持つ）。

    Raises:
        InvalidTimestampError: タイムスタンプが None・空・数値変換不可能な場合。
    """
    words = batch_seg.get("words", [])
    target_id = batch_seg.get("id", "unknown")

    for w in words:
        start_val = w.get("start")
        end_val   = w.get("end")

        # None または空文字のチェック
        if start_val is None or end_val is None or start_val == "" or end_val == "":
            raise InvalidTimestampError(
                f"タイムスタンプが None または空です [ID:{target_id}] "
                f"start={start_val!r}, end={end_val!r}"
            )

        # 数値変換可能かどうかのチェック
        try:
            float(start_val)
            float(end_val)
        except (ValueError, TypeError) as e:
            raise InvalidTimestampError(
                f"タイムスタンプを数値に変換できません [ID:{target_id}] "
                f"start={start_val!r}, end={end_val!r}"
            ) from e


# ------------------------------------------------------------------
# LLM 出力のパース
# ------------------------------------------------------------------

def _parse_llm_response(llm_lines: list[str], target_id: int, old_text: str) -> str:
    """LLM の出力行リストから、指定した ID に対応するテキストを抽出する関数。

    LLM が途中で改行を入れて複数行で返してきた場合でも、
    次の「ID:」行が出現するまですべての行を結合して回収します。

    空文字が返ってきた場合（LLM の誤動作）は、元のテキストを維持するガードが働きます。

    Args:
        llm_lines: LLM の応答を改行で分割した文字列リスト。
        target_id: 取り出したいセグメントの ID 番号。
        old_text: 元のテキスト（ID が見つからない場合・空文字が返った場合のフォールバック）。

    Returns:
        LLM による校正後テキスト。見つからない場合・空文字の場合は old_text を返す。
    """
    found_target  = False   # 対象 ID の行を見つけたかどうかのフラグ
    collected_lines = []    # 複数行にまたがる場合のテキスト回収バッファ

    for line in llm_lines:
        # 対象 ID の開始行を検出する（スペースあり・なし両方に対応）
        if f"ID: {target_id} " in line or f"ID:{target_id}" in line:
            found_target = True
            if "TEXT:" in line:
                # TEXT: の後ろの部分を抽出してバッファに追加
                collected_lines.append(line.split("TEXT:", 1)[1].strip())
            continue

        # 対象 ID の回収中に別の「ID:」行が現れたら回収終了
        if found_target:
            if "ID: " in line or "ID:" in line:
                break
            # 別の ID でなければ LLM が挿入した改行とみなしてテキストを追記
            collected_lines.append(line.strip())

    if not found_target:
        # 対象 ID が LLM の出力から見つからなかった場合は元のテキストで維持
        return old_text

    # 複数行に分かれていたテキストを 1 つに結合（スペースを挟まずに結合）
    parsed_text = "".join(collected_lines).strip()

    # LLM が誤って空文字を返してきた場合は元のテキストを維持するガード
    if parsed_text == "" and old_text != "":
        return old_text

    return parsed_text


# ------------------------------------------------------------------
# LLM バッチ校正（メイン関数）
# ------------------------------------------------------------------

def refine_context_with_llm(
    segments: list,
    prompt_file_path: str,
) -> tuple[list, float]:
    """各セグメントの前後文脈をもとに LLM でバッチ校正し、校正済みセグメントと処理時間を返す関数。

    処理の流れ:
      1. プロンプトファイルを読み込み、出力フォーマットブロックと結合してベースプロンプトを作成
      2. セグメントを BATCH_SIZE_LLM 件ずつ（または句読点区切りで）バッチ化
      3. 各バッチを Ollama に送信して LLM の応答を取得
      4. 応答を ID ごとにパースして seg["text"] に反映
      5. タイムスタンプ異常・API エラーはセグメント/バッチ単位で捕捉し生データで代替

    seg["text"] のみを更新します。wordsの生タイムスタンプは一切触れません。

    Args:
        segments: フィラー除去済みのセグメントリスト（transcriber.py の出力）。
        prompt_file_path: LLM に渡すプロンプトテンプレートファイルのパス。

    Returns:
        (refined_segments, llm_elapsed_time) のタプル。
            refined_segments: LLM による校正が反映されたセグメントリスト。
            llm_elapsed_time: LLM 処理全体にかかった秒数（float）。

    Raises:
        PromptFileNotFoundError: プロンプトファイルが存在しない・空の場合（処理停止）。
        InvalidConfigError: BATCH_SIZE_LLM が 0 以下の不正な値の場合（処理停止）。
    """
    tqdm.write(f"[*] ローカル LLM による文脈バッチ校正を開始します... (バッチサイズ: {BATCH_SIZE_LLM})")

    # BATCH_SIZE_LLM の設定値が不正な場合は処理を停止する（無限ループや空バッチ送信を防ぐ）
    if BATCH_SIZE_LLM <= 0:
        raise InvalidConfigError(
            f"BATCH_SIZE_LLM の設定値が不正です: {BATCH_SIZE_LLM}（1 以上の整数を設定してください）"
        )

    # プロンプトファイルを読み込み、出力フォーマットブロックを末尾に結合してベースを作成
    # ループ内では base_prompt の末尾にバッチテキストを足すだけの最速処理にする
    prompt_template = load_prompt_template(prompt_file_path)
    base_prompt = prompt_template + "\n" + _OUTPUT_FORMAT_BLOCK

    llm_start_time = time.perf_counter()  # LLM 全体の処理時間計測開始（ループの外で 1 回だけ）

    refined_segments = []
    current_batch    = []

    for index, seg in enumerate(segments):
        current_batch.append(seg)

        # 句読点（。、）があるか、指定件数溜まったか、最後の要素ならバッチ処理を実行する
        has_punctuation = "。" in seg["text"] or "、" in seg["text"]
        is_batch_full   = len(current_batch) >= BATCH_SIZE_LLM or index == len(segments) - 1

        if not (has_punctuation or is_batch_full):
            continue

        # バッチ内の各セグメントを「ID: X | TEXT: Y」形式の文字列に整形する
        batch_prompt_text = ""
        for batch_seg in current_batch:
            batch_prompt_text += f"ID: {batch_seg['id']} | TEXT: {batch_seg['text']}\n"

        # ベースプロンプトの末尾にバッチテキストを結合して完全なプロンプトを組み立てる
        prompt = base_prompt + batch_prompt_text

        try:
            # Ollama API を呼び出してローカル LLM（ELYZA 等）を実行する
            response = ollama.chat(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )

            # LLM からの返答を行ごとに分解して解析しやすくする
            llm_lines = response["message"]["content"].strip().split("\n")

            # 各セグメントに対して LLM の修正結果をマッピングする
            for batch_seg in current_batch:
                target_id = batch_seg["id"]
                old_text  = batch_seg["text"]

                # タイムスタンプの健全性チェック（異常があればそのセグメントは生データで維持）
                try:
                    _validate_timestamps(batch_seg)
                except InvalidTimestampError as e:
                    tqdm.write(f"\n[警告 タイムスタンプ異常 ID:{target_id}] 生データを維持します / 原因: {e}")
                    refined_segments.append(batch_seg)
                    continue

                # LLM の出力から対象 ID のテキストを抽出する
                corrected_text = _parse_llm_response(llm_lines, target_id, old_text)

                # 修正が行われた場合のみ差分をログ出力する
                if old_text != corrected_text:
                    tqdm.write(f"\n[修正検出 ID:{target_id}]")
                    tqdm.write(f"  BEFORE: {old_text}")
                    tqdm.write(f"  AFTER : {corrected_text}")

                # seg["text"] だけを更新する（words の生タイムスタンプには一切触れない）
                batch_seg["text"] = corrected_text
                refined_segments.append(batch_seg)

        except LlmApiError as e:
            # Ollama 通信エラーはバッチ単位でキャッチし、生データで代替して処理を継続する
            tqdm.write(f"\n[警告] LLM API エラーが発生しました（バッチを生データで維持します）: {e}")
            for batch_seg in current_batch:
                refined_segments.append(batch_seg)

        except Exception as e:
            # 予期せぬエラーもバッチ単位でキャッチし、生データで代替して処理を継続する
            # （パイプライン全体のクラッシュを防ぐ堅牢性優先の設計）
            tqdm.write(f"\n[警告] バッチ処理中に予期せぬエラーが発生しました（生データを維持します）: {e}")
            for batch_seg in current_batch:
                refined_segments.append(batch_seg)

        # バッチをリセットして次のグループに備える
        current_batch = []

    # LLM 全体の処理時間を計測終了（pipeline.py 側で一元表示するため、ここでは出力しない）
    llm_elapsed_time = time.perf_counter() - llm_start_time

    return refined_segments, llm_elapsed_time