import os      # プロンプトファイルのパス確認・読み込みに使うライブラリ
import sys     # プロンプト不在時にエラー終了させるためのライブラリ
import ollama  # ローカルLLM実行エンジン「Ollama」と通信するためのライブラリ
import time    # 処理時間を小数点2桁まで精密に計測するためのライブラリ
from tqdm import tqdm  # tqdm.write を使って、進捗バーを破壊せずにログを出力するためのライブラリ

# 設定ファイルからLLMの設定を参照する
from src.config import LLM_MODEL_NAME, LLM_PROMPT_FILE, BATCH_SIZE_LLM

# 出力フォーマットの末尾に「【補正対象データ】」の見出しまで含めて定義
OUTPUT_FORMAT_PROMPT = """
【出力フォーマット】（このフォーマットを厳守してください）
ID: 番号 | TEXT: 校正後のテキスト
ID: 番号 | TEXT: 校正後のテキスト

【補正対象データ】
"""
# =====================================================================


def load_prompt_template(file_path):
    """外部テキストファイルからプロンプトテンプレートを読み込む関数。

    ファイルが存在しない、または空の場合はフォールバックせず、
    エラーを出力してプログラムを完全に終了（明示的な例外を発生）させます。
    """
    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"\n[❌ 致命的エラー] プロンプトファイルが見つかりません: {file_path}")
        raise FileNotFoundError(f"必須のプロンプトファイルが存在しません: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            template = f.read().strip()

        if not template:
            tqdm.write(f"\n[❌ 致命的エラー] プロンプトファイルが空です: {file_path}")
            raise ValueError(f"プロンプトファイルの内容が空です: {file_path}")

        prompt_filename = os.path.basename(file_path)
        tqdm.write(f"[*] プロンプトテンプレート [{prompt_filename}] を読み込みました")
        return template

    except Exception as e:
        tqdm.write(f"\n[❌ 致命的エラー] プロンプトファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        raise e


def refine_context_with_llm(segments: list) -> list:
    """
    【第2工程】
    各セグメントの前後の文脈をもとに、タイムスタンプを完全に維持したまま
    日本語として不自然な箇所をバッチ校正します。

    ※ seg["text"] のみを更新します。
       formatter.py は seg["text"] をBudouXで文節分割してSRTに書き出します。
       words の生タイムスタンプは一切触らず、時間情報を保護します。
    """
    tqdm.write(f"[*] ローカルLLMによる文脈バッチ校正を開始します... (設定バッチ数: {BATCH_SIZE_LLM})")

    # プロンプトテンプレートを外部ファイルから読み込む（失敗時はここでプログラムが終了します）
    prompt_template = load_prompt_template(LLM_PROMPT_FILE)

    # 見出しまで含めたフォーマットとベーステンプレートを、最初のほうで一元的に結合
    base_prompt_ready = prompt_template + "\n" + OUTPUT_FORMAT_PROMPT

    llm_start_time = time.perf_counter()  # LLM全体の処理時間計測開始（ループの外で1回だけ）

    # 設定値が 0 以下の異常値の時、無限ループや空バッチ送信を完璧に防ぐ安全ガード
    current_batch_size_limit = BATCH_SIZE_LLM
    if current_batch_size_limit <= 0:
        tqdm.write(f"\n[⚠ 設定エラー] BATCH_SIZE_LLM が {BATCH_SIZE_LLM} に設定されています。安全のため最小値の '1' (逐次処理) として処理します。")
        current_batch_size_limit = 1

    refined_segments = []
    current_batch = []

    for index, seg in enumerate(segments):
        current_batch.append(seg)

        # 句読点（。、）があるか、指定件数溜まったか、最後の要素ならバッチ処理を実行
        has_punctuation = "。" in seg["text"] or "、" in seg["text"]
        is_batch_full = len(current_batch) >= current_batch_size_limit or index == len(segments) - 1

        if has_punctuation or is_batch_full:

            # バッチ内の各セグメントを「ID: X | TEXT: Y」形式の文字列に整形
            batch_prompt_text = ""
            for batch_seg in current_batch:
                batch_prompt_text += f"ID: {batch_seg['id']} | TEXT: {batch_seg['text']}\n"

            # ループ内では、事前に組み立てたベースの末尾にテキストを足すだけの最速処理
            prompt = base_prompt_ready + batch_prompt_text

            try:
                # Ollama APIを呼び出してローカルLLM（ELYZA等）を実行
                response = ollama.chat(
                    model=LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )

                # LLMからの返答を行ごとに分解して解析しやすくする
                llm_lines = response["message"]["content"].strip().split("\n")

                # 各セグメントに対して、LLMの修正結果をマッピング
                for batch_seg in current_batch:
                    target_id = batch_seg["id"]
                    old_text = batch_seg["text"]
                    corrected_text = old_text  # 一致するIDが見つからない場合のフォールバック（現状維持）

                    # タイムスタンプの健全性チェック（異常があれば生データのまま維持）
                    words = batch_seg.get("words", [])
                    is_timestamp_valid = True

                    for w in words:
                        start_val = w.get("start")
                        end_val = w.get("end")
                        # タイムスタンプがNone・空・数値変換不可の場合は異常と判定
                        if start_val is None or end_val is None or start_val == "" or end_val == "":
                            tqdm.write(f"\n[⚠ タイムスタンプ空エラー ID:{target_id}] 安全のため生データを維持します。")
                            is_timestamp_valid = False
                            break
                        try:
                            float(start_val)
                            float(end_val)
                        except (ValueError, TypeError):
                            tqdm.write(f"\n[⚠ タイムスタンプ数値不正 ID:{target_id}] start:{start_val}, end:{end_val}")
                            is_timestamp_valid = False
                            break

                    # タイムスタンプが正常な場合のみLLMの校正結果を反映する
                    if is_timestamp_valid:
                        # 💡【バグ修正】LLMが途中で改行を入れて複数行で返してきた場合でも、すべて結合して回収するロジック
                        found_target = False
                        collected_lines = []

                        for line in llm_lines:
                            # ターゲットとなるIDの開始行を見つける
                            if f"ID: {target_id} " in line or f"ID:{target_id}" in line:
                                found_target = True
                                if "TEXT:" in line:
                                    # TEXT: の後ろの部分を抽出
                                    collected_lines.append(line.split("TEXT:", 1)[1].strip())
                                continue
                            
                            # ターゲットの回収中に、別の「ID:」行が出現したら回収を終了する
                            if found_target:
                                if "ID: " in line or "ID:" in line:
                                    break
                                # 別のIDでなければ、LLMが勝手に入れた改行とみなしてテキストを追記
                                collected_lines.append(line.strip())

                        if found_target:
                            # 複数行に分かれていたテキストを1つに結合（スペースを挟まずに結合）
                            parsed_text = "".join(collected_lines).strip()
                            
                            # LLMが誤って空文字を返してきた場合は元のテキストを維持するガード
                            if parsed_text == "" and old_text != "":
                                corrected_text = old_text
                            else:
                                corrected_text = parsed_text

                        # 修正が行われた場合のみ差分をログ出力
                        if old_text != corrected_text:
                            tqdm.write(f"\n[修正検出 ID:{target_id}]")
                            tqdm.write(f"  BEFORE: {old_text}")
                            tqdm.write(f"  AFTER : {corrected_text}")

                        # seg["text"] だけを更新する（wordsの生タイムスタンプには一切触れない）
                        batch_seg["text"] = corrected_text

                    refined_segments.append(batch_seg)

            except Exception as e:
                # 【堅牢なエラーハンドリング】Ollamaのエラーが起きてもクラッシュさせない。
                # このバッチはWhisperの生データをそのまま採用して生存ルートを確保する。
                tqdm.write(f"\n[*] ブロック処理中にエラーが発生しました（安全のため生データを維持します）: {e}")
                for batch_seg in current_batch:
                    refined_segments.append(batch_seg)

            # バッチをリセットして次のグループに備える
            current_batch = []

    # LLM全体の処理時間を計測終了・返却（main側で一元表示するため、ここでは出力しない）
    llm_elapsed_time = time.perf_counter() - llm_start_time

    return refined_segments, llm_elapsed_time