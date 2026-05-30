"""
formatter.py
【第3工程：デザイナー】BudouX による文節分割と SRT ファイル書き出しを担当するモジュール。

責務:
  - LLM 校正済みセグメントを 10〜20 文字単位の行に分割（BudouX + タイムスタンプ比率推定）
  - 分割した行データを SRT 規約に沿ったファイルとして書き出す
  - 秒数を SRT タイムスタンプ形式（HH:MM:SS,mmm）に変換

ファイルの書き込みは src/utils.py の write_text_file を経由します。
例外は src/exceptions.py で定義したカスタム例外として送出します。
"""

import os      # SRT ファイルの書き込み先ディレクトリ確認に使うライブラリ
import budoux  # Google 製。日本語の文脈を解析し、テロップが不自然な位置で改行されないよう美しい文節区切りを計算するライブラリ
from tqdm import tqdm  # tqdm.write を使って進捗バーを破壊せずにログを出力するためのライブラリ

from src.exceptions import FileWriteError

# ------------------------------------------------------------------
# タイムスタンプの変換
# ------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """秒数（浮動小数点数）を SRT 字幕規格のフォーマット（HH:MM:SS,mmm）に変換する関数。

    Args:
        seconds: 変換したい秒数（float）。

    Returns:
        SRT 規格のタイムスタンプ文字列（例: "00:01:23,456"）。
    """
    hours        = int(seconds // 3600)
    minutes      = int((seconds % 3600) // 60)
    secs         = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))

    # 丸め処理によってミリ秒が 1000 に達した場合の、上位桁への繰り上げ安全処理
    if milliseconds >= 1000:
        milliseconds -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


# ------------------------------------------------------------------
# セグメントの行分割
# ------------------------------------------------------------------

def _split_segment_to_lines(
    segment: dict,
    min_len: int,
    max_len: int,
) -> list[dict]:
    """LLM 校正済みの 1 セグメントを、テロップルール（10〜20 文字）に応じて行データに分割する関数。

    タイムスタンプの割り当て方針:
      - Whisper が測定した words の生タイムスタンプを時系列順に取り出す
      - BudouX の chunk 数と Whisper の word 数は一致しないため、
        各 chunk が何番目の word に対応するかをインデックス比率で推定する
      - 推定インデックスをもとに、その行の「最初の word の start」〜「最後の word の end」
        の生タイムスタンプを使う
      - 最後の行の終端は words[-1]["end"] から直接取得する（推定なし）
      - LLM 校正後も words のタイムスタンプは一切変更されていないため、常に正確な根拠になる

    Args:
        segment: セグメント辞書（"text" と "words" キーを持つ）。
        min_len: 1 行の最低文字数（末尾行が短すぎる場合は直前行への結合を試みる）。
        max_len: 1 行の最大文字数（絶対に超えないようにガードする）。

    Returns:
        行データのリスト。各要素は {"text": str, "start": float, "end": float} の辞書。
        テキストが空またはタイムスタンプがない場合は空リストを返す。
    """
    # 1. Whisper が計測した生タイムスタンプ（start/end）を全 word から時系列順に取り出す
    #    LLM 校正後も words のタイムスタンプは一切変更していないため、ここが常に正確な時間の根拠になる
    words_info = []
    for w in segment.get("words", []):
        words_info.append({
            "start": float(w.get("start", 0.0)),
            "end":   float(w.get("end",   0.0)),
        })

    # セグメント全体のテキスト（LLM 校正後）を取得し、句読点を除去してから BudouX に渡す
    full_text = segment.get("text", "").replace("、", "").replace("。", "").strip()
    if not full_text or not words_info:
        return []

    # BudouX の日本語解析デフォルトモデルをメモリに読み込み（文章を美しい文節単位にチョップする準備）
    _budoux_parser = budoux.load_default_japanese_parser()

    # 2. BudouX で文章を自然な文節単位（chunks）に分解する
    chunks = _budoux_parser.parse(full_text)

    final_lines  = []         # 確定した行データを格納するリスト
    current_text = ""         # 現在ビルド中の行テキスト
    chunk_buffer = []         # 現在行に含まれる chunk を記録（タイムスタンプ推定のインデックス計算に使用）
    total_chunks = len(chunks)
    total_words  = len(words_info)

    for idx, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        # 単一の文節 chunk が max_len を超える超特殊ケースの安全弁（通常は発生しない）
        if chunk_len > max_len:
            if current_text:
                # 書きかけの行を先に確定させてからはみ出し chunk を処理する
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx   = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                final_lines.append({
                    "text":  current_text,
                    "start": words_info[max(0, start_w_idx)]["start"],
                    "end":   words_info[max(0, end_w_idx)]["end"],
                })
                current_text = ""
                chunk_buffer = []

            # はみ出し chunk をそのまま 1 行として出力する
            w_idx = min(int(idx * total_words / total_chunks), total_words - 1)
            final_lines.append({
                "text":  chunk,
                "start": words_info[w_idx]["start"],
                "end":   words_info[w_idx]["end"],
            })
            continue

        # 現在行にこの chunk を追加すると max_len を超える場合 → 現在行を確定して次の行を開始する
        # min_len 未満でも、結合すると max_len を超えるなら確定させる（min_len は参考値扱い）
        if len(current_text) + chunk_len > max_len:
            if current_text:
                # タイムスタンプは「この行の最初の word の start」〜「最後の word の end」の生データを使う
                # chunk 数と word 数が一致しないため、インデックスを比率で推定している
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx   = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                final_lines.append({
                    "text":  current_text,
                    "start": words_info[max(0, start_w_idx)]["start"],
                    "end":   words_info[max(0, end_w_idx)]["end"],
                })
            current_text = chunk
            chunk_buffer = [chunk]
        else:
            # max_len 以内に収まるのでそのまま結合する
            current_text += chunk
            chunk_buffer.append(chunk)

    # ループ終了後にバッファに残った最後の 1 行を回収する
    if current_text:
        start_w_idx = int((total_chunks - len(chunk_buffer)) * total_words / total_chunks)
        l_start     = words_info[max(0, start_w_idx)]["start"]
        l_end       = words_info[-1]["end"]  # 最後の行の終端は音声データの実際の終了時間を直接使う

        # 最後の行が min_len 未満なら、直前行と結合できるか試みる（短すぎる字幕を防ぐ）
        if len(current_text) < min_len and final_lines:
            if len(final_lines[-1]["text"]) + len(current_text) <= max_len:
                final_lines[-1]["text"] += current_text
                final_lines[-1]["end"]   = l_end
            else:
                final_lines.append({"text": current_text, "start": l_start, "end": l_end})
        else:
            final_lines.append({"text": current_text, "start": l_start, "end": l_end})

    return final_lines


# ------------------------------------------------------------------
# SRT ファイルの書き出し
# ------------------------------------------------------------------

def write_srt_file(
        segments: list,
        output_srt_path: str,
        min_char_len: int,
        max_char_len: int
    ) -> None:
    """指定された文字数に切り分けられた全セグメントを SRT 規約に沿ったファイルとして書き出す関数。

    各セグメントを _split_segment_to_lines() で行データに分割してから、
    SRT の規約（通し番号 / タイムスタンプ / テキスト本体 / 空行）に従って書き出します。

    Args:
        segments: LLM 校正済みセグメントのリスト（refiner.py の出力）。
        output_srt_path: 書き出し先の SRT ファイルパス。

    Raises:
        FileWriteError: SRT ファイルへの書き込みに失敗した場合。
    """
    try:
        srt_index = 1
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for segment in segments:
                # セグメントを 10〜20 文字単位の行データに分割する
                split_lines = _split_segment_to_lines(segment, min_len=min_char_len, max_len=max_char_len)

                for line_data in split_lines:
                    line_text = line_data["text"].strip()
                    if not line_text:
                        continue

                    f.write(f"{srt_index}\n")  # 字幕の通し番号
                    f.write(
                        f"{_format_timestamp(line_data['start'])} --> {_format_timestamp(line_data['end'])}\n"
                    )  # タイムスタンプ（HH:MM:SS,mmm --> HH:MM:SS,mmm 形式）
                    f.write(f"{line_text}\n\n")  # 字幕テキスト本体（末尾の空行は SRT 規約上必須）

                    srt_index += 1

        tqdm.write(f"[+] SRT ファイルを書き出しました: {output_srt_path}")

    except OSError as e:
        raise FileWriteError(
            f"SRT ファイルへの書き込みに失敗しました: {output_srt_path} / 原因: {e}"
        ) from e