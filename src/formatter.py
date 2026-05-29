import os      # ファイルパスの結合や存在確認など、OS依存のファイル操作を行うライブラリ
import budoux  # Google製。日本語の文脈を解析し、テロップが不自然な位置で改行されないよう美しい文節区切りを計算するライブラリ
from tqdm import tqdm  # tqdm.write を使って、進捗バーを破壊せずにログを出力するためのライブラリ

# 設定ファイルから文字数制限の設定値を参照する
from src.config import MIN_CHAR_LEN, MAX_CHAR_LEN

# BudouXの日本語解析デフォルトモデルをメモリに読み込み（文章を美しい文節単位にチョップする準備）
parser = budoux.load_default_japanese_parser()

def get_unique_filepath(file_path):
    """ファイルの上書きを完全に防止する関数

    指定された出力先ファイルパスがすでに存在する場合、既存データを破壊しないよう
    ファイル名の末尾に「_1」「_2」「_3」のような連番を自動付与してユニークなパスを返します。
    """
    if not os.path.exists(file_path):
        return file_path

    base_path, ext = os.path.splitext(file_path)
    counter = 1

    while True:
        new_file_path = f"{base_path}_{counter}{ext}"
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1


def format_timestamp(seconds):
    """秒数（浮動小数点数）を、SRT字幕規格のフォーマット（HH:MM:SS,mmm）に変換する関数"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))

    # 丸め処理によってミリ秒が1000に達した場合の、上位桁への繰り上げ安全処理
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


def process_segment_to_lines(segment, min_len=10, max_len=20):
    """LLMが校正した1つのセグメントを、テロップルール（10〜20文字）に応じてミリ秒単位で切り刻む関数"""

    # 1. Whisperが計測した生のタイムスタンプ（start/end）を全wordから時系列順に取り出す
    #    LLM校正後も words のタイムスタンプは一切変更していないため、ここが常に正確な時間の根拠になる
    words_info = []
    for w in segment.get("words", []):
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        words_info.append({
            "start": w_start,
            "end": w_end
        })

    # セグメント全体のテキスト（LLM校正後）を取得し、句読点を除去してからBudouXに渡す
    full_text = segment.get("text", "").replace("、", "").replace("。", "").strip()
    if not full_text or not words_info:
        return []

    # 2. BudouXで文章を文節単位（chunks）に分解する
    chunks = parser.parse(full_text)

    final_processed_lines = []
    current_line_text = ""
    chunk_buffer = []  # 現在ビルド中の行に含まれるchunkを記録（タイムスタンプ推定のインデックス計算に使用）

    total_chunks = len(chunks)
    total_words = len(words_info)

    for idx, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        # 単一の文節チャンクが20文字を超える超特殊ケースの安全弁（通常は発生しない）
        if chunk_len > max_len:
            if current_line_text:
                # 書きかけの行を先に確定させてからはみ出しchunkを処理する
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                l_start = words_info[max(0, start_w_idx)]["start"]
                l_end = words_info[max(0, end_w_idx)]["end"]
                final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})
                current_line_text = ""
                chunk_buffer = []

            # はみ出しchunkをそのまま1行として出力
            w_idx = min(int(idx * total_words / total_chunks), total_words - 1)
            final_processed_lines.append({
                "text": chunk,
                "start": words_info[w_idx]["start"],
                "end": words_info[w_idx]["end"]
            })
            continue

        # 現在の行にこのchunkを追加すると20文字を超える場合→現在行を確定して次の行を開始
        # ※ min_len 未満でも、結合すると max_len を超えるなら確定させる（min_lenは参考値扱い）
        if len(current_line_text) + chunk_len > max_len:
            if current_line_text:
                # タイムスタンプは「この行に含まれる最初のword」〜「最後のword」の生データを使う
                # chunk数とword数が一致しないため、インデックスを比率で推定している
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                l_start = words_info[max(0, start_w_idx)]["start"]
                l_end = words_info[max(0, end_w_idx)]["end"]
                final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})

            current_line_text = chunk
            chunk_buffer = [chunk]
        else:
            # 20文字以内に収まるのでそのまま結合
            current_line_text += chunk
            chunk_buffer.append(chunk)

    # ループ終了後にバッファに残った最後の1行を回収
    if current_line_text:
        start_w_idx = int((total_chunks - len(chunk_buffer)) * total_words / total_chunks)
        l_start = words_info[max(0, start_w_idx)]["start"]
        l_end = words_info[-1]["end"]  # 最後の行の終端は音声データの実際の終了時間を使う

        # 最後の行が min_len 未満なら、直前行と結合できるか試みる（短すぎる字幕を防ぐ）
        if len(current_line_text) < min_len and final_processed_lines:
            if len(final_processed_lines[-1]["text"]) + len(current_line_text) <= max_len:
                final_processed_lines[-1]["text"] += current_line_text
                final_processed_lines[-1]["end"] = l_end
            else:
                final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})
        else:
            final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})

    return final_processed_lines


def write_srt_file(refined_segments, output_srt_path):
    """10〜20文字に切り分けられた全行データを、SRT規約に沿ったファイルとして書き出す関数"""
    try:
        srt_index = 1
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for segment in refined_segments:
                # 各セグメントを10〜20文字単位の行データに分割
                split_lines = process_segment_to_lines(
                    segment, min_len=MIN_CHAR_LEN, max_len=MAX_CHAR_LEN
                )

                for line_data in split_lines:
                    line_text = line_data["text"].strip()
                    if not line_text:
                        continue

                    f.write(f"{srt_index}\n")  # 字幕の通し番号
                    f.write(f"{format_timestamp(line_data['start'])} --> {format_timestamp(line_data['end'])}\n")  # タイムスタンプ
                    f.write(f"{line_text}\n\n")  # 字幕テキスト本体（末尾の空行はSRT規約上必須）

                    srt_index += 1

        return True
    except IOError as e:
        tqdm.write(f"[*] エラー: SRTファイルの書き込みに失敗しました: {e}")
        return False