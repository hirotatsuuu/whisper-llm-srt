import os  # ファイルパスの結合や、指定したファイルが実在するかの確認など、OS依存のファイル操作を行うライブラリ
import budoux  # Google製。機械学習モデルを用いて日本語の文脈を解析し、テロップが「不自然な位置」で改行されないように美しい区切りを計算するライブラリ

# BudouXの日本語解析デフォルトモデルをメモリに読み込み（文章を美しい文節単位にチョップする準備）
parser = budoux.load_default_japanese_parser()

# 司令塔（main.py）と共通の文字数制限ルールを定義（最終防衛ライン）
MIN_CHAR_LEN = 10  # 1行の最低文字数。これより短い場合は極力次の単語と結合させます
MAX_CHAR_LEN = 20  # 1行の最大文字数。YouTubeやTikTokのテロップとして最も見やすい20文字を絶対上限とします


def get_unique_filepath(file_path):
    """ファイルの上書きを完全に防止する関数

    もし指定された出力先ファイルパスがすでにフォルダ内に存在する場合、既存のデータを破壊（上書き）しないよう、
    ファイル名の末尾に「_1」「_2」「_3」のような連番を自動で付与し、完全に重複のない新しいユニークなファイルパスを作成して返します。
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
    """秒数（浮動小数点数）を、SRT字幕規格の厳密なフォーマット（HH:MM:SS,mmm）に変換する関数"""
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
    """ELYZAが綺麗にした1つのセグメントを、テロップルール（10〜20文字）に応じてミリ秒単位で美しく切り刻む関数"""
    
    # 1. セグメント内の元の単語データから、時系列順の正確な生のタイムスタンプ情報を抽出
    words_info = []
    for w in segment.get("words", []):
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        words_info.append({
            "start": w_start,
            "end": w_end
        })

    # セグメント全体のテキスト（LLM校正後）を取得
    full_text = segment.get("text", "").replace("、", "").replace("。", "").strip()
    if not full_text or not words_info:
        return []

    # 2. BudouXで文章を美しい文節（chunks）に分解
    chunks = parser.parse(full_text)
    
    final_processed_lines = []
    current_line_text = ""
    
    total_chunks = len(chunks)
    total_words = len(words_info)
    chunk_buffer = []
    
    for idx, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        # 万が一、単一の文節チャンク自体が20文字を超えている超特殊ケースの安全弁
        if chunk_len > max_len:
            if current_line_text:
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                l_start = words_info[max(0, start_w_idx)]["start"]
                l_end = words_info[max(0, end_w_idx)]["end"]
                final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})
                current_line_text = ""
                chunk_buffer = []
            
            w_idx = min(int(idx * total_words / total_chunks), total_words - 1)
            final_processed_lines.append({
                "text": chunk, 
                "start": words_info[w_idx]["start"], 
                "end": words_info[w_idx]["end"]
            })
            continue

        # 💡【文字数結合の指定ルール】
        # 指定したmin以下のテキストでも、次のテキストと結合したときにmaxを越えてしまう場合はminのままでOKとする。
        if len(current_line_text) + chunk_len > max_len:
            if current_line_text:
                # 推定（比率計算）は絶対にせず、該当する最初のwordの生startから最後のwordの生endを厳格に取得
                start_w_idx = int((idx - len(chunk_buffer)) * total_words / total_chunks)
                end_w_idx = min(int(idx * total_words / total_chunks) - 1, total_words - 1)
                l_start = words_info[max(0, start_w_idx)]["start"]
                l_end = words_info[max(0, end_w_idx)]["end"]
                
                final_processed_lines.append({"text": current_line_text, "start": l_start, "end": l_end})
            
            current_line_text = chunk
            chunk_buffer = [chunk]
        else:
            current_line_text += chunk
            chunk_buffer.append(chunk)

    # ループ終了後にバッファに残った最後の1行を確実に回収
    if current_line_text:
        start_w_idx = int((total_chunks - len(chunk_buffer)) * total_words / total_chunks)
        l_start = words_info[max(0, start_w_idx)]["start"]
        l_end = words_info[-1]["end"]  # 音声データの終端時間を直接掴み取る
        
        # 最後の残り行に対するマージ判定
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
    """【最終出力】10〜20文字に切り分けられたすべての行データを、規約に沿ったSRT字幕ファイルとして書き出す関数"""
    try:
        srt_index = 1
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for segment in refined_segments:
                # 綺麗になった各文章を10〜20文字にカットするデザイナー関数を呼び出す
                split_lines = process_segment_to_lines(
                    segment, min_len=MIN_CHAR_LEN, max_len=MAX_CHAR_LEN
                )
                
                for line_data in split_lines:
                    line_text = line_data["text"].strip()
                    if not line_text:
                        continue

                    # SRT字幕ファイルのフォーマット規約に従ってテキストをファイルに書き出し
                    f.write(f"{srt_index}\n")  # 字幕の通し番号
                    f.write(
                        f"{format_timestamp(line_data['start'])} --> {format_timestamp(line_data['end'])}\n"
                    )  # 表示する時間枠（タイムスタンプ）
                    f.write(f"{line_text}\n\n")  # 実際の字幕テキスト本体（最後に空行が必要）

                    srt_index += 1
                    
        return True
    except IOError as e:
        print(f"[*] エラー: SRTファイルの書き込みに失敗しました: {e}")
        return False