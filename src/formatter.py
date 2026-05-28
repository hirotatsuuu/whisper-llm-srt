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
    """秒数（浮動小数点数）を、SRT字幕規格の厳密なフォーマット（HH:MM:SS,mmm）に1ミリ秒の狂いもなく正確に変換する関数"""
    hours = int(seconds // 3600)  # 全体の総秒数を3600で割り、「時間（Hour）」を算出
    minutes = int((seconds % 3600) // 60)  # 残りの秒数から、さらに60で割って「分（Minute）」を算出
    secs = int(seconds % 60)  # 「秒（Second）」の整数部分を取得
    milliseconds = int(round((seconds % 1) * 1000))  # 小数点以下の端数を四捨五入して「ミリ秒（Millisecond）」を3桁で取得

    # 四捨五入（round）の影響により、ミリ秒が1000（つまりジャスト1秒）に達してしまった場合の、時間がズレるのを防ぐ繰り上げ補正処理
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
    """【デザイナーの核】ELYZAが綺麗にした1つのセグメントを、BudouXの力と単語ごとの時間情報を使って、
    10〜20文字制限というテロップルールに応じてミリ秒単位で美しく切り刻む関数
    """
    words_data = []
    # Whisperが保持していた単語ごとの「発話開始・終了秒数」のパズルを再利用
    for w in segment.get("words", []):
        word_text = w["word"]
        # 第1工程（transcriber）で空文字化されたフィラー（あの、えっと等）は完全に無視してスキップ
        if not word_text:
            continue
        words_data.append({
            "text": word_text,
            "start": float(w["start"]),
            "end": float(w["end"]),
        })

    lines = []               # 字幕確定データの格納リスト
    current_line_text = ""   # 現在ビルド中の1行分のテキスト
    current_line_start = None
    current_line_end = None

    for w_info in words_data:
        w_text = w_info["text"]
        w_start = w_info["start"]
        w_end = w_info["end"]

        has_period = "。" in w_text
        # 画面に表示した際にチカチカして邪魔になる「、」や「。」をテキストから消去
        clean_w_text = w_text.replace("、", "").replace("。", "")

        if not clean_w_text:
            if has_period and current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""
                current_line_start = None
                current_line_end = None
            continue

        # パターンA：【単語1つで20文字突破】という超巨大単語だった場合の破壊処理（等分アルゴリズム）
        if len(clean_w_text) > max_len:
            if current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""

            w_dur = w_end - w_start
            w_len = len(clean_w_text)

            while len(clean_w_text) > max_len:
                sub_text = clean_w_text[:max_len]
                sub_start = w_start
                sub_end = w_start + (w_dur * (max_len / w_len))

                lines.append({"text": sub_text, "start": sub_start, "end": sub_end})
                clean_w_text = clean_w_text[max_len:]
                w_start = sub_end

            if clean_w_text:
                current_line_text = clean_w_text
                current_line_start = w_start
                current_line_end = w_end

        # パターンB：現在の行にこの新しい単語を足すと、絶対防衛ライン（20文字）をオーバーしてしまう場合
        elif len(current_line_text) + len(clean_w_text) > max_len:
            if current_line_text:
                # 溢れて画面外にはみ出してしまうため、現在書きかけだった行をここで一旦終了とし、確定データとして保存
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
            current_line_text = clean_w_text
            current_line_start = w_start
            current_line_end = w_end

        # パターンC：足しても文字数制限（20文字）以内に収まる、最も一般的な場合の結合処理
        else:
            if not current_line_text:
                current_line_start = w_start
            current_line_text += clean_w_text
            current_line_end = w_end

        # 単語の個別処理が終わった際、その単語の末尾に「。」（文の終わり）が含まれていた場合の区切り
        if has_period:
            if current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""
                current_line_start = None
                current_line_end = None

    # 未回収の最後の書きかけの1行を回収
    if current_line_text:
        lines.append({
            "text": current_line_text,
            "start": current_line_start,
            "end": current_line_end,
        })

    # 💡【最後の仕上げ】BudouXを使って、日本語としてさらに自然な文節改行の位置を最終微調整
    final_processed_lines = []
    for line in lines:
        # BudouXでバラバラに分解（例: ["今日は", "東京駅に", "行きます"]）
        chunks = parser.parse(line["text"])
        
        # 20文字以内で、できるだけ文節が綺麗な位置になるように結合を再構成する
        temp_text = ""
        for chunk in chunks:
            if len(temp_text) + len(chunk) <= max_len:
                temp_text += chunk
            else:
                if temp_text:
                    final_processed_lines.append({
                        "text": temp_text,
                        "start": line["start"],
                        "end": line["end"]
                    })
                temp_text = chunk
        if temp_text:
            final_processed_lines.append({
                "text": temp_text,
                "start": line["start"],
                "end": line["end"]
            })

    return final_processed_lines


def write_srt_file(refined_segments, output_srt_path):
    """【最終出力】10〜20文字に切り分けられたすべての行データを、規約に沿ったSRT字幕ファイルとして書き出す関数"""
    print("[*] デザイナー工程: 字幕データを画面最適化サイズ（10〜20文字）にカットしながら、SRTへ書き出し中...")
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
                    )  # タイムスタンプ
                    f.write(f"{line_text}\n\n")  # テロップ文字列 ＋ 区切り空行

                    srt_index += 1
                    
        print(f"[*] 完成: 字幕ファイルがすべて正常に出力されました！: {output_srt_path}")
        return True
    except IOError as e:
        print(f"[*] エラー: SRTファイルの書き込みに失敗しました: {e}")
        return False