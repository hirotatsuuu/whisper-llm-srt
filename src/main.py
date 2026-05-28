import os          # ファイルやフォルダのパス操作、ファイルの実在確認（os.path.exists）に使用
import math        # 今回のコードでは直接使用していませんが、数値計算の拡張用に読み込まれています
import argparse    # コマンドラインから引数（ファイルパスやモデルサイズ等）を受け取り、解析するために使用
import time        # 処理時間を小数点2桁まで精密に計測（time.perf_counter）するために使用
import subprocess  # 外部プログラム（ffmpeg）をバックグラウンドで安全に呼び出して実行するために使用
import sys         # エラー発生時にプログラムを途中で安全に強制終了（sys.exit）させるために使用
from whisper import load_model  # OpenAIの音声認識AI「Whisper」のモデルをパソコンに読み込むために使用
import budoux      # Google製の日本語文節区切りライブラリ。テロップを不自然な単語の途中でぶつ切りさせないために使用

# BudouXの日本語モデルを読み込み
parser = budoux.load_default_japanese_parser()

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE = "./data/test.m4a"   # 引数なしで実行した際に自動で読み込まれる既定のファイル
DEFAULT_DICT_FILE  = "dictionary.txt"    # 優先的に認識させたい固有名詞・専門用語を並べたテキストファイル
DEFAULT_MODEL_SIZE = "base"              # 使用するWhisperのモデルサイズ（tiny/base/small/medium/large）

# 字幕（テロップ）として画面に表示させる際の文字数制限
MIN_CHAR_LEN = 10  # 1行の最低文字数
MAX_CHAR_LEN = 20  # 1行の最大文字数（絶対にこの文字数を超えないようにガードします）

# 動画ファイルとして認識する拡張子のリスト
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 動画から抽出した一時音声ファイル（.m4a）を処理後に削除するかどうか
# False: ファイルを消さずにそのまま保存する（初期設定。動画編集ソフト等で音声も使いたい場合に便利）
# True : 文字起こしが終わったら、抽出した .m4a ファイルを自動で削除してフォルダをスッキリさせる
REMOVE_TEMP_AUDIO = False
# =====================================================================

def get_unique_filepath(file_path):
    """
    ファイルの上書きを防止する関数
    もし指定されたファイルパスがすでに存在する場合、ファイル名の末尾に「_1」「_2」のように
    自動で連番を付与して、完全にユニークな新しいファイルパスを返します。
    """
    # そもそもファイルが存在しない場合は、重複していないのでそのままのパスを返す
    if not os.path.exists(file_path):
        return file_path
        
    # パスを「拡張子の前の部分」と「拡張子（.srtなど）」に分離
    base_path, ext = os.path.splitext(file_path)
    counter = 1  # ファイル名の末尾に付与する連番の初期値
    
    # 重複しないファイル名が見つかるまで無限ループで探す
    while True:
        new_file_path = f"{base_path}_{counter}{ext}"  # 例: test_1.srt を作成
        if not os.path.exists(new_file_path):          # そのファイル名がまだ使われていなければ確定
            return new_file_path
        counter += 1                                   # すでに存在していれば、数値を1増やして再チェック

def load_word_dictionary(file_path):
    """外部のテキストファイル（単語辞書）から単語リストを読み込む"""
    # パスが空、またはファイルが存在しない場合は空のリストを返す
    if not file_path or not os.path.exists(file_path):
        return []
        
    word_dict = []  # 読み込んだ単語を格納するための空リストを定義
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後の余白や改行コードを除去
                # 空行ではなく、かつ「#」から始まるコメント行でもない場合にリストに追加
                if word and not word.startswith("#"):
                    word_dict.append(word)
    except Exception as e:
        # 読み込み中にエラーが起きても全体を巻き込んでクラッシュさせず、警告を出して続行
        print(f"[!] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")
    return word_dict

def format_timestamp(seconds):
    """秒数をSRT形式の時間フォーマット（HH:MM:SS,mmm）に正確に変換"""
    hours = int(seconds // 3600)             # 全体の秒数から「時間」を計算
    minutes = int((seconds % 3600) // 60)    # 残りの秒数から「分」を計算
    secs = int(seconds % 60)                 # さらに残った「秒」の整数部分を計算
    milliseconds = int(round((seconds % 1) * 1000))  # 小数点以下の部分をミリ秒（3桁）に変換
    
    # 四捨五入の影響でミリ秒が1000（1秒）に達してしまった場合の繰り上げ処理
    if milliseconds >= 1000:
        milliseconds -= 1000                 # ミリ秒を0にリセット
        secs += 1                            # 秒を1増やす
        if secs >= 60:
            secs -= 60                       # 秒が60になったら0にリセット
            minutes += 1                     # 分を1増やす
            if minutes >= 60:
                minutes -= 60                # 分が60になったら0にリセット
                hours += 1                   # 時間を1増やす
                
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def extract_audio_from_video(video_path, output_audio_path):
    """ffmpegをバックグラウンドで呼び出し、動画から音声だけを高速・無劣化で抽出する"""
    print(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")
    
    # 1段階目：動画の音声を変換せずそのままコピーして高速抽出するコマンド群
    command = [
        "ffmpeg", "-y",             # 上書き許可オプション
        "-i", video_path,           # 入力動画ファイルパス
        "-vn",                      # 映像ストリームを無視（音声のみにするフラグ）
        "-acodec", "copy",          # 音声を再エンコードせず無劣化でそのまま複製
        output_audio_path           # 出力ファイルパス
    ]
    
    try:
        # ffmpegのログが画面を埋め尽くさないよう非表示（DEVNULL）で実行
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[✓] 音声の抽出が完了しました: {output_audio_path}")
        return True
    except subprocess.CalledProcessError:
        # 動画の音声形式（PCM等）によってはcopyが使えない場合があるため、その際の救済措置
        print("[!] 音声の無劣化抽出に失敗しました。エンコード抽出に切り替えます...")
        # 2段階目：汎用性の高いAAC形式に音声エンコードしながら抽出するコマンド群
        fallback_command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "aac", output_audio_path]
        try:
            subprocess.run(fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[✓] 音声の抽出（再エンコード）が完了しました: {output_audio_path}")
            return True
        except Exception as e:
            print(f"エラー: ffmpegでの音声抽出に致命的な失敗をしました。ffmpegがインストールされているか確認してください。 {e}")
            return False
    except FileNotFoundError:
        print("エラー: システムに 'ffmpeg' コマンドが見つかりません。READMEの手順に従って ffmpeg をインストールしてください。")
        return False

def process_segment_to_lines(segment, min_len=10, max_len=20):
    """Whisperのセグメントを文字数制限・ルールに応じて精密に分割する"""
    words_data = []  # セグメント内の単語ごとのテキストと時間データを整理して格納するリスト
    for w in segment.get("words", []):
        word_text = w["word"]  # 切り出された単語テキスト（例: "こんにちは"）
        words_data.append({
            "text": word_text,
            "start": float(w["start"]),  # 単語の発話開始時間（秒）
            "end": float(w["end"])       # 単語の発話終了時間（秒）
        })
        
    lines = []                 # 最終的にSRTに書き出すため、文字数調整が完了した行データを詰め込むリスト
    current_line_text = ""     # 現在文字を積み上げている途中の、1行分のテキスト変数
    current_line_start = None  # 現在の行の開始時間（最初の単語が追加されたタイミングで記録）
    current_line_end = None    # 現在の行の終了時間（単語が追加されるたびに常に最新情報に更新）
    
    # 整理した単語データを先頭から1つずつチェックして組み立てるループ
    for w_info in words_data:
        w_text = w_info["text"]    # 処理対象の単語
        w_start = w_info["start"]  # 処理対象の単語の開始秒数
        w_end = w_info["end"]      # 処理対象の単語の終了秒数
        
        has_period = "。" in w_text  # 単語の中に句点（。）が含まれているかどうかの判定フラグ
        clean_w_text = w_text.replace("、", "").replace("。", "")  # 画面表示の邪魔になる読点・句点を消去
        
        # 句読点を消した結果、文字が空っぽになった場合の処理
        if not clean_w_text:
            # 文字は無いが「。」があった場合、そこまでの文章が存在していれば強制改行（区切り）を執行
            if has_period and current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
                current_line_start = None
                current_line_end = None
            continue  # 次の単語の処理へスキップ
            
        # パターンA：単語1つだけで最大制限文字数（20文字）を超えてしまっている場合の処理
        if len(clean_w_text) > max_len:
            # すでに書きかけの行があれば、一旦そこで区切って確定させる
            if current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
            
            w_dur = w_end - w_start   # 巨大な単語全体にかかっている発話時間（秒数）
            w_len = len(clean_w_text) # 巨大な単語の合計文字数
            
            # 最大文字数を超えている間、頭から20文字ずつ強引に切り取って分割していくループ
            while len(clean_w_text) > max_len:
                sub_text = clean_w_text[:max_len]  # 頭から最大文字数分（20文字）を抽出
                sub_start = w_start                # 切り出し行の開始時間
                # 文字数の比率に応じて、発話時間を割り算（等分）して表示時間を伸縮させる
                sub_end = w_start + (w_dur * (max_len / w_len))
                
                lines.append({"text": sub_text, "start": sub_start, "end": sub_end})
                clean_w_text = clean_w_text[max_len:]  # 処理した20文字を元の単語から削る（残りを次に回す）
                w_start = sub_end                      # 次の切り出しの開始時間は、今の終了時間にする
            
            # 20文字ずつ切り刻んで最後に残った端数文字があれば、それを新しい行の書き出しとする
            if clean_w_text:
                current_line_text = clean_w_text
                current_line_start = w_start
                current_line_end = w_end
                
        # パターンB：現在の行にこの単語を足すと、制限（20文字）を超えてしまう場合の処理
        elif len(current_line_text) + len(clean_w_text) > max_len:
            # 溢れてしまうので、現在書きかけの行をここで一旦確定させて保存
            if current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
            # 溢れた単語を、次の新しい行の最初の文字としてセット
            current_line_text = clean_w_text
            current_line_start = w_start
            current_line_end = w_end
            
        # パターンC：足しても制限（20文字）以内に安全に収まる場合の処理
        else:
            # まだ新しい行を書き始めたばかりで開始時間が未設定なら、この単語の時間を開始時間とする
            if not current_line_text:
                current_line_start = w_start
            current_line_text += clean_w_text   # テキストに単語を結合
            current_line_end = w_end            # 行の終了時間をこの単語の終了時間へ更新
            
        # 単語の処理が終わった際、そこに「。」が含まれていた場合の処理
        if has_period:
            if current_line_text:
                # 「。」は文章の終わりを意味するので、文字数に余裕があってもここで1回区切る
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
                current_line_start = None
                current_line_end = None

    # すべての単語のチェックが終わった後、保存されずに残っている書きかけの最後の行があれば回収
    if current_line_text:
        lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
        
    return lines

def transcribe_to_custom_srt(audio_path, output_srt_path, dict_file_path=None, model_size="base"):
    """音声ファイルを読み込んでWhisper文字起こしを実行し、カスタムSRTを出力する"""
    print(f"[*] モデル '{model_size}' を読み込み中...")
    
    try:
        model = load_model(model_size)  # 指定されたサイズのWhisperモデルをメモリ上に展開
    except MemoryError:
        print(f"エラー: パソコンのメモリが不足しているため、モデル '{model_size}' を読み込めませんでした。")
        print("初期設定エリアの DEFAULT_MODEL_SIZE を 'tiny' や 'base' に下げて再試行してください。")
        return False
    except Exception as e:
        print(f"エラー: Whisperモデルの読み込み中に予期せぬエラーが発生しました: {e}")
        return False
    
    prompt_string = ""  # Whisperに認識の癖を仕込むためのプロンプト初期文字列
    if dict_file_path and os.path.exists(dict_file_path):
        word_dict = load_word_dictionary(dict_file_path)  # テキストから単語リストを生成
        if word_dict:
            # 辞書単語を「。」と「、」で囲んで1つの文章のように成形（Whisperが最も理解しやすい構造）
            prompt_string = "。" + "、".join(word_dict) + "。"
            print(f"[*] 【辞書適用】'{dict_file_path}' から単語を読み込みました")

    print(f"[*] 文字起こしを開始します: {audio_path}")
    
    try:
        # Whisperでの音声解析コア処理
        result = model.transcribe(
            audio_path, 
            verbose=None,                 # 進捗ログを非表示にして動作を軽量化
            fp16=False,                   # CPU環境でもエラーが出ないように安全な32bit浮動小数点演算を強制
            initial_prompt=prompt_string, # 成形した単語辞書をプロンプトとしてAIに注入
            language="ja",                # 言語を日本語に完全固定して誤認識を防止
            word_timestamps=True          # ミリ秒単位での結合分割を行うため、単語ごとの時間データを取得
        )
    except RuntimeError as e:
        print(f"エラー: 文字起こし処理中にエラーが発生しました（ファイルが破損している可能性があります）: {e}")
        return False
    except Exception as e:
        print(f"エラー: 予期せぬエラーが発生しました: {e}")
        return False
    
    print("[*] SRTファイルを生成中...")
    try:
        srt_index = 1  # SRT字幕ブロックに割り振る連番インデックスの初期値
        with open(output_srt_path, "w", encoding="utf-8") as f:
            # 解析結果から、まとまった発話ブロック（セグメント）を順に取り出すループ
            for segment in result["segments"]:
                # 1つの長いセグメントを、自前の10〜20文字制限ルールに従ってバラバラの行に分解
                split_lines = process_segment_to_lines(segment, min_len=MIN_CHAR_LEN, max_len=MAX_CHAR_LEN)
                for line_data in split_lines:
                    line_text = line_data["text"].strip()  # 行の前後のゴミスペースを掃除
                    if not line_text:                      # 中身が完全に空っぽの行は書き込まない
                        continue
                        
                    line_start = line_data["start"]        # 整形された行の表示開始秒数
                    line_end = line_data["end"]            # 整形された行の表示終了秒数
                    
                    # SRTファイル形式の標準規約に合わせてテキストを書き出し
                    f.write(f"{srt_index}\n")   # 1行目：連番番号
                    # 2行目：タイムスタンプ（例：00:01:23,456 --> 00:01:25,789）
                    f.write(f"{format_timestamp(line_start)} --> {format_timestamp(line_end)}\n")
                    f.write(f"{line_text}\n\n") # 3行目：表示するテロップ文字 ＋ 間隔を空けるための空行
                    
                    srt_index += 1  # 次のテロップのために番号を1増やす
        print(f"[✓] 字幕作成が完了しました！ 保存先: {output_srt_path}")
        return True
    except IOError as e:
        print(f"エラー: SRTファイルの書き込みに失敗しました（ファイルが開いたままになっていませんか？）: {e}")
        return False

def main():
    # 引数を受け取るための解析機（パーサー）を準備
    parser = argparse.ArgumentParser(description="動画または音声ファイルから10〜20文字に最適化されたSRT字幕を出力するスクリプト")
    
    # 3つの引数を設定（指定がなければ、初期設定エリアで決めたデフォルト値が自動で割り当てられる）
    parser.add_argument("input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help=f"入力ファイル（動画または音声）のパス")
    parser.add_argument("-d", "--dict", default=DEFAULT_DICT_FILE, help=f"単語リストのパス")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_SIZE, help=f"モデルサイズ")

    args = parser.parse_args()  # コマンドラインから渡された引数を実際に解析して確定

    # 指定されたファイルがパソコン内に存在するかチェックし、無ければエラー終了
    if not os.path.exists(args.input_file):
        print(f"エラー: 指定されたファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    # 処理にかかったトータル時間を計測するため、開始時の精密クロック時間を取得
    start_time = time.perf_counter()

    # 入力ファイルパスから「名前の部分」と「拡張子（.mp4など）」を切り離す
    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()  # 判定ミスを防ぐため大文字（.MP4等）を小文字（.mp4）に統一

    # SRTのファイル重複を防ぐためにユニークなパスを自動計算
    raw_srt_path = base_path + ".srt"
    output_srt_path = get_unique_filepath(raw_srt_path)
    
    target_audio_file = args.input_file             # Whisperに最終的に投入するファイルパス用の変数
    is_video_input = ext_lower in VIDEO_EXTENSIONS  # 入力ファイルが動画拡張子リストに含まれるかどうかの真偽値フラグ

    # もし入力されたのが動画ファイルだった場合の音声事前抽出処理
    if is_video_input:
        # 音声抽出ファイルもすでに存在する場合は自動で連番にする
        raw_audio_path = base_path + ".m4a"
        extracted_audio_path = get_unique_filepath(raw_audio_path)
        
        # 動画から音声ストリームを抜き出す関数を実行、失敗したらそこでプログラムを終了
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("エラー: 音声抽出に失敗したため処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path  # 解析対象を抽出した音声ファイルパスに切り替える

    # 文字起こしとSRTファイル書き出しのメイン関数を実行
    success = transcribe_to_custom_srt(
        audio_path=target_audio_file,
        output_srt_path=output_srt_path,
        dict_file_path=args.dict,
        model_size=args.model
    )
    
    # 動画からの抽出かつ削除設定がTrueの場合、最後に一時音声を消去する
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                os.remove(target_audio_file)  # 不要になった一時音声ファイルを削除
                print(f"[*] 一時音声ファイルを自動削除しました: {target_audio_file}")
        except Exception as e:
            print(f"[!] 警告: 一時音声ファイルの削除中にエラーが発生しました: {e}")

    # 文字起こし自体に失敗していた場合はエラーとして終了
    if not success:
        print("エラー: 文字起こし処理が正常に完了しませんでした。")
        sys.exit(1)

    # 処理がすべて成功したら、終了時の精密クロック時間を取得して引き算し、かかった秒数を算出
    end_time = time.perf_counter()
    print("処理時間(秒数):", '{:.2f}'.format((end_time - start_time)))

if __name__ == "__main__":
    main()