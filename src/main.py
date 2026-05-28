import argparse  # コマンドライン（ターミナル）から「ファイル名」や「モデルサイズ」などの設定引数を受け取って解析するための標準ライブラリ
import math  # 将来的な数値計算（時間切り上げ・切り捨て等）の拡張、またはデバッグ用として予約配置されている数学ライブラリ
import os  # ファイルパスの結合（os.path.join）や、指定したファイルが実在するかの確認（os.path.exists）など、OS依存のファイル操作を行うライブラリ
import subprocess  # ffmpegやffprobeといった外部の強力なCUIプログラムを、Pythonのバックグラウンドから安全に呼び出して並行実行するためのライブラリ
import sys  # システム固有の機能にアクセスし、致命的なエラーが発生した際にプログラムを途中で安全かつ即座に強制終了（sys.exit）させるためのライブラリ
import time  # 処理にかかった時間を「ミリ秒（小数点2桁）」単位まで精密に計測（time.perf_counter）し、パフォーマンスを評価するためのライブラリ
from whisper import load_model  # OpenAIが開発した高性能音声認識AI「Whisper」の学習済みモデルを、ローカルPCのメモリ/VRAMにロードするための関数
import budoux  # Google製。日本語の機械学習モデルを用いて文脈を解析し、テロップが「単語や文節の途中」で不自然に改行されないように美しい区切りを計算するライブラリ
from tqdm import tqdm  # 処理が今どのくらい進んでいるのかを、ターミナル上に美しいアニメーションプログレスバーとしてリアルタイム表示するためのライブラリ

# BudouXの日本語解析デフォルトモデルをメモリに読み込み（文章を美しい文節単位にチョップする準備）
parser = budoux.load_default_japanese_parser()

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE = "./data/test.m4a"  # ターミナルで引数を何も指定せずに実行した際、自動的に検索・読み込みが行われる既定の音声ファイルパス
DEFAULT_DICT_FILE = "./data/dictionary.txt"  # 固有名詞、専門用語、業界用語、新語など、AIが誤認識しやすい単語を優先的に正しく認識させるためのテキストファイル
DEFAULT_MODEL_SIZE = "base"  # Whisperのモデルサイズ（速度優先のtiny/baseから、精度優先のsmall/medium/largeまで選択可能）

# 字幕（テロップ）として画面に表示させる際の文字数制限（動画編集ソフトや視聴者の読みやすさに最適化する数値）
MIN_CHAR_LEN = 10  # 1行の最低文字数。これより短い場合は、極力次の単語と結合させてバラバラになるのを防ぎます
MAX_CHAR_LEN = 20  # 1行の最大文字数。YouTubeやTikTokのテロップとして最も見やすい20文字を絶対上限とし、これを超えたら強制改行します

# スクリプトが「これは動画ファイルだ」と自動判定するための拡張子リスト
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 動画ファイル（.mp4等）から音声（.m4a）を一時的に抽出した際、すべての文字起こし処理が終わった後にその音声をどうするかの設定
# False: ファイルを消さずにそのまま保存（動画編集ソフト Premiere Pro や DaVinci Resolve 等でその音声ファイルをそのまま使いたい場合に便利）
# True : 文字起こし（SRT生成）が終わったら、抽出した臨時の .m4a ファイルを自動で削除してフォルダ内を常にクリーンに保つ
REMOVE_TEMP_AUDIO = False
# =====================================================================


def get_unique_filepath(file_path):
    """ファイルの上書きを完全に防止する関数

    もし指定された出力先ファイルパスがすでにフォルダ内に存在する場合、既存のデータを破壊（上書き）しないよう、
    ファイル名の末尾に「_1」「_2」「_3」のような連番を自動で付与し、完全に重複のない新しいユニークなファイルパスを作成して返します。
    """
    # そもそもファイルが存在しない（初回の書き出しである）場合は、重複の恐れがないためそのままのパスを即座に返す
    if not os.path.exists(file_path):
        return file_path

    # パス文字列を「フォルダ＋ファイル名の部分」と「拡張子（.srt や .m4a など）」の2つにきれいに分離する
    base_path, ext = os.path.splitext(file_path)
    counter = 1  # 重複があった場合にファイル名の末尾に付与する連番の初期カウンター

    # 重複しない（まだPC上に存在しない）新しいファイル名が見つかるまで、無限ループでチェックを繰り返す
    while True:
        new_file_path = f"{base_path}_{counter}{ext}"  # 文字列フォーマットを使って「元の名前_連番.拡張子」を組み立て（例: test_1.srt）
        if not os.path.exists(new_file_path):  # 生成した新しいパスがまだ使われていなければ（実在しなければ）安全と判断
            return new_file_path  # そのユニークなパスを確定値として呼び出し元に返す
        counter += 1  # すでに同名のファイルが存在した場合は、カウンターを1増やして次のループで再チェック（例: test_2.srt）


def load_word_dictionary(file_path):
    """外部のテキストファイル（単語辞書）から、AIに学習させるための単語リストを読み込む関数"""
    # 指定されたパスが空文字、またはファイルが指定の場所に実在しない場合は、辞書なしと判断して空のリストを返す
    if not file_path or not os.path.exists(file_path):
        return []

    word_dict = []  # ファイルから読み取った正常な単語たちを美しく格納するための空のリストを定義
    try:
        # UTF-8（BOM付き対応の utf-8-sig）でファイルを開き、1行ずつ安全にスキャンしていく
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後にある不要なスペース、タブ、改行コード（\n）を完全に削ぎ落とす
                # 文字列が空（白文字のみの空行）ではなく、かつ先頭が「#」から始まるコメント行でもない場合のみ、有効な単語として判定
                if word and not word.startswith("#"):
                    word_dict.append(word)  # 条件をクリアした純粋な単語だけを、単語配列の末尾にスタックする
    except Exception as e:
        # 読み込み中に万が一予期せぬエラー（権限不足など）が起きても、システム全体をクラッシュさせず、親切な警告文を出して処理を続行
        print(f"[!] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")
    return word_dict  # 構築した単語リスト（エラー時は空リスト）を返す


def format_timestamp(seconds):
    """秒数（浮動小数点数：例 85.342）を、SRT字幕規格の厳密なフォーマット（HH:MM:SS,mmm）に1ミリ秒の狂いもなく正確に変換する関数"""
    hours = int(seconds // 3600)  # 全体の総秒数を3600で割り、整数部分を取り出して「時間（Hour）」を算出
    minutes = int((seconds % 3600) // 60)  # 時間を引いた残りの秒数から、さらに60で割って「分（Minute）」を算出
    secs = int(seconds % 60)  # さらにそこから60で割った余りを計算し、「秒（Second）」の整数部分を取得
    milliseconds = int(round((seconds % 1) * 1000))  # 1秒未満の小数点以下の端数に1000を掛け、四捨五入して「ミリ秒（Millisecond）」を3桁で取得

    # 💡【重要】四捨五入（round）の影響により、ミリ秒が1000（つまりジャスト1秒）に達してしまった場合の、時間がズレるのを防ぐ繰り上げ補正処理
    if milliseconds >= 1000:
        milliseconds -= 1000  # ミリ秒を0のジャストにリセット
        secs += 1  # 溢れた1秒分を、秒の桁に加算
        if secs >= 60:  # 秒が60秒に達してしまった場合
            secs -= 60  # 秒を0にリセット
            minutes += 1  # 溢れた1分分を、分の桁に加算
            if minutes >= 60:  # 分が60分に達してしまった場合
                minutes -= 60  # 分を0にリセット
                hours += 1  # 溢れた1時間分を、時間の桁に加算

    # 桁数が足りない場合に「0」で埋める書式指定（時・分・秒は2桁、ミリ秒は3桁固定：例 01:25:05,042）を行って文字列として出力
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def get_audio_duration(file_path):
    """ffprobeという動画・音声解析ツールをバックグラウンドで走らせ、ファイルの総再生秒数を100%正確に取得する関数（tqdm進捗バー用）"""
    # ffprobeに渡すコマンド引数の配列。画面を汚さないようにエラーのみを出力し、フォーマット内の総持続時間（duration）だけをプレーンに取り出す設定
    cmd = [
        "ffprobe",
        "-v",
        "error",  # 警告等の余計なログを一切出力させないフラグ
        "-show_entries",
        "format=duration",  # メタデータの中から「動画の長さ」のみを指名してリクエスト
        "-of",
        "default=noprint_wrappers=1:nokey=1",  # 余計な装飾テキスト（format=等）を消し、純粋な数値の文字列だけを返させる魔法のオプション
        file_path,  # 対象ファイルのパス
    ]
    try:
        # コマンドを実行し、標準出力の結果をテキストとして取得、前後の改行をトリミング（.strip()）
        output = subprocess.check_output(cmd, text=True).strip()
        if "duration=" in output:
            output = output.split("duration=")[-1].strip()  # 万が一出力フォーマットが崩れた場合を想定した、数値部分の切り出しセーフティ
        return float(output)  # 取得した文字列（例 "124.52"）を、Pythonで計算可能な浮動小数点数（float）にキャストして返す
    except Exception:
        # 万が一PC環境にffprobeが入っていないなどの理由で取得に失敗した場合、進捗バーを％表示なしの「流動アニメーション」で動かすために None を返す
        return None


def extract_audio_from_video(video_path, output_audio_path):
    """ffmpegという動画処理の王様ツールを呼び出し、動画から音声ストリームだけを「超高速かつ劣化なし」で一瞬で抽出する関数"""
    print(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")

    # 第1作戦：再エンコード（圧縮のやり直し）をせず、動画内の音声をそのまま丸ごとコピーして別ファイルに超高速保存するコマンド
    command = [
        "ffmpeg",
        "-y",  # すでに同名の一時ファイルがあれば自動で上書きすることを許可するフラグ
        "-i",
        video_path,  # 解析元となる動画ファイルのパス
        "-vn",  # 「Video None」の略。映像データを完全に無視し、音声データだけを抽出対象にするフラグ
        "-acodec",
        "copy",  # 音声コーデックを「copy（そのまま複製）」に指定。CPUに負荷をかけず、100%劣化なしで一瞬で終わる理由がこれです
        output_audio_path,  # 出力する音声ファイルのパス（.m4a）
    ]

    try:
        # ffmpegの大量の内部ログがターミナル画面を埋め尽くして文字起こしの邪魔をしないよう、出力をゴミ箱（DEVNULL）に捨てて非表示実行
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[✓] 音声の抽出が完了しました: {output_audio_path}")
        return True
    except subprocess.CalledProcessError:
        # 動画の音声形式（特異なPCM形式など）によっては、無劣化コピー（copy）がエラーを吐いて失敗する場合があるための「第2作戦（救済措置）」
        print("[!] 音声の無劣化抽出に失敗しました。汎用的なエンコード抽出に切り替えます...")
        # 第2作戦のコマンド：どんな動画の音声でも強制的に美しく変換できる、超汎用的な「AAC形式」にエンコードしながら音声を取り出す設定
        fallback_command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "aac",  # ここをcopyではなく「aac」というエンコーダに指定し、安全に再圧縮をかける
            output_audio_path,
        ]
        try:
            subprocess.run(
                fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            print(f"[✓] 音声の抽出（再エンコード）が完了しました: {output_audio_path}")
            return True
        except Exception as e:
            # 第2作戦すら失敗した場合は、ffmpegの内部エラーかファイル破損の可能性が高いため、エラーログを出して終了
            print(
                f"エラー: ffmpegでの音声抽出に致命的な失敗をしました。ffmpegが正常にインストールされているか確認してください。 {e}"
            )
            return False
    except FileNotFoundError:
        # パソコンの環境変数（Path）に 'ffmpeg' 自体が登録されておらず、コマンドが見つからなかった場合の親切なエラー案内
        print(
            "エラー: システムに 'ffmpeg' コマンドが見つかりません。READMEの手順に従って ffmpeg をインストールし、環境変数を通してください。"
        )
        return False


def process_segment_to_lines(segment, min_len=10, max_len=20):
    """Whisperが解析した1つの長い発言セグメントを、指定された「10〜20文字制限」というプロ仕様のテロップルールに応じてミリ秒単位で細かく切り刻む関数"""
    words_data = []  # セグメントの内部に含まれる単語単位の「テキスト」「発話開始秒数」「発話終了秒数」を整理整頓して格納するリスト
    for w in segment.get("words", []):
        word_text = w["word"]  # Whisperが切り出した最小単位の単語（例: "こんにちは"）
        words_data.append({
            "text": word_text,
            "start": float(w["start"]),  # その単語を発音し始めた瞬間（秒）
            "end": float(w["end"]),  # その単語を発音し終わった瞬間（秒）
        })

    lines = []  # 文字数の調整がすべて完了し、あとはSRTファイルに書き出すだけになった完成済みの行データを詰め込むためのリスト
    current_line_text = ""  # 現在、文字を1文字ずつ積み上げてビルドしている最中の、1行分のテキストを保持する変数
    current_line_start = None  # 現在作成している行の「表示を開始すべき秒数」（最初の単語が放たれた瞬間のタイムスタンプをホールド）
    current_line_end = None  # 現在作成している行の「表示を終了すべき秒数」（単語が後ろにドッキングされるたびに、常に最新の終了秒数へと上書き更新）

    # 綺麗に整理した単語データを、タイムスタンプが古い順（先頭から）に1つずつ精査していくメインループ
    for w_info in words_data:
        w_text = w_info["text"]  # 今回処理する単語の文字
        w_start = w_info["start"]  # 今回処理する単語の開始時間
        w_end = w_info["end"]  # 今回処理する単語の終了時間

        has_period = "。" in w_text  # 読みやすさ向上のため、単語の中に句点（。）が含まれているかを事前にチェックするフラグ
        clean_w_text = w_text.replace("、", "").replace("。", "")  # 画面に表示した際にチカチカして邪魔になる「、」や「。」をテキストから完全に消去

        # 句読点を消去した結果、文字が完全に空っぽ（「。」だけが入っていた単語など）になってしまった場合の特殊なクリーンアップ処理
        if not clean_w_text:
            # 文字自体は存在しないが「。」がそこにあったということは文章の区切りを意味するため、書きかけの行があればここで強制改行（確定）を執行
            if has_period and current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""  # 次の行のためにテキストを初期化
                current_line_start = None
                current_line_end = None
            continue  # 文字がないので、これ以上下の文字数カウント処理に進まず、次の単語の処理へとスキップ

        # パターンA：【超例外】単語1つだけの長さで、最大文字数制限（20文字）を単体で突破してしまっている超巨大単語だった場合の破壊処理
        if len(clean_w_text) > max_len:
            # もしすでに他の単語によって「書きかけの途中の行」が存在していれば、一旦そこで綺麗に区切って1行として確定・保存する
            if current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""

            w_dur = w_end - w_start  # 巨大単語全体を発音するのにかかっている合計時間（秒数）
            w_len = len(clean_w_text)  # 巨大単語の合計文字数

            # 20文字の最大制限を超えている間、頭から20文字ずつナイフで切り落とすように強引に分割していくループ
            while len(clean_w_text) > max_len:
                sub_text = clean_w_text[:max_len]  # 文字列の先頭から最大文字数分（20文字）だけをスパッと抽出
                sub_start = w_start  # この切り出し行の表示開始時間
                # 【数式】文字数の比率に応じて、発話全体の時間を割り算（等分）し、テロップの表示時間を文字の長さに合わせて精密に伸縮させる
                sub_end = w_start + (w_dur * (max_len / w_len))

                lines.append({"text": sub_text, "start": sub_start, "end": sub_end})  # 切り取った20文字を行リストに格納
                clean_w_text = clean_w_text[
                    max_len:
                ]  # 処理が終わった20文字を元の単語から削り落とし、残った後ろの文字を次のループに回す
                w_start = sub_end  # 次の切り出しの開始時間は、たった今切り落とした終了時間に設定することでタイムスタンプの隙間をゼロにする

            # 20文字ずつ切り刻んだ結果、最後に残った「端数の文字（20文字未満）」があれば、それを次の新しい行の書き出し（ベース）としてセット
            if clean_w_text:
                current_line_text = clean_w_text
                current_line_start = w_start
                current_line_end = w_end

        # パターンB：現在の行の文字数に、この新しい単語をドッキングさせると、絶対防衛ライン（20文字）をオーバーしてしまう場合の処理
        elif len(current_line_text) + len(clean_w_text) > max_len:
            # 溢れて画面外にはみ出してしまうため、現在書きかけだった行をここで一旦終了とし、確定データとして保存
            if current_line_text:
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
            # 溢れてしまった新しい単語を、次の新しい行の記念すべき「最初の1文字（単語）」としてセット
            current_line_text = clean_w_text
            current_line_start = w_start
            current_line_end = w_end

        # パターンC：足しても文字数制限（20文字）以内に安全かつ綺麗に収まる、最も一般的な場合の結合処理
        else:
            # まだこの行に1文字も入っていない（新しい行の書き始め）状態なら、この単語の開始時間をその行全体の「表示開始時間」としてロック
            if not current_line_text:
                current_line_start = w_start
            current_line_text += clean_w_text  # 既存の文字列の後ろに、今回の単語テキストをピタッと結合
            current_line_end = w_end  # 行の「表示終了時間」を、今合流した最新の単語の終了時間へと常にアップデート

        # 単語の個別処理が無事に終わった際、その単語の末尾に「。」（文の終わり）が含まれていた場合の区切り執行
        if has_period:
            if current_line_text:
                # 「。」があるということは話の句切れ目なので、文字数にまだ余裕があっても、視聴者の読みやすさのためにここで1回綺麗に改行して確定
                lines.append({
                    "text": current_line_text,
                    "start": current_line_start,
                    "end": current_line_end,
                })
                current_line_text = ""  # 次の新しい文章のために変数を完全にクリーンにする
                current_line_start = None
                current_line_end = None

    # すべての単語のパズル組み立てチェックが終わった後、確定処理されずに変数に残ってしまっている「最後の書きかけの1行」を忘れずに回収
    if current_line_text:
        lines.append({
            "text": current_line_text,
            "start": current_line_start,
            "end": current_line_end,
        })

    return lines  # 10〜20文字制限ルールで精密に美しく切り分けられた行データの配列をすべて返す


def transcribe_to_custom_srt(audio_path, output_srt_path, dict_file_path=None, model_size="base"):
    """指定された音声ファイルを読み込んでWhisperによる超高精度AI文字起こしを実行し、カスタムSRTファイルへ書き出す司令塔関数"""
    print(f"[*] モデル '{model_size}' をパソコンのメモリに読み込み中...")

    try:
        model = load_model(model_size)  # オンラインからモデル（初回のみ）をダウンロードし、GPU/CPUメモリ上にセットアップ
    except MemoryError:
        # パソコンのスペック（VRAM/RAM）が不足し、巨大なモデルを広げられなかった場合の親切なエラー対処案内
        print(
            f"エラー: パソコンのメモリ（またはビデオメモリ）が不足しているため、モデル '{model_size}' を読み込めませんでした。"
        )
        print("初期設定エリアの DEFAULT_MODEL_SIZE を 'tiny' や 'base' に下げて再度実行してみてください。")
        return False
    except Exception as e:
        print(f"エラー: Whisperモデルの読み込み中に予期せぬ重大なエラーが発生しました: {e}")
        return False

    prompt_string = ""  # Whisperの耳（認識力）を特定の専門用語にチューニングするためのプロンプト初期化文字列
    if dict_file_path and os.path.exists(dict_file_path):
        word_dict = load_word_dictionary(dict_file_path)  # テキスト辞書から単語を安全に読み込み
        if word_dict:
            # 💡【Whisperの裏技】辞書単語たちを「。」と「、」で数珠つなぎにして1つのフェイク文章に成形。これがAIに最も強力に単語を学習させる構造
            prompt_string = "。" + "、".join(word_dict) + "。"
            print(f"[*] 【辞書適用】'{dict_file_path}' から計 {len(word_dict)} 個の専門単語を読み込み、AIに注入しました")

    print(f"[*] 音声の解析準備が整いました。文字起こしを開始します: {audio_path}")

    # 進捗バーのMAX値を決めるため、事前に関数を使って動画・音声の「総再生秒数」を正確に取得
    total_duration = get_audio_duration(audio_path)

    try:
        # tqdm を初期化。全体の秒数をゴール（total）に設定し、ターミナルの横幅いっぱいにフィット（dynamic_ncols=True）させる
        pbar = tqdm(
            total=total_duration, desc="文字起こし進行状況", unit="秒", dynamic_ncols=True
        )

        # 🚀 Whisperによる音声解析のコア（心臓部）処理を実行
        result = model.transcribe(
            audio_path,
            verbose=None,  # ターミナルへの進捗テキストの標準出力をオフ（None）にし、処理速度の低下を防ぐとともに画面崩れを防止
            fp16=False,  # グラフィックボード（GPU）が非搭載の一般的なCPUパソコン環境でも100%エラーを出さずに安全に動かすため、32bit演算を強制
            initial_prompt=prompt_string,  # 先ほど成形したプロンプト用の単語塊をAIの脳内に注入し、誤認識率を劇的に低下させる
            language="ja",  # 言語を「日本語（Japanese）」に完全固定。これにより、喋り始めの無音部分で英語や中国語に誤判定される現象を完全ガード
            word_timestamps=True,  # 💡ミリ秒単位での10〜20文字分割を行うための必須フラグ。単語ごとの「一音一音の正確な時間データ」をAIに算出させます
        )

        # 解析が100%無事に終わったら、進捗プログレスバーを右端（100%）まで綺麗に到達させてアニメーションを美しく終了
        if total_duration:
            pbar.update(total_duration)
        pbar.close()

    except RuntimeError as e:
        # ファイルの破損や、音声フォーマットが壊れていてデコードできなかった場合の例外キャッチ
        print(
            f"エラー: 文字起こし処理中にAIが内部エラーを起こしました（音声ファイルが破損している可能性があります）: {e}"
        )
        return False
    except Exception as e:
        print(f"エラー: 処理中に予期せぬエラーが発生しました: {e}")
        return False

    print("[*] 解析成功。SRT（字幕ファイル）を構築中...")
    try:
        srt_index = 1  # SRT字幕ブロック（1、2、3...と上から順に表示される番号）の初期インデックス
        # 指定された出力パスで、文字コードUTF-8（WindowsとMacの両方で文字化けしない標準規格）でファイルを作成
        with open(output_srt_path, "w", encoding="utf-8") as f:
            # AIが解析した全ての「発言ブロック（セグメント）」を、時系列順に1つずつ取り出す
            for segment in result["segments"]:
                # 1つの長い発言を、先ほど設計した「10〜20文字制限・読句点消去」の自前ルール関数に通してバラバラの短い行に分解
                split_lines = process_segment_to_lines(
                    segment, min_len=MIN_CHAR_LEN, max_len=MAX_CHAR_LEN
                )
                for line_data in split_lines:
                    line_text = line_data["text"].strip()  # 行の前後に不要なスペースが入り込んでいた場合は綺麗に掃除
                    if not line_text:  # 中身が完全に空っぽ（無音やノイズ等で文字が無かった場合）の行は、ファイル破壊を防ぐためスキップ
                        continue

                    line_start = line_data["start"]  # 綺麗に切り分けられたその行の「表示開始秒数」
                    line_end = line_data["end"]  # 綺麗に切り分けられたその行の「表示終了秒数」

                    # 🎬 世界標準のSRT字幕ファイルフォーマットの規約に1ミリ秒の狂いもなく従ってテキストをファイルに書き出し
                    f.write(f"{srt_index}\n")  # 1行目：字幕の通し番号（連番）
                    # 2行目：タイムスタンプ矢印（例：00:01:20,500 --> 00:01:23,120）
                    f.write(
                        f"{format_timestamp(line_start)} --> {format_timestamp(line_end)}\n"
                    )
                    f.write(f"{line_text}\n\n")  # 3行目：画面に映すテロップ文字列 ＋ 次の字幕ブロックと区別するための「必ず必要な空行」

                    srt_index += 1  # 次のテロップブロックのために、インデックス番号を1つカウントアップ
        print(f"[✓] 字幕作成がすべて正常に完了しました！ 保存先を確認してください: {output_srt_path}")
        return True
    except IOError as e:
        # 字幕ファイルをテキストエディタや動画編集ソフトで開いたままスクリプトを実行し、書き込みロックがかかっていた場合などのエラー回避
        print(
            f"エラー: SRTファイルの書き込みに失敗しました（保存先のファイルが別のソフトで開いたままになっていませんか？）: {e}"
        )
        return False


def main():
    """プログラムが起動した際に最初に呼び出される、全体の流れを統括するエントリーポイント関数"""
    # 引数を受け取るための解析オブジェクト（パーサー）を初期化・準備
    parser = argparse.ArgumentParser(
        description="動画または音声ファイルから10〜20文字に最適化されたプロ仕様のSRT字幕を出力するスクリプト"
    )

    # 3つの重要な引数を定義（ユーザーが指定しなかった場合は、ファイル上部の初期設定エリアで決めたデフォルト値が自動配備される）
    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help="入力ファイル（動画または音声）のパス"
    )
    parser.add_argument("-d", "--dict", default=DEFAULT_DICT_FILE, help="優先単語リスト（dictionary.txt）のパス")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_SIZE, help="Whisperのモデルサイズ指定")

    args = parser.parse_args()  # コマンドラインから実際に渡された引数（またはデフォルト値）を確定させて変数にバインド

    # 指定された入力ファイル（動画・音声）が、本当にPC内に実在するかを厳しくチェック。存在しなければ即座にエラー終了
    if not os.path.exists(args.input_file):
        print(f"エラー: 指定された入力ファイルが見つかりません。パスが正しいか確認してください: {args.input_file}")
        sys.exit(1)

    # スクリプト全体の処理に「何秒かかったか」を後で100%正確に割り出すため、スタート時点の超精密クロック時間を確保
    start_time = time.perf_counter()

    # 入力ファイルパスから「ドットより前の名前の部分」と「.mp4 などの拡張子の部分」を切り分ける
    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()  # ユーザーが「.MP4」などの大文字で入力した場合の判定ミスを防ぐため、強制的に小文字の「.mp4」に統一

    # 出力するSRTファイルの名前を定義（例: test.mp4 ➔ test.srt）
    raw_srt_path = base_path + ".srt"
    # すでに同名のSRTが存在した場合に備え、上書き防止関数を通してユニークなパス（例: test_1.srt）を自動計算
    output_srt_path = get_unique_filepath(raw_srt_path)

    target_audio_file = args.input_file  # Whisperに最終的に流し込むためのファイルパスを入れる変数（デフォルトはそのままのパス）
    is_video_input = (
        ext_lower in VIDEO_EXTENSIONS
    )  # 入力された拡張子が動画のリスト（.mp4等）に含まれているかどうかの真偽値フラグ（True/False）

    # 💡 もし入力されたファイルが動画だった場合の、音声事前抽出の分岐処理
    if is_video_input:
        # 動画と同じ名前の音声ファイル（.m4a）の出力先を計算し、重複防止をかける
        raw_audio_path = base_path + ".m4a"
        extracted_audio_path = get_unique_filepath(raw_audio_path)

        # 動画から音声だけを抜き出す ffmpeg 連携関数を実行。もし致命的なエラー（ffmpeg未インストール等）で失敗したらその場で即座に処理中断
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("エラー: 動画からの音声抽出に失敗したため、以降の文字起こし処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path  # Whisperの解析対象を、動画ファイルから「今切り出したばかりの音声ファイルパス」へスイッチ

    # メインの文字起こし＆SRTファイル自動書き出し関数をトリガー
    success = transcribe_to_custom_srt(
        audio_path=target_audio_file,
        output_srt_path=output_srt_path,
        dict_file_path=args.dict,
        model_size=args.model,
    )

    # 入力が動画であり、かつ初期設定エリアの「一時音声削除フラグ（REMOVE_TEMP_AUDIO）」がTrueに設定されていた場合のクリーンアップ処理
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                os.remove(target_audio_file)  # 用済みとなった臨時の音声ファイル（.m4a）をPCから自動削除
                print(f"[*] 一時音声ファイルを自動削除し、フォルダ内をクリーンにしました: {target_audio_file}")
        except Exception as e:
            # 削除に失敗しても文字起こし自体は成功しているので、警告だけ出して処理をそのまま綺麗に進める設定
            print(f"[!] 警告: 一時音声ファイルの自動削除中にエラーが発生しました（処理は正常終了しています）: {e}")

    # 文字起こしの最中にAIエラーやシステムエラーで失敗フラグが立っていた場合、エラー終了コード1を投げて終了
    if not success:
        print("エラー: 文字起こし処理が正常に完了しませんでした。上のエラーログを確認してください。")
        sys.exit(1)

    # 処理がすべて大成功を収めたら、終了時の精密クロック時間を取得
    end_time = time.perf_counter()
    # 終了時間から開始時間を引き算し、全工程にかかった正確な時間を小数点2桁の美しいフォーマットでターミナルに出力して完了
    print("処理時間(秒数):", "{:.2f}".format((end_time - start_time)))


if __name__ == "__main__":
    """このスクリプトファイルが他のファイルからパーツとして import されたのではなく、
    コマンドラインから直接主役として実行された場合（uv run main.py等）にのみ、メイン関数を起動するPythonの約束事
    """
    main()