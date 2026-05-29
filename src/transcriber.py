import os  # ファイルパスの判定や存在チェックなど、OS依存のファイル操作を行うライブラリ
import subprocess  # ffmpegやffprobeといった外部の強力なCUIプログラムを、Pythonのバックグラウンドから安全に呼び出して実行するためのライブラリ
from whisper import load_model  # OpenAIが開発した高性能音声認識AI「Whisper」の学習済みモデルを、ローカルPCにロードするための関数
import time  # 処理にかかった時間を「ミリ秒（小数点2桁）」単位まで精密に計測（time.perf_counter）し、パフォーマンスを評価するためのライブラリ


def load_word_dictionary(file_path):
    """外部のテキストファイル（単語辞書）から、AIに学習させるための単語リストを読み込む関数"""
    if not file_path or not os.path.exists(file_path):
        print(f"[*] 注意: 単語辞書ファイルが見つかりません: {file_path} (辞書なしで処理を続行します)")
        return []

    word_dict = []  # ファイルから読み取った正常な単語たちを格納するための空のリストを定義
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後にある不要なスペース、タブ、改行コードを完全に削ぎ落とす
                if word and not word.startswith("#"):
                    word_dict.append(word)  # 条件をクリアした純粋な単語だけを、単語配列の末尾にスタック

        dict_filename = os.path.basename(file_path)
        print(f"\n[*] 情報: 単語辞書 [{dict_filename}] を読み込みました（登録数: {len(word_dict)}語）")

    except Exception as e:
        print(f"[*] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")

    return word_dict


def load_filler_list(file_path):
    """外部のテキストファイル（フィラーリスト）から、除去対象の口癖・フィラー語を読み込む関数。

    dictionary.txt と同じ書式（1行1語、#でコメント）に対応しています。
    ファイルが存在しない場合や中身が空の場合は、あらかじめ定義したデフォルトのフィラーリストで代替します。
    """
    # フィラーファイルが未指定・見つからない場合に使う安全網（フォールバック用のデフォルトリスト）
    DEFAULT_FILLERS = ["えっと", "あの", "あのー", "えー", "まあ", "そのー", "なんか", "うーん"]

    if not file_path or not os.path.exists(file_path):
        print(f"[*] 注意: フィラーリストファイルが見つかりません: {file_path} (デフォルトのフィラーリストで処理を続行します)")
        return DEFAULT_FILLERS

    filler_list = []  # ファイルから読み取った正常なフィラー語たちを格納するための空のリストを定義
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後にある不要なスペース、タブ、改行コードを完全に削ぎ落とす
                if word and not word.startswith("#"):
                    filler_list.append(word)  # 条件をクリアした純粋なフィラー語だけを、リストの末尾にスタック

        filler_filename = os.path.basename(file_path)

        # ファイルはあったが中身が空（コメント行しかなかった等）の場合もデフォルトに切り替える
        if not filler_list:
            print(f"[*] 注意: フィラーリストファイル [{filler_filename}] が空です。(デフォルトのフィラーリストで処理を続行します)")
            return DEFAULT_FILLERS

        print(f"[*] 情報: フィラーリスト [{filler_filename}] を読み込みました（登録数: {len(filler_list)}語）")

    except Exception as e:
        print(f"[*] 警告: フィラーリストの読み込み中にエラーが発生しました（デフォルトのフィラーリストで処理を続行します）: {e}")
        return DEFAULT_FILLERS

    return filler_list


def get_audio_duration(file_path):
    """ffprobeという動画・音声解析ツールをバックグラウンドで走らせ、ファイルの総再生秒数を正確に取得する関数（tqdm進捗バー用）"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if "duration=" in output:
            output = output.split("duration=")[-1].strip()
        return float(output)
    except Exception:
        return None


def extract_audio_from_video(video_path, output_audio_path):
    """ffmpegという動画処理ツールを呼び出し、動画から音声ストリームだけを抽出する関数"""
    print(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")

    command = [
        "ffmpeg",
        "-y", "-i", video_path,
        "-vn", "-acodec", "copy",
        output_audio_path,
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[*] 音声の抽出が完了しました: {output_audio_path}")
        return True
    except subprocess.CalledProcessError:
        print("[[*] 音声の無劣化抽出に失敗しました。汎用的なエンコード抽出に切り替えます...")
        fallback_command = [
            "ffmpeg",
            "-y", "-i", video_path,
            "-vn", "-acodec", "aac",
            output_audio_path,
        ]
        try:
            subprocess.run(fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[*] 音声の抽出（再エンコード）が完了しました: {output_audio_path}")
            return True
        except Exception as e:
            print(f"[*] エラー: ffmpegでの音声抽出に致命的な失敗をしました。 {e}")
            return False
    except FileNotFoundError:
        print("[*] エラー: システムに 'ffmpeg' コマンドが見つかりません。")
        return False


def run_whisper_transcribe(audio_path, word_dict, model_size="base"):
    """指定された音声ファイルを読み込んでWhisperによる文字起こしを実行し、生のセグメントデータを返す関数（最大文字数制限はかけない）。

    ※ 進捗バーはmain側の全体バーで管理するため、ここでは出力しません。
       処理時間は呼び出し元で表示できるよう elapsed_time を一緒に返します。
    """
    prompt_string = ""
    if word_dict:
        prompt_string = "。" + "、".join(word_dict) + "。"

    print(f"[*] モデル '{model_size}' をパソコンのメモリに読み込み中...")
    try:
        model = load_model(model_size)
    except MemoryError:
        print(f"[*] エラー: パソコンのメモリ不足のため、モデル '{model_size}' を読み込めませんでした。")
        return [], 0.0
    except Exception as e:
        print(f"[*] エラー: Whisperモデルの読み込み中にエラーが発生しました: {e}")
        return [], 0.0

    print(f"[*] 音声の解析準備が整いました。文字起こしを開始します: {audio_path}")

    try:
        # Whisper単体の処理時間計測開始
        whisper_start_time = time.perf_counter()

        result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=False,  # GPU非搭載環境でも動くように32bit演算を強制
            initial_prompt=prompt_string,
            language="ja",  # 日本語に固定
            word_timestamps=True,  # ミリ秒単位での後続処理のための必須フラグ
        )

        # Whisper単体の処理時間計測終了
        whisper_elapsed_time = time.perf_counter() - whisper_start_time

        return result.get("segments", []), whisper_elapsed_time

    except Exception as e:
        print(f"[*] エラー: Whisper文字起こし処理中に予期せぬエラーが発生しました: {e}")
        return [], 0.0


def clean_fillers_keep_timing(segments, filler_list):
    """【あなたのアイデア】タイムスタンプを維持したまま、言葉のヒゲ（フィラー）だけを『空文字』に駆逐するゴミ出し関数。

    文字を詰めるのではなく空文字に置換するため、発話のタイミング情報（秒数）が狂うのを100%防止します。
    除去対象のフィラー語は、外部から受け取った filler_list を使用します（dictionary.txt と同じ設計パターン）。
    """

    print("\n[*] 情報: タイムスタンプ維持型のフィラー（口癖）除去を実行中...")
    cleaned_segments = []

    for seg in segments:
        # 1. セグメント全体のテキストに対するフィラー置換
        seg_text = seg.get("text", "")
        for filler in filler_list:
            seg_text = seg_text.replace(filler, "")

        # 2. 【重要】単語単位（words）のタイムスタンプ配列を精査し、フィラー単語のみを『空文字』に差し替える
        cleaned_words = []
        for w in seg.get("words", []):
            word_text = w["word"].strip()

            # もし単語がフィラーリストに含まれていたら、時間はそのままにテキストだけを空文字化
            if word_text in filler_list:
                w["word"] = ""

            cleaned_words.append(w)

        # データを上書きして、次のLLM工程に引き渡す綺麗なセグメントを作成
        seg["text"] = seg_text
        seg["words"] = cleaned_words
        cleaned_segments.append(seg)

    return cleaned_segments