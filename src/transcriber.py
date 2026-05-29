import os  # ファイルパスの判定や存在チェックなど、OS依存のファイル操作を行うライブラリ
import subprocess  # ffmpegやffprobeといった外部プログラムをバックグラウンドから安全に呼び出して実行するためのライブラリ
from whisper import load_model  # OpenAIのWhisperモデルをローカルにロードするための関数
from tqdm import tqdm  # tqdm.write を使って、進捗バーを破壊せずにログを出力するためのライブラリ
import time  # 処理時間を小数点2桁まで精密に計測するためのライブラリ


def load_word_dictionary(file_path):
    """外部のテキストファイル（単語辞書）から、Whisperの初期プロンプト用の単語リストを読み込む関数"""
    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"[*] 注意: 単語辞書ファイルが見つかりません: {file_path} (辞書なしで処理を続行します)")
        return []

    word_dict = []  # ファイルから読み取った単語を格納するリスト
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後のスペース・タブ・改行を除去
                if word and not word.startswith("#"):  # 空行とコメント行（#）を除外
                    word_dict.append(word)

        dict_filename = os.path.basename(file_path)
        tqdm.write(f"[*] 単語辞書 [{dict_filename}] を読み込みました（登録数: {len(word_dict)}語）")

    except Exception as e:
        tqdm.write(f"[*] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")

    return word_dict


def load_filler_list(file_path):
    """外部のテキストファイル（フィラーリスト）から、除去対象の口癖・フィラー語を読み込む関数。

    dictionary.txt と同じ書式（1行1語、#でコメント）に対応しています。
    ファイルが存在しない場合や中身が空の場合は、デフォルトのフィラーリストで代替します。
    """
    # フィラーファイルが見つからない・空の場合に使うフォールバック用デフォルトリスト
    DEFAULT_FILLERS = ["えっと", "あの", "あのー", "えー", "まあ", "そのー", "なんか", "うーん"]

    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"[*] 注意: フィラーリストファイルが見つかりません: {file_path} (デフォルトのフィラーリストで続行します)")
        return DEFAULT_FILLERS

    filler_list = []  # ファイルから読み取ったフィラー語を格納するリスト
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()  # 行の前後のスペース・タブ・改行を除去
                if word and not word.startswith("#"):  # 空行とコメント行（#）を除外
                    filler_list.append(word)

        filler_filename = os.path.basename(file_path)

        # ファイルはあるが中身が空（コメント行のみ等）の場合もデフォルトに切り替える
        if not filler_list:
            tqdm.write(f"[*] 注意: フィラーリストファイル [{filler_filename}] が空です（デフォルトのフィラーリストで続行します）")
            return DEFAULT_FILLERS

        tqdm.write(f"[*] フィラーリスト [{filler_filename}] を読み込みました（登録数: {len(filler_list)}語）")

    except Exception as e:
        tqdm.write(f"[*] 警告: フィラーリストの読み込み中にエラーが発生しました（デフォルトのフィラーリストで続行します）: {e}")
        return DEFAULT_FILLERS

    return filler_list


def get_audio_duration(file_path):
    """ffprobeを使ってファイルの総再生秒数を取得する関数（進捗バー用）"""
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
    """ffmpegを呼び出し、動画ファイルから音声ストリームだけを抽出する関数"""
    tqdm.write(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")

    command = [
        "ffmpeg",
        "-y", "-i", video_path,
        "-vn", "-acodec", "copy",  # 音声を無劣化でコピー
        output_audio_path,
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tqdm.write(f"[*] 音声の抽出が完了しました: {output_audio_path}")
        return True
    except subprocess.CalledProcessError:
        # 無劣化コピーに失敗した場合、AACエンコードにフォールバック
        tqdm.write("[*] 音声の無劣化抽出に失敗しました。汎用的なエンコード抽出に切り替えます...")
        fallback_command = [
            "ffmpeg",
            "-y", "-i", video_path,
            "-vn", "-acodec", "aac",
            output_audio_path,
        ]
        try:
            subprocess.run(fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            tqdm.write(f"[*] 音声の抽出（再エンコード）が完了しました: {output_audio_path}")
            return True
        except Exception as e:
            tqdm.write(f"[*] エラー: ffmpegでの音声抽出に致命的な失敗をしました: {e}")
            return False
    except FileNotFoundError:
        tqdm.write("[*] エラー: システムに 'ffmpeg' コマンドが見つかりません。")
        return False


def run_whisper_transcribe(audio_path, word_dict, model_size="base"):
    """音声ファイルを読み込んでWhisperで文字起こしを実行し、セグメントデータと処理時間を返す関数。

    ※ 進捗バーはmain側の全体バーで管理するため、ここでは出力しません。
       処理時間は呼び出し元で一元表示できるよう、elapsed_time を一緒に返します。
    """
    # 辞書が登録されている場合は、WhisperのinitialPromptに単語リストを埋め込んで認識精度を向上させる
    prompt_string = ""
    if word_dict:
        prompt_string = "。" + "、".join(word_dict) + "。"

    tqdm.write(f"[*] モデル '{model_size}' をメモリに読み込み中...")
    try:
        model = load_model(model_size)
    except MemoryError:
        tqdm.write(f"[*] エラー: メモリ不足のため、モデル '{model_size}' を読み込めませんでした。")
        return [], 0.0
    except Exception as e:
        tqdm.write(f"[*] エラー: Whisperモデルの読み込み中にエラーが発生しました: {e}")
        return [], 0.0

    tqdm.write(f"[*] 文字起こしを開始します: {audio_path}")

    try:
        whisper_start_time = time.perf_counter()  # Whisper処理時間の計測開始

        result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=False,         # GPU非搭載環境でも動くように32bit演算を強制
            initial_prompt=prompt_string,
            language="ja",      # 日本語に固定
            word_timestamps=True,  # ミリ秒単位の後続処理のために必須
        )

        whisper_elapsed_time = time.perf_counter() - whisper_start_time  # 計測終了

        return result.get("segments", []), whisper_elapsed_time

    except Exception as e:
        tqdm.write(f"[*] エラー: Whisper文字起こし処理中に予期せぬエラーが発生しました: {e}")
        return [], 0.0


def clean_fillers_keep_timing(segments, filler_list):
    """タイムスタンプを維持したまま、フィラー（口癖）だけを『空文字』に置換するゴミ出し関数。

    文字を詰めるのではなく空文字に置換するため、発話タイミング（秒数）が狂うのを100%防止します。
    除去対象のフィラー語は、外部ファイルから読み込んだ filler_list を使用します。
    """
    tqdm.write("[*] タイムスタンプ維持型フィラー除去を実行中...")
    cleaned_segments = []

    for seg in segments:
        # 1. セグメント全体のテキスト（seg["text"]）からフィラーを文字列置換で除去
        seg_text = seg.get("text", "")
        for filler in filler_list:
            seg_text = seg_text.replace(filler, "")

        # 2. 単語単位（words）のタイムスタンプ配列を精査し、フィラー単語のみ空文字に差し替える
        #    タイムスタンプ（start/end）はそのまま維持し、テキストだけを消す
        cleaned_words = []
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            if word_text in filler_list:
                w["word"] = ""  # フィラーを空文字化（タイムスタンプは維持）
            cleaned_words.append(w)

        # 上書きして次のLLM工程に渡す
        seg["text"] = seg_text
        seg["words"] = cleaned_words
        cleaned_segments.append(seg)

    return cleaned_segments