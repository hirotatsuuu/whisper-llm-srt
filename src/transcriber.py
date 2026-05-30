"""
transcriber.py
【第1工程：耳】音声・動画ファイルの処理を担当するモジュール。

責務:
  - 動画ファイルからの音声抽出（ffmpeg）
  - Whisper による音声の文字起こし（セグメント・単語タイムスタンプ付き）
  - フィラー語の除去（タイムスタンプを維持したまま空文字に置換）
  - 辞書・フィラーリストのファイル読み込み

ファイルの読み書きは src/utils.py の関数を経由します。
例外は src/exceptions.py で定義したカスタム例外として送出します。
"""

import os          # ファイルパスの判定・存在チェックに使うライブラリ
import subprocess  # ffmpeg/ffprobe といった外部プログラムをバックグラウンドから呼び出すためのライブラリ
import time        # Whisper 処理時間を小数点 2 桁まで計測するためのライブラリ
from whisper import load_model  # OpenAI Whisper のモデルをローカルにロードするための関数
from tqdm import tqdm  # tqdm.write を使って進捗バーを破壊せずにログを出力するためのライブラリ

from src.exceptions import (
    AudioExtractionError,
    FfmpegNotFoundError,
    FileReadError,
    WhisperModelLoadError,
    WhisperTranscribeError,
)
from src.utils import read_lines_file


# ------------------------------------------------------------------
# 辞書・フィラーリストの読み込み
# ------------------------------------------------------------------

def load_word_dictionary(file_path: str) -> list[str]:
    """外部テキストファイル（単語辞書）から、Whisper の初期プロンプト用の単語リストを読み込む関数。

    登録された単語は Whisper の initial_prompt に埋め込まれ、認識精度の向上に使われます。
    ファイルが存在しない場合は警告を出して空リストを返します（辞書なしで処理続行）。

    Args:
        file_path: 読み込みたい辞書ファイルのパス。

    Returns:
        辞書に登録された単語の文字列リスト。ファイルが存在しない場合は空リスト。
    """
    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"[*] 注意: 単語辞書ファイルが見つかりません: {file_path}")
        return []

    try:
        word_dict = read_lines_file(file_path)
        dict_filename = os.path.basename(file_path)
        tqdm.write(f"[*] 単語辞書 [{dict_filename}] を読み込みました（登録数: {len(word_dict)} 語）")
        return word_dict
    except FileReadError as e:
        # 辞書は必須ではないため、読み込みエラーは警告にとどめて処理を継続する
        tqdm.write(f"[*] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")
        return []


def load_filler_list(file_path: str) -> list[str]:
    """外部テキストファイル（フィラーリスト）から、除去対象の口癖・フィラー語を読み込む関数。

    dictionary.txt と同じ書式（1 行 1 語、# でコメント）に対応しています。
    ファイルが存在しない場合は空を返します。

    Args:
        file_path: 読み込みたいフィラーリストファイルのパス。

    Returns:
        フィラー語の文字列リスト。ファイルが存在しない場合は空。
    """
    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"[*] 注意: フィラーリストが見つかりません: {file_path} ")
        return []

    try:
        filler_list = read_lines_file(file_path)
    except FileReadError as e:
        tqdm.write(f"[*] 警告: フィラーリストの読み込み中にエラーが発生しました: {e}")
        return []

    filler_filename = os.path.basename(file_path)

    # ファイルはあるが中身が空（コメント行のみ等）の場合もデフォルトに切り替える
    if not filler_list:
        tqdm.write(f"[*] 注意: フィラーリスト [{filler_filename}] が空です")
        return []

    tqdm.write(f"[*] フィラーリスト [{filler_filename}] を読み込みました（登録数: {len(filler_list)} 語）")
    return filler_list


# ------------------------------------------------------------------
# 動画からの音声抽出
# ------------------------------------------------------------------

def get_audio_duration(file_path: str) -> float | None:
    """ffprobe を使ってファイルの総再生秒数を取得する関数。

    文字起こし開始前に音声の長さをログ表示するために使います。
    取得に失敗した場合は None を返します（None の場合でも処理は続行可能です）。

    Args:
        file_path: 再生秒数を取得したい音声・動画ファイルのパス。

    Returns:
        総再生秒数（float）。取得に失敗した場合は None。
    """
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
        # ffprobe が見つからない・対応外フォーマット等は警告せず None を返す
        # 呼び出し元で None チェックしてから表示の有無を判断する
        return None


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """ffmpeg を呼び出し、動画ファイルから音声ストリームだけを抽出する関数。

    無劣化コピー（-acodec copy）は動画の音声コーデックによって
    コンテナとの組み合わせが不正になり、壊れたファイルが生成される場合があるため使用しない。
    常に AAC 再エンコードで出力することで、どんなコーデックの動画でも安定して抽出できる。

    Whisper は 16kHz モノラルで処理するため、サンプリングレートとチャンネル数を
    ここで統一しておくことで Whisper 内部の再デコード負荷を最小限にする。

    Args:
        video_path:        入力動画ファイルのパス。
        output_audio_path: 抽出した音声の出力パス（.m4a）。

    Raises:
        FfmpegNotFoundError: ffmpeg がシステムにインストールされていない場合。
        AudioExtractionError: 音声抽出に失敗した場合。
    """
    tqdm.write(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")

    command = [
        "ffmpeg",
        "-y",                   # 出力ファイルが存在する場合は上書きする
        "-i", video_path,       # 入力ファイル
        "-vn",                  # 映像ストリームを除外する
        "-acodec", "aac",       # AAC に再エンコードする（コーデック不問で安定して動く）
        "-ar", "16000",         # サンプリングレートを 16kHz に統一（Whisper の推奨値）
        "-ac", "1",             # モノラルに変換（Whisper はモノラルで処理する）
        "-b:a", "128k",         # ビットレートを 128kbps に指定（音質と容量のバランス）
        output_audio_path,      # 出力先（.m4a）
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,   # stderr を取得してエラー内容をログに残せるようにする
            check=True,
        )
        tqdm.write(f"[*] 音声の抽出が完了しました: {output_audio_path}")

    except subprocess.CalledProcessError as e:
        stderr_message = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else "不明"
        raise AudioExtractionError(
            f"動画からの音声抽出に失敗しました: {video_path}\n"
            f"ffmpeg エラー出力: {stderr_message}"
        ) from e

    except FileNotFoundError as e:
        raise FfmpegNotFoundError(
            "システムに 'ffmpeg' コマンドが見つかりません。"
            "winget install Gyan.FFmpeg でインストールしてください。"
        ) from e


# ------------------------------------------------------------------
# Whisper による文字起こし
# ------------------------------------------------------------------

def run_whisper_transcribe(
    audio_path: str,
    word_dict: list[str],
    model_size: str = "base",
) -> tuple[list, float]:
    """音声ファイルを読み込んで Whisper で文字起こしを実行し、セグメントデータと処理時間を返す関数。

    word_dict に登録された単語を initial_prompt に埋め込むことで、
    固有名詞や専門用語の認識精度を向上させます。

    進捗バーは pipeline.py 側の全体バーで管理するため、この関数内では表示しません。
    処理時間は呼び出し元（pipeline.py）で一元表示できるよう、タプルで返します。

    Args:
        audio_path: 文字起こしする音声ファイルのパス。
        word_dict: Whisper の initial_prompt に埋め込む単語リスト。
        model_size: 使用する Whisper モデルのサイズ（"tiny"/"base"/"small"/"medium"/"large"）。

    Returns:
        (segments, elapsed_time) のタプル。
            segments: Whisper が出力したセグメントのリスト。失敗時は空リスト。
            elapsed_time: Whisper 処理にかかった秒数（float）。失敗時は 0.0。

    Raises:
        WhisperModelLoadError: モデルのメモリへの読み込みに失敗した場合。
        WhisperTranscribeError: 文字起こし処理中に予期せぬエラーが発生した場合。
    """
    # 音声ファイルの存在とサイズを確認する。
    # 抽出に失敗して壊れた（空の）ファイルが渡された場合に早期検出できる。
    if not os.path.exists(audio_path):
        raise WhisperTranscribeError(
            f"音声ファイルが見つかりません: {audio_path}"
        )
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise WhisperTranscribeError(
            f"音声ファイルのサイズが 0 バイトです。音声抽出が正常に完了しなかった可能性があります: {audio_path}"
        )
    tqdm.write(f"[*] 音声ファイルを確認しました（サイズ: {file_size / 1024:.1f} KB）: {audio_path}")

    duration = get_audio_duration(audio_path)
    if duration is not None:
        # 秒数を「X 分 Y 秒」形式に変換して表示する
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        tqdm.write(f"[*] 音声の長さ: {minutes} 分 {seconds} 秒（{duration:.1f} 秒）")
    else:
        tqdm.write("[*] 音声の長さを取得できませんでした（処理は続行します）")
    
    # 辞書が登録されている場合は、Whisper の initial_prompt に単語リストを埋め込む
    # 句読点で囲むことで、Whisper が単語の区切りを誤認識するのを防ぐ
    prompt_string = ""
    if word_dict:
        prompt_string = "。" + "、".join(word_dict) + "。"

    tqdm.write(f"[*] Whisper モデル '{model_size}' をメモリに読み込み中...")
    try:
        model = load_model(model_size)
    except MemoryError as e:
        raise WhisperModelLoadError(
            f"メモリ不足のため、モデル '{model_size}' を読み込めませんでした。"
            "より小さいモデルサイズ（tiny / base）を試してください。"
        ) from e
    except Exception as e:
        raise WhisperModelLoadError(
            f"Whisper モデル '{model_size}' の読み込み中に予期せぬエラーが発生しました: {e}"
        ) from e

    tqdm.write(f"[*] 文字起こしを開始します: {audio_path}")

    try:
        whisper_start_time = time.perf_counter()  # Whisper 処理時間の計測開始

        result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=False,            # CPUの場合 32bit 演算を強制、GPUの場合はTrueにすると高速化できる
            initial_prompt=prompt_string,
            language="ja",         # 日本語に固定
            word_timestamps=True,  # 単語単位のタイムスタンプを取得（後続処理で必須）
        )

        whisper_elapsed_time = time.perf_counter() - whisper_start_time  # 計測終了

        return result.get("segments", []), whisper_elapsed_time

    except Exception as e:
        raise WhisperTranscribeError(
            f"Whisper 文字起こし処理中に予期せぬエラーが発生しました: {e}"
        ) from e


# ------------------------------------------------------------------
# フィラー除去
# ------------------------------------------------------------------

def clean_fillers_keep_timing(segments: list, filler_list: list[str]) -> list:
    """タイムスタンプを維持したまま、フィラー（口癖）だけを空文字に置換するゴミ出し関数。

    「文字を詰める」のではなく「空文字に置換する」ことで、
    フィラーが占めていた時間の箱（タイムスタンプ）をそのまま残します。
    これにより、後続の formatter.py での字幕タイミングが音声とズレるのを 100% 防止します。

    処理は 2 段階で行います:
      1. seg["text"]（セグメント全体テキスト）からフィラーを文字列置換で除去
      2. seg["words"]（単語単位リスト）の中でフィラーに一致する単語を空文字に差し替え

    Args:
        segments: Whisper が出力したセグメントのリスト。
        filler_list: 除去するフィラー語の文字列リスト。

    Returns:
        フィラーを空文字に置換したセグメントのリスト。
    """

    if not filler_list:
        tqdm.write("[*] フィラーリストが空のため、フィラー除去処理をスキップします。")
        return segments  # segments をそのまま返す（コピーせず参照渡し。変更しないため問題なし）
    
    tqdm.write("[*] タイムスタンプ維持型フィラー除去を実行中...")
    cleaned_segments = []

    for seg in segments:
        # 1. セグメント全体テキスト（seg["text"]）からフィラーを文字列置換で除去する
        seg_text = seg.get("text", "")
        for filler in filler_list:
            seg_text = seg_text.replace(filler, "")

        # 2. 単語単位（words）のタイムスタンプ配列を精査し、フィラー単語のみ空文字に差し替える
        #    タイムスタンプ（start/end）はそのまま維持し、テキストだけを空文字化する
        cleaned_words = []
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            if word_text in filler_list:
                w["word"] = ""  # フィラー単語を空文字化（タイムスタンプは一切変更しない）
            cleaned_words.append(w)

        # 上書きして次の LLM 工程（refiner.py）に渡す
        seg["text"] = seg_text
        seg["words"] = cleaned_words
        cleaned_segments.append(seg)

    return cleaned_segments