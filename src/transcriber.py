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
    ファイルが存在しない場合は空

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

    Whisper の処理進捗を把握するための補助的な情報取得に使います。
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
        # 取得できなくても処理は続行できるため、例外は握りつぶさず None を返す
        return None


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """ffmpeg を呼び出し、動画ファイルから音声ストリームだけを抽出する関数。

    まず無劣化コピー（-acodec copy）を試みます。
    コーデックの非互換等で失敗した場合は AAC 再エンコードにフォールバックします。

    Args:
        video_path: 入力動画ファイルのパス。
        output_audio_path: 抽出した音声の出力パス（.m4a 等）。

    Raises:
        FfmpegNotFoundError: ffmpeg がシステムにインストールされていない場合。
        AudioExtractionError: 無劣化・再エンコードともに音声抽出に失敗した場合。
    """
    tqdm.write(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")

    command = [
        "ffmpeg",
        "-y", "-i", video_path,
        "-vn", "-acodec", "copy",  # 音声ストリームを無劣化でコピー
        output_audio_path,
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        tqdm.write(f"[*] 音声の抽出が完了しました: {output_audio_path}")
        return

    except subprocess.CalledProcessError:
        # 無劣化コピーに失敗した場合、AAC エンコードにフォールバックして再試行する
        tqdm.write("[*] 音声の無劣化抽出に失敗しました。AAC エンコード抽出に切り替えます...")
        fallback_command = [
            "ffmpeg",
            "-y", "-i", video_path,
            "-vn", "-acodec", "aac",
            output_audio_path,
        ]
        try:
            subprocess.run(fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            tqdm.write(f"[*] 音声の抽出（AAC 再エンコード）が完了しました: {output_audio_path}")
            return
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(
                f"動画からの音声抽出に失敗しました（無劣化・再エンコードともに失敗）: {video_path}"
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
            fp16=False,            # GPU 非搭載環境でも動くように 32bit 演算を強制
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