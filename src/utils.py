"""
utils.py
プロジェクト全体で共通して使用する汎用ユーティリティ関数を定義するモジュール。

特定の工程（Whisper・LLM・SRT）に依存しない、ファイル読み書き・パス操作・
テキスト保存などの横断的な処理をここに集約します。
各専門モジュール（transcriber.py / refiner.py / formatter.py）は
ファイル操作が必要な場合、このモジュールの関数を呼び出します。
"""

import json  # セグメントの生データを JSON 形式で保存・読み込みするためのライブラリ
import os    # ファイルパスの結合・存在確認・ディレクトリ作成など、OS 依存のファイル操作を行うライブラリ
from tqdm import tqdm  # tqdm.write を使って、進捗バーを破壊せずにログを出力するためのライブラリ

from src.exceptions import FileReadError, FileWriteError
from src.config import OUTPUT_SRT_DIR, OUTPUT_TRANSCRIPT_DIR, OUTPUT_TEXT_DIR, OUTPUT_AUDIO_DIR


# ------------------------------------------------------------------
# パス操作
# ------------------------------------------------------------------

def get_unique_filepath(file_path: str) -> str:
    """ファイルの上書きを完全に防止する関数。

    指定された出力先ファイルパスがすでに存在する場合、既存データを破壊しないよう
    ファイル名の末尾に「_1」「_2」「_3」のような連番を自動付与してユニークなパスを返します。
    存在しないパスがそのまま返ってきた場合は、新規作成として扱えます。

    Args:
        file_path: 重複チェックしたいファイルパス。

    Returns:
        重複しないことが保証されたファイルパス。
    """
    if not os.path.exists(file_path):
        return file_path

    base_path, ext = os.path.splitext(file_path)
    counter = 1

    # カウンターをインクリメントしながら、使われていない番号を探す
    while True:
        new_file_path = f"{base_path}_{counter}{ext}"
        if not os.path.exists(new_file_path):
            return new_file_path
        counter += 1


def ensure_directory(dir_path: str) -> None:
    """指定したディレクトリが存在しない場合に作成する関数。

    出力先ディレクトリを事前に確保するために使用します。
    すでに存在する場合は何もしません（exist_ok=True）。

    Args:
        dir_path: 作成したいディレクトリのパス。

    Raises:
        FileWriteError: ディレクトリの作成に失敗した場合。
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        raise FileWriteError(f"ディレクトリの作成に失敗しました: {dir_path} / 原因: {e}") from e


# ------------------------------------------------------------------
# テキストファイルの読み書き
# ------------------------------------------------------------------

def read_text_file(file_path: str) -> str:
    """テキストファイルを読み込んで文字列として返す関数。

    BOM 付き UTF-8（utf-8-sig）にも対応しています。
    Windows のメモ帳などで保存したファイルに BOM が付く場合があるため、
    utf-8-sig を使うことで BOM を自動的に除去します。

    Args:
        file_path: 読み込みたいファイルのパス。

    Returns:
        ファイルの内容（文字列）。末尾の空白・改行は strip() で除去されます。

    Raises:
        FileReadError: ファイルが存在しない、または読み込みに失敗した場合。
    """
    if not os.path.exists(file_path):
        raise FileReadError(f"ファイルが見つかりません: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return f.read().strip()
    except OSError as e:
        raise FileReadError(f"ファイルの読み込みに失敗しました: {file_path} / 原因: {e}") from e


def read_lines_file(file_path: str) -> list[str]:
    """テキストファイルを行ごとに読み込んで、有効な行のリストを返す関数。

    空行と「#」で始まるコメント行を自動的に除外します。
    辞書ファイル・フィラーリストなど、1 行 1 エントリ形式のファイルに使用します。

    Args:
        file_path: 読み込みたいファイルのパス。

    Returns:
        コメント・空行を除いた有効な行の文字列リスト。

    Raises:
        FileReadError: ファイルが存在しない、または読み込みに失敗した場合。
    """
    if not os.path.exists(file_path):
        raise FileReadError(f"ファイルが見つかりません: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                stripped = line.strip()  # 行前後の空白・改行を除去
                if stripped and not stripped.startswith("#"):  # 空行とコメント行を除外
                    lines.append(stripped)
        return lines
    except OSError as e:
        raise FileReadError(f"ファイルの読み込みに失敗しました: {file_path} / 原因: {e}") from e


def write_text_file(file_path: str, content: str) -> None:
    """テキストファイルにコンテンツを書き込む関数。

    ファイルが存在しない場合は新規作成します。
    存在する場合は上書きします（呼び出し元で get_unique_filepath を使って
    上書き防止をしてから呼び出すことを推奨します）。

    Args:
        file_path: 書き込み先ファイルのパス。
        content: 書き込む文字列。

    Raises:
        FileWriteError: ファイルへの書き込みに失敗した場合。
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        raise FileWriteError(f"ファイルへの書き込みに失敗しました: {file_path} / 原因: {e}") from e


# ------------------------------------------------------------------
# セグメントデータの保存（Whisper / LLM 共通）
# ------------------------------------------------------------------

def save_segments_as_json(segments: list, file_path: str) -> None:
    """セグメントデータを JSON 形式でファイルに保存する関数。

    Whisper が出力した生のセグメントデータ（タイムスタンプ・単語単位データを含む）を
    そのまま JSON として保存します。後からデバッグや再処理に使用できます。

    Args:
        segments: Whisper が出力したセグメントのリスト。各要素は辞書形式です。
        file_path: 保存先のファイルパス。

    Raises:
        FileWriteError: ディレクトリ作成またはファイル書き込みに失敗した場合。
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise FileWriteError(f"JSON の書き込みに失敗しました: {file_path} / 原因: {e}") from e


def save_segments_as_plaintext(segments: list, file_path: str) -> None:
    """セグメントデータからテキストのみを抽出して、改行区切りのプレーンテキストとして保存する関数。

    タイムスタンプや単語単位データを除いた、人間が読みやすい形式で保存します。
    Whisper 生テキスト確認・LLM 校正結果の確認などに使用します。

    Args:
        segments: セグメントのリスト。各要素は "text" キーを持つ辞書形式です。
        file_path: 保存先のファイルパス。

    Raises:
        FileWriteError: ディレクトリ作成またはファイル書き込みに失敗した場合。
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        lines = [seg.get("text", "").strip() + "\n" for seg in segments]
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except OSError as e:
        raise FileWriteError(f"プレーンテキストの書き込みに失敗しました: {file_path} / 原因: {e}") from e


# ------------------------------------------------------------------
# 出力パスの生成（命名規則の一元管理）
# ------------------------------------------------------------------

def build_output_paths(input_file_path: str) -> dict:
    """出力ファイル群のパスを命名規則に従って一括生成する関数。

    引数には入力ファイル（音声・動画）のパスを渡します。
    stem（ファイル名の拡張子なし部分）を入力ファイル名から取り出し、
    各出力ファイルのパスを生成します。

    上書き防止のため、同名ファイルが存在する場合は末尾に _1, _2 を自動付与します。

    命名規則:
        - Whisper SRT  : output/srt/<stem>_whisper.srt
        - Whisper JSON : output/transcript/<stem>_whisper.json
        - Whisper TEXT : output/text/<stem>_whisper.txt
        - refined SRT  : output/srt/<stem>_refined.srt
        - refined JSON : output/transcript/<stem>_refined.json
        - refined TEXT : output/text/<stem>_refined.txt
        - 抽出音声     : output/audio/<stem>_extracted.wav

    Args:
        input_file_path: 入力ファイル（音声・動画）のパス。
                         例: "./input/lecture.mp4" → stem = "lecture"

    Returns:
        各出力ファイルのパスを格納した辞書。

    Raises:
        FileWriteError: 出力ディレクトリの作成に失敗した場合。
    """
    # 入力ファイル名から拡張子を除いた stem を取り出す
    # 例: "./input/lecture.mp4" → stem = "lecture"
    # pipeline.py 側で連番付きのパスを作っていた旧設計と異なり、
    # ここでは純粋な入力ファイル名だけを使う。上書き防止は後段の get_unique_filepath() に任せる。
    stem = os.path.splitext(os.path.basename(input_file_path))[0]

    # 出力ディレクトリが存在しない場合は作成する
    for target_dir in [OUTPUT_SRT_DIR, OUTPUT_TRANSCRIPT_DIR, OUTPUT_TEXT_DIR, OUTPUT_AUDIO_DIR]:
        ensure_directory(target_dir)

    # 各出力ファイルのベースパスを生成する（まだ上書き防止処理はしていない）
    base_paths = {
        "whisper_srt":      os.path.join(OUTPUT_SRT_DIR,        f"{stem}_whisper.srt"),
        "whisper_json":     os.path.join(OUTPUT_TRANSCRIPT_DIR, f"{stem}_whisper.json"),
        "whisper_txt":      os.path.join(OUTPUT_TEXT_DIR,       f"{stem}_whisper.txt"),
        "refined_srt":      os.path.join(OUTPUT_SRT_DIR,        f"{stem}_refined.srt"),
        "refined_json":     os.path.join(OUTPUT_TRANSCRIPT_DIR, f"{stem}_refined.json"),
        "refined_txt":      os.path.join(OUTPUT_TEXT_DIR,       f"{stem}_refined.txt"),
        "extracted_audio":  os.path.join(OUTPUT_AUDIO_DIR,      f"{stem}_audio.m4a"),
    }

    # 同名ファイルが存在する場合に末尾に _1, _2 を付与してユニークなパスにする
    # whisper と refined のファイルは同一 stem から派生するため、
    # どちらか一方のパスだけを get_unique_filepath() で決めて連番を統一する方針もあるが、
    # ここでは各ファイル独立して連番を付与するシンプルな設計にしている。
    paths = {key: get_unique_filepath(path) for key, path in base_paths.items()}

    return paths