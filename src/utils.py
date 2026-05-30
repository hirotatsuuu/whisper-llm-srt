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
from src.config import OUTPUT_SRT_DIR, OUTPUT_TRANSCRIPT_DIR, OUTPUT_TEXT_DIR


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

def build_output_paths(output_srt_path: str) -> dict:
    """出力ファイル群のパスを命名規則に従って一括生成する関数。

    出力先ディレクトリと命名規則をこの関数に集約することで、
    将来的に命名規則を変更する際にここだけを修正すれば済むようにしています。

    命名規則:
        - Whisper SRT     : output/srt/<stem>_whisper.srt
        - Whisper JSON    : output/transcript/<stem>_whisper.json
        - Whisper TEXT    : output/text/<stem>_whisper.txt
        - Elyza SRT       : output/srt/<stem>__refined.srt
        - Elyza JSON      : output/transcript/<stem>_refined.json
        - Elyza TEXT      : output/elyza/<stem>_refined.txt

    Args:
        output_srt_path: LLM 校正版 SRT の出力パス（get_unique_filepath で確定済みのもの）。

    Returns:
        各出力ファイルのパスを格納した辞書。キーは以下の通りです。
            "elyza_srt"     : LLM 校正版 SRT（メイン出力）
            "elyza_json"    : LLM 生データ JSON
            "elyza_txt"     : LLM 校正後のプレーンテキスト
            "whisper_srt"   : Whisper 生データ SRT
            "whisper_json"  : Whisper 生データ JSON
            "whisper_txt"   : Whisper 生データ プレーンテキスト
    """
    # ファイル名の stem（拡張子なし）を取り出す（例: "test_2"）
    stem = os.path.splitext(os.path.basename(output_srt_path))[0]

    # 指定された3つのディレクトリが存在しない場合は、ここで自動的に新規作成します
    for target_dir in [OUTPUT_SRT_DIR, OUTPUT_TRANSCRIPT_DIR, OUTPUT_TEXT_DIR]:
        os.makedirs(target_dir, exist_ok=True)

    # 既存の paths 辞書定義は触らず残し、直下で今回の指定ディレクトリ・ファイル名規則に基づき上書き定義します
    paths = {
        "whisper_srt":  os.path.join(OUTPUT_SRT_DIR,        f"{stem}_whisper.srt"),
        "whisper_json": os.path.join(OUTPUT_TRANSCRIPT_DIR, f"{stem}_whisper.json"),
        "whisper_txt":  os.path.join(OUTPUT_TEXT_DIR,       f"{stem}_whisper.txt"),
        "refined_srt":  os.path.join(OUTPUT_SRT_DIR,        f"{stem}_refined.srt"),
        "refined_json": os.path.join(OUTPUT_TRANSCRIPT_DIR, f"{stem}_refined.json"),
        "refined_txt":  os.path.join(OUTPUT_TEXT_DIR,       f"{stem}_refined.txt"),
    }

    # 同名ファイルの上書きを防止するため、既存の get_unique_filepath を適用して末尾に _1, _2 を自動付与
    for key in paths:
        paths[key] = get_unique_filepath(paths[key])

    # 元のロジック（一部変数作成など）はそのまま通過させ、最終的に上書き済みの辞書を返却
    return paths