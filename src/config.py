"""
config.py
プロジェクト全体の設定値を一元管理するモジュール。

ユーザーがカスタマイズしたい値はすべてここに定義されています。
このファイルだけ編集すれば、全モジュールの動作を変更できます。
各モジュールは必要な値を「from src.config import ...」でインポートして使います。
"""

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================

# --- 入力ファイルのデフォルトパス ---
DEFAULT_AUDIO_FILE  = "./input/test.m4a"           # 引数なしで実行した際に自動で読み込まれる既定の音声・動画ファイル
DEFAULT_DICT_FILE   = "./resources/dictionary.txt" # 優先的に認識させたい固有名詞・専門用語のテキストファイル
DEFAULT_FILLER_FILE = "./resources/filler.txt"     # 除去したいフィラー語（「えっと」「あのー」等）のテキストファイル
DEFAULT_PROMPT_FILE = "./resources/prompt.txt"     # LLM 校正に使うプロンプトテンプレートのテキストファイル

# --- 出力ファイルのデフォルトパス ---
OUTPUT_SRT_DIR        = "./output/srt/"             # 生成された字幕ファイル(srt形式)
OUTPUT_TRANSCRIPT_DIR = "./output/transcript/"      # 生成されたwhisperの生データファイル(json形式)
OUTPUT_TEXT_DIR       = "./output/text/"            # 生成された全てのテキストをまとめたファイル(txt形式)
OUTPUT_AUDIO_DIR      = "./output/audio/"           # 動画から抽出した一時的な音声ファイル(wav形式)

# --- Whisper ---
DEFAULT_MODEL_SIZE = "base"  # モデルサイズ（速度優先: tiny/base、精度優先: small/medium/large）

# --- LLM（ELYZA 等） ---
LLM_MODEL_NAME  = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"  # 使用する Ollama モデル名
BATCH_SIZE_LLM  = 10  # LLM に一度に送る字幕セグメントの数（前後文脈を持たせる単位）

# --- テロップデザインルール（字幕の改行制御） ---
MIN_CHAR_LEN = 10   # 1 行あたりの最低文字数（これ未満の短い行は、極力前の行と結合する）
MAX_CHAR_LEN = 20   # 1 行あたりの最高文字数（これを超える場合は、BudouX の文節で美しく改行する）

# --- クリーンアップ設定 ---
REMOVE_TEMP_AUDIO = False  # 動画から抽出した一時的な wav 音声ファイルを、処理終了後に自動削除するかどうか

# =====================================================================
# システム固定エリア：ここは原則変更しないでください
# =====================================================================

# 動画ファイルの拡張子リスト（これらに該当する場合は自動で音声抽出処理を挟みます）
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"]