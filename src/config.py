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
DEFAULT_AUDIO_FILE  = "./data/test.m4a"           # 引数なしで実行した際に自動で読み込まれる既定の音声・動画ファイル
DEFAULT_DICT_FILE   = "./resources/dictionary.txt" # 優先的に認識させたい固有名詞・専門用語のテキストファイル
DEFAULT_FILLER_FILE = "./resources/filler.txt"     # 除去したいフィラー語（「えっと」「あのー」等）のテキストファイル
DEFAULT_PROMPT_FILE = "./resources/prompt.txt"     # LLM 校正に使うプロンプトテンプレートのテキストファイル

# --- Whisper ---
DEFAULT_MODEL_SIZE = "base"  # モデルサイズ（速度優先: tiny/base、精度優先: small/medium/large）

# --- LLM（ELYZA 等） ---
LLM_MODEL_NAME  = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"  # 使用する Ollama モデル名
BATCH_SIZE_LLM  = 10  # LLM に一度に送るセグメント数の上限（大きいほど処理は速いが、応答精度が下がることがある）

# --- 字幕レイアウト ---
MIN_CHAR_LEN = 10  # 1 行の最低文字数（これより短い末尾行は直前行への結合を試みます）
MAX_CHAR_LEN = 20  # 1 行の最大文字数（絶対にこの文字数を超えないようにガードします）

# --- 動作設定 ---
REMOVE_TEMP_AUDIO = False  # 動画から抽出した一時音声ファイルを処理後に削除するか（True=削除 / False=残す）
VIDEO_EXTENSIONS  = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]  # 動画と判定する拡張子リスト

# --- フィラーリストのデフォルト値 ---
# resources/filler.txt が見つからない場合にフォールバックとして使用されるリスト
DEFAULT_FILLERS = ["えっと", "あの", "あのー", "えー", "まあ", "そのー", "なんか", "うーん"]