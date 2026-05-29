# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================

# --- 入力ファイル ---
DEFAULT_AUDIO_FILE  = "./data/test.m4a"
DEFAULT_DICT_FILE   = "./data/dictionary.txt"
DEFAULT_FILLER_FILE = "./data/filler.txt"

# --- Whisper ---
DEFAULT_MODEL_SIZE = "base"

# --- LLM ---
LLM_MODEL_NAME  = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"
LLM_PROMPT_FILE = "./data/llm_refine_prompt_template.txt"

# --- 字幕レイアウト ---
MIN_CHAR_LEN = 10
MAX_CHAR_LEN = 20

# --- 動作設定 ---
REMOVE_TEMP_AUDIO = False
VIDEO_EXTENSIONS  = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]