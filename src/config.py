# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================

# --- 入力ファイル ---
DEFAULT_AUDIO_FILE  = "./data/test.m4a"        # 引数なしで実行した際に自動で読み込まれる既定のファイル
DEFAULT_DICT_FILE   = "./data/dictionary.txt"  # 優先的に認識させたい固有名詞・専門用語のテキストファイル
DEFAULT_FILLER_FILE = "./data/filler.txt"      # 除去したいフィラー語（「えっと」「あのー」等）のテキストファイル

# --- Whisper ---
DEFAULT_MODEL_SIZE  = "base"                   # Whisperのモデルサイズ（tiny/base/small/medium/large等）

# --- LLM ---
# 将来モデルを変更したい場合や、軽量モデル（gemma2:2bなど）を試したい場合はここを書き換えてください。
LLM_MODEL_NAME  = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"

# LLMへのプロンプトテンプレートファイルのパス
# ファイルの末尾に【補正対象データ】を自動で結合して使います。
# ファイルが存在しない場合は、下記のDEFAULT_PROMPTにフォールバックします。
LLM_PROMPT_FILE = "./data/llm_refine_prompt_template.txt"

# --- 字幕レイアウト ---
MIN_CHAR_LEN = 10  # 1行の最低文字数。これより短い場合は極力次の単語と結合させます
MAX_CHAR_LEN = 20  # 1行の最大文字数。YouTubeやTikTokのテロップとして最も見やすい20文字を絶対上限とします

# --- 動作設定 ---
# 処理完了後に、動画から一時的に抽出した音声ファイル（.m4a）を自動削除するかどうか（True=削除, False=残す）
REMOVE_TEMP_AUDIO = False

# 動画ファイルとして認識する拡張子のリスト（これらに該当する場合のみ自動で音声を抽出します）
VIDEO_EXTENSIONS  = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]