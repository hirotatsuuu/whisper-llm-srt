import argparse  # コマンドライン引数（ファイル名・モデルサイズ等）を受け取って解析するための標準ライブラリ
from src.pipeline import run  #

# 設定ファイルからデフォルト値を参照する
from src.config import DEFAULT_AUDIO_FILE, DEFAULT_DICT_FILE, DEFAULT_FILLER_FILE, DEFAULT_MODEL_SIZE

def main():
    
    # スクリプトの実行時引数の受付を設定
    parser = argparse.ArgumentParser(
        description="動画または音声ファイルからローカルLLMで校正され、10〜20文字に最適化されたSRT字幕を出力するスクリプト"
    )
    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help="入力ファイル（動画または音声）のパス"
    )
    parser.add_argument("-d", "--dict",   default=DEFAULT_DICT_FILE,   help="優先単語リスト（dictionary.txt）のパス")
    parser.add_argument("-f", "--filler", default=DEFAULT_FILLER_FILE, help="フィラーリスト（filler.txt）のパス")
    parser.add_argument("-m", "--model",  default=DEFAULT_MODEL_SIZE,  help="Whisperのモデルサイズ指定")
    # --no-llm をつけて実行するとLLM校正をスキップし、Whisper生データのまま書き出します
    parser.add_argument("--no-llm", action="store_true", help="LLMによる文脈校正工程をスキップする")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()