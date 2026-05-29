import argparse  # コマンドライン引数（ファイル名・モデルサイズ等）を受け取って解析するための標準ライブラリ
from src.pipeline import run  # パイプラインの一連の処理を実行する関数をインポート

# 設定ファイルからデフォルト値を参照する
from src.config import DEFAULT_AUDIO_FILE, DEFAULT_DICT_FILE, DEFAULT_FILLER_FILE, DEFAULT_MODEL_SIZE, LLM_PROMPT_FILE

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
    
    # 💡【機能拡張】実行時引数としてLLM用のプロンプトテンプレートのファイルパスを受け取れるように拡張
    parser.add_argument("-p", "--prompt", default=LLM_PROMPT_FILE,   help="LLM校正用プロンプトテンプレート（prompt.txt）のパス")
    
    # --no-llm をつけて実行するとLLM校正をスキップし、Whisper生データのまま書き出します
    parser.add_argument("--no-llm", action="store_true", help="LLMによる文脈校正工程をスキップする")

    args = parser.parse_args()

    # 引数オブジェクトをそのままパイプラインへ引き渡す
    run(args)

if __name__ == "__main__":
    main()