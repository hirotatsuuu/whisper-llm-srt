import argparse  # コマンドライン（ターミナル）から「ファイル名」や「モデルサイズ」などの設定引数を受け取って解析するための標準ライブラリ
import os  # ファイルパスの結合（os.path.join）や、指定したファイルが実在するかの確認（os.path.exists）など、OS依存のファイル操作を行うライブラリ
import sys  # システム固有の機能にアクセスし、致命的なエラーが発生した際にプログラムを途中で安全かつ即座に強制終了（sys.exit）させるためのライブラリ
import time  # 処理にかかった時間を「ミリ秒（小数点2桁）」単位まで精密に計測（time.perf_counter）し、パフォーマンスを評価するためのライブラリ

# 新しく分割した3つの専門家（モジュール）から必要な関数をインポート
from src.formatter import get_unique_filepath, write_srt_file
from src.llm_checker import generate_summary, refine_context_with_llm # llm_checker に美しく統一！
from src.transcriber import clean_fillers_keep_timing, load_word_dictionary, run_whisper_transcribe

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE = "./data/test.m4a"  # ターミナルで引数を何も指定せずに実行した際、自動的に検索・読み込みが行われる既定の音声ファイルパス
DEFAULT_DICT_FILE = "./data/dictionary.txt"  # 固有名詞、専門用語、業界用語、新語など、AIが誤認識しやすい単語を優先的に正しく認識させるためのテキストファイル
DEFAULT_MODEL_SIZE = "base"  # Whisperのモデルサイズ（速度優先のtiny/baseから、精度優先のsmall/medium/largeまで選択可能）

# スクリプトが「これは動画ファイルだ」と自動判定するための拡張子リスト
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 動画ファイル（.mp4等）から音声（.m4a）を一時的に抽出した際、すべての文字起こし処理終わった後にその音声をどうするかの設定
REMOVE_TEMP_AUDIO = False
# =====================================================================


def main():
    """プログラムが起動した際に最初に呼び出される、全体の流れを統括するメイン関数"""
    parser = argparse.ArgumentParser(
        description="動画または音声ファイルからローカルLLMで校正され、10〜20文字に最適化されたSRT字幕を出力するスクリプト"
    )

    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help="入力ファイル（動画または音声）のパス"
    )
    parser.add_argument("-d", "--dict", default=DEFAULT_DICT_FILE, help="優先単語リスト（dictionary.txt）のパス")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_SIZE, help="Whisperのモデルサイズ指定")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"[*] エラー: 指定された入力ファイルが見つかりません。パスが正しいか確認してください: {args.input_file}")
        sys.exit(1)

    start_time = time.perf_counter()

    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    raw_srt_path = base_path + ".srt"
    output_srt_path = get_unique_filepath(raw_srt_path)

    target_audio_file = args.input_file  
    is_video_input = ext_lower in VIDEO_EXTENSIONS

    if is_video_input:
        raw_audio_path = base_path + ".m4a"
        extracted_audio_path = get_unique_filepath(raw_audio_path)

        from src.transcriber import extract_audio_from_video
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("[*] エラー: 動画からの音声抽出に失敗したため、以降の文字起こし処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path  

    # -----------------------------------------------------------------
    # 🔥 【ここから時系列順の完璧な一本道データライン】
    # -----------------------------------------------------------------

    # 【工程 1-A】単語辞書（dictionary.txt）の読み込み
    word_dict = load_word_dictionary(args.dict)

    # 【工程 1-B】Whisperによる音声解析（自然な塊のまま、制限なしで耳に専念させる）
    raw_segments = run_whisper_transcribe(
        audio_path=target_audio_file,
        word_dict=word_dict,
        model_size=args.model
    )
    
    if not raw_segments:
        print("[*] エラー: Whisperによる音声解析に失敗したか、データが空です。処理を中断します。")
        sys.exit(1)

    # 【工程 1-C】タイムスタンプを守るため、フィラー（えっと、あの等）を「空文字」に置換
    cleaned_raw_segments = clean_fillers_keep_timing(raw_segments)

    # 【工程 2】全体像の把握 ＋ 専門辞書のインプット ＋ 文脈バッチ校正（頭脳：llm_checker）
    # 1. まず全体のあらすじを取得
    summary = generate_summary(cleaned_raw_segments)

    # あらすじの内容を出力
    print("-" * 30)
    print("【生成されたあらすじ（ELYZAが学習した文脈）】")
    print(summary)
    print("-" * 30)

    # 2. そのあらすじを渡してバッチ校正を実行
    llm_refined_segments = refine_context_with_llm(cleaned_raw_segments, word_dict, summary)

    # 【工程 3】[最終整形] BudouXによる10〜20文字カット ＋ SRTファイルへの書き出し
    success = write_srt_file(llm_refined_segments, output_srt_path)

    # -----------------------------------------------------------------
    # 【後始末クリーンアップ】
    # -----------------------------------------------------------------
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                os.remove(target_audio_file)
                print(f"[*] 一時音声ファイルを自動削除し、フォルダ内をクリーンにしました: {target_audio_file}")
        except Exception as e:
            print(f"[*] 警告: 一時音声ファイルの自動削除中にエラーが発生しました（処理は正常終了しています）: {e}")

    if not success:
        print("[*] エラー: 字幕ファイルの書き出しに失敗しました。上のエラーログを確認してください。")
        sys.exit(1)

    end_time = time.perf_counter()
    print("[*] 総処理時間(秒数):", "{:.2f}".format((end_time - start_time)))


if __name__ == "__main__":
    main()