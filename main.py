import argparse  # コマンドライン引数（ファイル名・モデルサイズ等）を受け取って解析するための標準ライブラリ
import os        # ファイルパスの結合や存在確認など、OS依存のファイル操作を行うライブラリ
import sys       # 致命的なエラー発生時にプログラムを即座に強制終了（sys.exit）させるためのライブラリ
import time      # 処理時間を小数点2桁まで精密に計測するためのライブラリ
from tqdm import tqdm  # 全体の処理進捗をターミナル上にアニメーションバーで表示するためのライブラリ

# 3つの専門モジュールから必要な関数をインポート
from src.formatter import get_unique_filepath, write_srt_file
from src.llm_checker import refine_context_with_llm
from src.transcriber import clean_fillers_keep_timing, load_filler_list, load_word_dictionary, run_whisper_transcribe

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE = "./data/test.m4a"        # 引数なしで実行した際に自動で読み込まれる既定のファイル
DEFAULT_DICT_FILE  = "./data/dictionary.txt"  # 優先的に認識させたい固有名詞・専門用語のテキストファイル
DEFAULT_FILLER_FILE = "./data/filler.txt"     # 除去したいフィラー語（「えっと」「あのー」等）のテキストファイル
DEFAULT_MODEL_SIZE = "base"                   # Whisperのモデルサイズ（tiny/base/small/medium/large）

# スクリプトが「これは動画ファイルだ」と自動判定するための拡張子リスト
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 動画ファイルから音声（.m4a）を一時抽出した際、処理終了後に削除するか（True = 自動削除 / False = 残す）
REMOVE_TEMP_AUDIO = False

# 全体進捗バーの工程ラベル（表示順に定義）。各工程完了後に pbar.update(1) で1ステップ進む
PIPELINE_STEPS = [
    "辞書・フィラー読み込み",
    "Whisper 文字起こし",
    "フィラー除去",
    "LLM 校正",
    "SRT 書き出し",
]
# =====================================================================


def main():
    """プログラムが起動した際に最初に呼び出される、全体の流れを統括するメイン関数"""
    parser = argparse.ArgumentParser(
        description="動画または音声ファイルからローカルLLMで校正され、10〜20文字に最適化されたSRT字幕を出力するスクリプト"
    )
    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help="入力ファイル（動画または音声）のパス"
    )
    parser.add_argument("-d", "--dict",   default=DEFAULT_DICT_FILE,   help="優先単語リスト（dictionary.txt）のパス")
    parser.add_argument("-f", "--filler", default=DEFAULT_FILLER_FILE, help="フィラーリスト（filler.txt）のパス")
    parser.add_argument("-m", "--model",  default=DEFAULT_MODEL_SIZE,  help="Whisperのモデルサイズ指定")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"[*] エラー: 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    start_time = time.perf_counter()

    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    output_srt_path = get_unique_filepath(base_path + ".srt")

    target_audio_file = args.input_file
    is_video_input = ext_lower in VIDEO_EXTENSIONS

    if is_video_input:
        from src.transcriber import extract_audio_from_video
        extracted_audio_path = get_unique_filepath(base_path + ".m4a")
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("[*] エラー: 動画からの音声抽出に失敗したため、処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path

    # -----------------------------------------------------------------
    # 【一本道データライン】全体進捗バーを起動し、工程完了のたびに1ステップ進める
    # -----------------------------------------------------------------

    with tqdm(total=len(PIPELINE_STEPS), desc="全体進捗", unit="工程", dynamic_ncols=True) as pbar:

        # 【工程 1】辞書・フィラーリストの読み込み
        pbar.set_description(f"[1/5] {PIPELINE_STEPS[0]}")
        word_dict   = load_word_dictionary(args.dict)
        filler_list = load_filler_list(args.filler)
        pbar.update(1)

        # 【工程 2】Whisperによる音声解析（単語タイムスタンプ付きで文字起こし）
        pbar.set_description(f"[2/5] {PIPELINE_STEPS[1]}")
        raw_segments, whisper_elapsed = run_whisper_transcribe(
            audio_path=target_audio_file,
            word_dict=word_dict,
            model_size=args.model,
        )
        pbar.update(1)

        if not raw_segments:
            print("[*] エラー: Whisperによる音声解析に失敗したか、データが空です。処理を中断します。")
            sys.exit(1)

        # 【工程 3】フィラー（えっと・あの等）をタイムスタンプを維持したまま空文字に置換
        pbar.set_description(f"[3/5] {PIPELINE_STEPS[2]}")
        cleaned_segments = clean_fillers_keep_timing(raw_segments, filler_list)
        pbar.update(1)

        # 【工程 4】LLMが前後の文脈をもとにバッチ校正
        # ※ あらすじ・辞書は渡しません。前後TEXTの文脈のみで自然な日本語に補正させます。
        pbar.set_description(f"[4/5] {PIPELINE_STEPS[3]}")
        llm_refined_segments = refine_context_with_llm(cleaned_segments)
        pbar.update(1)

        # 【工程 5】BudouXによる10〜20文字カット ＋ SRTファイルへの書き出し
        pbar.set_description(f"[5/5] {PIPELINE_STEPS[4]}")
        success = write_srt_file(llm_refined_segments, output_srt_path)
        pbar.update(1)

    # -----------------------------------------------------------------
    # 【後始末クリーンアップ】
    # -----------------------------------------------------------------
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                os.remove(target_audio_file)
                print(f"[*] 一時音声ファイルを削除しました: {target_audio_file}")
        except Exception as e:
            print(f"[*] 警告: 一時音声ファイルの削除中にエラーが発生しました: {e}")

    if not success:
        print("[*] エラー: SRTファイルの書き出しに失敗しました。上のエラーログを確認してください。")
        sys.exit(1)

    # 処理時間のまとめ出力（LLM処理時間は llm_checker 内で出力済み）
    print(f"[*] Whisper処理時間: {whisper_elapsed:.2f} 秒")
    print(f"[*] 総処理時間: {time.perf_counter() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()