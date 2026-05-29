import argparse  # コマンドライン引数（ファイル名・モデルサイズ等）を受け取って解析するための標準ライブラリ
import os        # ファイルパスの結合や存在確認など、OS依存のファイル操作を行うライブラリ
import sys       # 致命的なエラー発生時にプログラムを即座に強制終了（sys.exit）させるためのライブラリ
import time      # 処理時間を小数点2桁まで精密に計測するためのライブラリ
import copy      # LLM校正前にWhisper生データをディープコピーして退避するための標準ライブラリ
from tqdm import tqdm  # 全体の処理進捗をターミナル上にアニメーションバーで表示するためのライブラリ

# 3つの専門モジュールから必要な関数をインポート
from src.formatter import get_unique_filepath, write_srt_file
from src.llm_checker import refine_context_with_llm
from src.transcriber import clean_fillers_keep_timing, load_filler_list, load_word_dictionary, run_whisper_transcribe

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE  = "./data/test.m4a"        # 引数なしで実行した際に自動で読み込まれる既定のファイル
DEFAULT_DICT_FILE   = "./data/dictionary.txt"  # 優先的に認識させたい固有名詞・専門用語のテキストファイル
DEFAULT_FILLER_FILE = "./data/filler.txt"      # 除去したいフィラー語（「えっと」「あのー」等）のテキストファイル
DEFAULT_MODEL_SIZE  = "base"                   # Whisperのモデルサイズ（tiny/base/small/medium/large等）

# 動画ファイルとして認識する拡張子のリスト（これらに該当する場合のみ自動で音声を抽出します）
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 処理完了後に、動画から一時的に抽出した音声ファイル（.m4a）を自動削除するかどうか（True=削除, False=残す）
REMOVE_TEMP_AUDIO = False

# 全体進捗バーで管理する5つの工程ラベル
PIPELINE_STEPS = [
    "辞書・フィラー読み込み",
    "Whisper 文字起こし",
    "フィラー除去",
    "LLM 校正",
    "SRT 書き出し",
]
# =====================================================================


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

    # 指定されたファイルが実在するか確認
    if not os.path.exists(args.input_file):
        print(f"[*] エラー: 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    start_time = time.perf_counter()  # 総処理時間の計測開始
    print("[*] whisper-llm-srt を起動しました")

    # 入力ファイルの拡張子を解析して動画か音声かを判別
    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    # 【上書き防止】通常版のユニークな出力パスをまず確定させる（例: test_2.srt）
    # その後、同じ連番ベースで _whisper.srt を派生させることで両ファイルが衝突しない
    output_srt_path = get_unique_filepath(base_path + ".srt")
    normal_srt_base, _ = os.path.splitext(output_srt_path)
    output_whisper_srt_path = normal_srt_base + "_whisper.srt"

    target_audio_file = args.input_file
    is_video_input = ext_lower in VIDEO_EXTENSIONS

    # もし動画ファイルだった場合、音声（.m4a）をバックグラウンドで抽出してから処理する
    if is_video_input:
        from src.transcriber import extract_audio_from_video
        extracted_audio_path = get_unique_filepath(base_path + ".m4a")
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("[*] エラー: 動画からの音声抽出に失敗したため、処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path

    # -----------------------------------------------------------------
    # 【一本道データライン】
    # 進捗バーは「純粋なバー＋進捗数字」のみのシンプルな表示にしています。
    # 各工程内のログは tqdm.write() で出力し、バーの表示を破壊しません。
    # -----------------------------------------------------------------
    custom_format = "{percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}]"

    with tqdm(
        total=len(PIPELINE_STEPS),
        bar_format=custom_format,
        dynamic_ncols=False,  # バーが横いっぱいに伸びてチカチカするのを防ぐ
        unit="step",
    ) as pbar:

        # 【工程 1/5】専門用語辞書とフィラーリストの読み込み
        word_dict   = load_word_dictionary(args.dict)
        filler_list = load_filler_list(args.filler)
        pbar.update(1)

        # 【工程 2/5】Whisperによる音声解析と文字起こし
        raw_segments, whisper_elapsed = run_whisper_transcribe(
            audio_path=target_audio_file,
            word_dict=word_dict,
            model_size=args.model,
        )
        tqdm.write(f"[*] Whisper処理時間: {whisper_elapsed:.2f} 秒")
        pbar.update(1)

        if not raw_segments:
            tqdm.write("[*] エラー: Whisperによる音声解析に失敗したか、データが空です。処理を中断します。")
            sys.exit(1)

        # 【工程 3/5】フィラー（えっと・あの等）をタイムスタンプを維持したまま空文字に置換
        cleaned_segments = clean_fillers_keep_timing(raw_segments, filler_list)
        pbar.update(1)

        # LLMが書き換える前の「フィラー除去済み生データ」をディープコピーして退避
        # これがWhisper限定版（校正前）のSRT出力データになります
        whisper_only_segments = copy.deepcopy(cleaned_segments)

        # 【工程 4/5】LLMが前後の文脈をもとにバッチ校正
        if args.no_llm:
            # --no-llm 指定時はLLM校正をスキップし、フィラー除去済み生データをそのまま使う
            tqdm.write("[*] --no-llm が指定されたため、LLM校正工程をスキップします。")
            llm_refined_segments = copy.deepcopy(whisper_only_segments)
        else:
            # ※ あらすじ・辞書は渡しません。前後TEXTの文脈のみで自然な日本語に補正させます。
            llm_refined_segments, llm_elapsed = refine_context_with_llm(cleaned_segments)
            tqdm.write(f"[*] LLM処理時間: {llm_elapsed:.2f} 秒")
        pbar.update(1)

        # 【工程 5/5】BudouXによる10〜20文字カット ＋ SRTファイルへの書き出し
        if args.no_llm:
            # --no-llm 時は通常版パスに生データ字幕のみを出力
            tqdm.write("[*] SRT書き出し中（通常版のみ）...")
            success_normal  = write_srt_file(llm_refined_segments, output_srt_path)
            success_whisper = True  # Whisper版はスキップのためTrue扱い
        else:
            # 通常版（LLM校正後）と、比較用のWhisper版（校正前）を両方出力
            tqdm.write("[*] SRT書き出し中（通常版）...")
            success_normal  = write_srt_file(llm_refined_segments, output_srt_path)
            tqdm.write("[*] SRT書き出し中（校正前Whisper版）...")
            success_whisper = write_srt_file(whisper_only_segments, output_whisper_srt_path)

        success = success_normal and success_whisper
        pbar.update(1)

    # -----------------------------------------------------------------
    # 【後始末クリーンアップ】
    # -----------------------------------------------------------------
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                os.remove(target_audio_file)  # 一時音声ファイルを削除
                print(f"[*] 一時音声ファイルを削除しました: {target_audio_file}")
        except Exception as e:
            print(f"[*] 警告: 一時音声ファイルの削除中にエラーが発生しました: {e}")

    if not success:
        print("[*] エラー: SRTファイルの書き出しに失敗しました。")
        sys.exit(1)

    # 成果物のパスと総処理時間をまとめて表示
    if args.no_llm:
        print(f"[+] 【出力ファイル】{output_srt_path}")
    else:
        print(f"[+] 【通常版（LLM校正後）】{output_srt_path}")
        print(f"[+] 【比較版（校正前Whisper）】{output_whisper_srt_path}")
    print(f"[*] 総処理時間: {time.perf_counter() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()