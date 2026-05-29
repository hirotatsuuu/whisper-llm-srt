import argparse  # コマンドライン引数（ファイル名・モデルサイズ等）を受け取って解析するための標準ライブラリ
import os        # ファイルパスの結合や存在確認など、OS依存のファイル操作を行うライブラリ
import sys       # 致命的なエラー発生時にプログラムを即座に強制終了（sys.exit）させるためのライブラリ
import time      # 処理時間を小数点2桁まで精密に計測するためのライブラリ
import copy      # 【追加機能用】Whisper生データをLLM校正前に退避・複製するための標準ライブラリ
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
DEFAULT_MODEL_SIZE = "base"                   # Whisperのモデルサイズ（tiny/base/small/medium/large等）

# 動画ファイルとして認識する拡張子のリスト（これらに該当する場合のみ自動で音声を抽出します）
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]

# 処理完了後に、動画から一時的に抽出した音声ファイル（.m4a）を自動削除するかどうか（True=削除, False=残す）
REMOVE_TEMP_AUDIO = False

# 全体進捗バーで管理する5つの明確な開発工程（タスク）
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
    # 💡【追加機能】LLM校正をスキップするための引数。--no-llm をつけて実行するとLLMを通さずに書き出します
    parser.add_argument("--no-llm", action="store_true", help="LLMによる文脈校正工程をスキップする")

    args = parser.parse_args()

    # 指定されたファイルが実在するか安全確認
    if not os.path.exists(args.input_file):
        print(f"\n[*] エラー: 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    # 総処理時間の計測を開始
    start_time = time.perf_counter()
    print("[*] whisper-llm-srtプロジェクトを起動しました")

    # 入力ファイルの拡張子を解析して動画か音声かを判別する準備
    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    # 【上書き防止のバグ修正】
    # 通常版のユニークな出力先パス（例: test_8.srt）をまず確定させます。
    output_srt_path = get_unique_filepath(base_path + ".srt")
    
    # 確定した通常版のパス（例: test_8.srt）から拡張子を除いたベース（例: test_8）を取得し、
    # そこに確実に「_whisper.srt」を付与（例: test_8_whisper.srt）することで、上書きを完全に防ぎます。
    normal_srt_base, _ = os.path.splitext(output_srt_path)
    output_whisper_srt_path = normal_srt_base + "_whisper.srt"

    target_audio_file = args.input_file
    is_video_input = ext_lower in VIDEO_EXTENSIONS

    # もし入力されたのが動画ファイルだった場合、バックグラウンドで高速に音声（.m4a）を抽出
    if is_video_input:
        from src.transcriber import extract_audio_from_video
        extracted_audio_path = get_unique_filepath(base_path + ".m4a")
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("\n[*] エラー: 動画からの音声抽出に失敗したため、処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path

    # -----------------------------------------------------------------
    # 【バーのみの独立出力 ＆ 改行調整】
    # - 前後の文字・説明文字（{desc}）をフォーマットから完全に除去しました。
    # - 進捗バーは一切の文字を排し、純粋なバーと進捗情報のみで一行を構成します。
    # -----------------------------------------------------------------
    custom_format = "{percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}]"

    with tqdm(
        total=len(PIPELINE_STEPS), 
        bar_format=custom_format,
        dynamic_ncols=False,  # バーが横いっぱいに伸びてチカチカするのを防ぐ
        unit="step"           # 予測表記のハラ落ちを防ぎ、?マークを綺麗に修正
    ) as pbar:

        # 【工程 1】専門用語辞書と口癖（フィラー）リストの読み込み
        word_dict   = load_word_dictionary(args.dict)
        filler_list = load_filler_list(args.filler)
        pbar.update(1)
        pbar.refresh()  # 💡【進捗バー表示の修正】描画を強制更新して20%を確実に表示

        # 【工程 2】OpenAI Whisperによる超高精度な音声解析と文字起こし
        raw_segments, whisper_elapsed = run_whisper_transcribe(
            audio_path=target_audio_file,
            word_dict=word_dict,
            model_size=args.model,
        )
        # 💡【タイミング修正】音声の解析が終わったタイミングで、改行を挟んで即座に出力します。
        tqdm.write(f"[*] Whisper処理時間: {whisper_elapsed:.2f} 秒")
        pbar.update(1)
        pbar.refresh()  # 💡【進捗バー表示の修正】40%を表示

        # 最低限のデータが取れているか安全チェック
        if not raw_segments:
            tqdm.write("\n[*] エラー: Whisperによる音声解析に失敗したか、データが空です。処理を中断します。")
            sys.exit(1)

        # 【工程 3】フィラー（えっと・あの等）をタイムスタンプを維持したまま空文字に置換
        cleaned_segments = clean_fillers_keep_timing(raw_segments, filler_list)
        pbar.update(1)
        pbar.refresh()  # 💡【進捗バー表示の修正】描画を強制更新して60%を確実に表示

        # 💡【追加機能】LLMが書き換える前の「フィラー除去済み生データ」をディープコピーして別変数に完全退避
        whisper_only_segments = copy.deepcopy(cleaned_segments)

        # 【工程 4】LLMが前後の文脈をもとにバッチ校正
        if args.no_llm:
            tqdm.write("[*] オプション検出: --no-llm が指定されたため、LLM校正工程をスキップします。")
            # LLMをスキップする場合は、フィラー除去済み生データをそのまま通常版のデータとする
            llm_refined_segments = copy.deepcopy(whisper_only_segments)
        else:
            # 💡【タイミング修正】工程4のLLM処理が始まる直前で、新しく正確に計測（計測開始）
            llm_start_time = time.perf_counter()

            # ※ あらすじ・辞書は渡しません。前後TEXTの文脈のみで自然な日本語に補正させます。
            llm_refined_segments = refine_context_with_llm(cleaned_segments)
            
            # 💡【タイミング修正】LLMのバッチ校正が完了した直後のタイミングで計測を完了し、改行を挟んで出力します。
            llm_elapsed_time = time.perf_counter() - llm_start_time
            tqdm.write(f"[*] LLM処理時間: {llm_elapsed_time:.2f} 秒")

        pbar.update(1)
        pbar.refresh()  # 💡【進捗バー表示の修正】80%を表示

        # 【工程 5】BudouXによる10〜20文字カット ＋ SRTファイルへの書き出し
        if args.no_llm:
            # 💡【修正】--no-llm 指定時は、通常版の出力パスに生データ字幕のみを出力します
            tqdm.write("[*] デザイナー工程: 通常字幕を出力中...")
            success_normal = write_srt_file(llm_refined_segments, output_srt_path)

            success_whisper = True  # スキップのためTrue扱い（successの論理積を崩さないため）
        else:
            # 通常版（LLM校正後、またはLLMスキップ時）の出力
            tqdm.write("[*] デザイナー工程: 通常字幕を出力中...")
            success_normal = write_srt_file(llm_refined_segments, output_srt_path)
            
            # 【追加機能】Whisper限定版（校正前）の出力
            tqdm.write("[*] デザイナー工程: 校正前のWhisper字幕を出力中...")
            success_whisper = write_srt_file(whisper_only_segments, output_whisper_srt_path)
        
        success = success_normal and success_whisper
        pbar.update(1)
        pbar.refresh()  # 💡【進捗バー表示の修正】100%を表示

    # -----------------------------------------------------------------
    # 【後始末クリーンアップ】
    # -----------------------------------------------------------------
    if is_video_input and REMOVE_TEMP_AUDIO:
        try:
            if os.path.exists(target_audio_file):
                print(f"\n[*] 一時音声ファイルを削除しました: {target_audio_file}")
        except Exception as e:
            print(f"\n[*] 警告: 一時音声ファイルの削除中にエラーが発生しました: {e}")

    # 最終的な成否判定
    if not success:
        print("\n[*] エラー: SRTファイルの書き出しに失敗しました。")
        sys.exit(1)

    # 画面に成果物のパスと、かかった総処理時間をわかりやすく表示
    if args.no_llm:
        # 💡【修正】--no-llm 指定時は通常版のパスのみを表示します
        print(f"[+] 【通常版】字幕ファイル: {output_whisper_srt_path}")
        print(f"[*] 総処理時間: {time.perf_counter() - start_time:.2f} 秒")
    else:
        print(f"[+] 【通常版】字幕ファイル: {output_srt_path}")
        print(f"[+] 【校正前】字幕ファイル: {output_whisper_srt_path}")
        print(f"[*] 総処理時間: {time.perf_counter() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()