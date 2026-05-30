"""
pipeline.py
5 つの工程を時系列順に管理するパイプラインモジュール。

責務:
  - 全体進捗バー（tqdm）の管理
  - 5 工程（辞書読込 / Whisper / フィラー除去 / LLM 校正 / SRT 書き出し）の順序制御
  - Whisper・LLM の出力ファイルを output/ 以下に保存
  - 動画からの音声抽出（video 入力時）
  - 一時音声ファイルのクリーンアップ
  - 処理時間のまとめ表示

各工程の実装は専門モジュール（transcriber / refiner / formatter）に委譲します。
"""

import copy  # LLM 校正前に Whisper 生データをディープコピーして退避するための標準ライブラリ
import os    # ファイルパスの存在確認・削除などに使うライブラリ
import sys   # 致命的なエラー発生時のプログラム強制終了に使うライブラリ
import time  # 総処理時間を計測するためのライブラリ
from tqdm import tqdm  # 全体の処理進捗をターミナル上にアニメーションバーで表示するためのライブラリ

from src.config import REMOVE_TEMP_AUDIO, VIDEO_EXTENSIONS
from src.exceptions import (
    AudioExtractionError,
    FfmpegNotFoundError,
    FileWriteError,
    InvalidConfigError,
    PromptFileNotFoundError,
    WhisperModelLoadError,
    WhisperTranscribeError,
)
from src.formatter import write_srt_file
from src.refiner import refine_context_with_llm
from src.transcriber import (
    clean_fillers_keep_timing,
    extract_audio_from_video,
    load_filler_list,
    load_word_dictionary,
    run_whisper_transcribe,
)
from src.utils import (
    build_output_paths,
    get_unique_filepath,
    save_segments_as_json,
    save_segments_as_plaintext,
)

# 全体進捗バーで管理する 5 工程のラベル（ユーザーが変更するものではないためここで定義）
_PIPELINE_STEPS = [
    "辞書・フィラー読み込み",
    "Whisper 文字起こし",
    "フィラー除去",
    "LLM 校正",
    "SRT 書き出し",
]

# 進捗バーのフォーマット（純粋なバー + 進捗数字のみ。各工程のログは tqdm.write() で別途出力）
_BAR_FORMAT = "{percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}]"


def run(args) -> None:
    """パイプライン全体を実行する関数。main.py から args を受け取って 5 工程を一本道で処理します。

    Args:
        args: argparse.Namespace。以下の属性を持ちます:
            input_file (str): 入力ファイルのパス（音声または動画）
            dict       (str): 単語辞書ファイルのパス
            filler     (str): フィラーリストファイルのパス
            prompt     (str): LLM プロンプトファイルのパス
            model      (str): Whisper モデルサイズ
            no_llm    (bool): True の場合 LLM 校正をスキップする

    各工程で回復不能なエラーが発生した場合は sys.exit(1) でプログラムを停止します。
    """
    start_time = time.perf_counter()  # 総処理時間の計測開始

    tqdm.write("[*] whisper-llm-srt を起動しました")

        # クリーンアップ処理で参照するため、with ブロックの外で初期化しておく
    target_audio_file = args.input_file
    is_video_input    = False

    # 入力ファイルの拡張子を解析して動画か音声かを判別する
    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    # 【上書き防止】LLM 校正版 SRT のユニークな出力パスをまず確定させる（例: test_2.srt）
    # その後 build_output_paths() で全出力ファイルのパスを一括生成する
    try:
        paths = build_output_paths(args.input_file)
    except FileWriteError as e:
        tqdm.write(f"[エラー] 出力ディレクトリの作成に失敗しました: {e}")
        sys.exit(1)
    
    is_video_input = ext_lower in VIDEO_EXTENSIONS

    # 動画ファイルの場合、ffmpeg で音声（.m4a）をバックグラウンドで抽出してから処理する
    if is_video_input:
        # REMOVE_TEMP_AUDIO = True  → 処理完了後に output/audio/ から削除する
        # REMOVE_TEMP_AUDIO = False → 処理完了後も output/audio/ に残す
        extracted_audio_path = paths["extracted_audio"]

        try:
            extract_audio_from_video(args.input_file, extracted_audio_path)
        except (AudioExtractionError, FfmpegNotFoundError) as e:
            tqdm.write(f"[エラー] 音声抽出に失敗しました: {e}")
            sys.exit(1)
    
        target_audio_file = extracted_audio_path

    # -----------------------------------------------------------------
    # 【一本道データライン】
    # 進捗バーは「純粋なバー + 進捗数字」のみのシンプルな表示にしています。
    # 各工程内のログは tqdm.write() で出力し、バーの表示を破壊しません。
    # -----------------------------------------------------------------
    try:
        with tqdm(
            total=len(_PIPELINE_STEPS),
            bar_format=_BAR_FORMAT,
            dynamic_ncols=False,  # バーが横いっぱいに伸びてチカチカするのを防ぐ
            unit="step",
        ) as pbar:

            # 【工程 1/5】専門用語辞書とフィラーリストの読み込み
            word_dict   = load_word_dictionary(args.dict)
            filler_list = load_filler_list(args.filler)
            pbar.update(1)

            # 【工程 2/5】Whisper による音声解析と文字起こし
            try:
                raw_segments, whisper_elapsed = run_whisper_transcribe(
                    audio_path=target_audio_file,
                    word_dict=word_dict,
                    model_size=args.model,
                )
            except (WhisperModelLoadError, WhisperTranscribeError) as e:
                tqdm.write(f"[エラー] Whisper の処理に失敗しました: {e}")
                sys.exit(1)

            tqdm.write(f"[*] Whisper 処理時間: {whisper_elapsed:.2f} 秒")
            pbar.update(1)

            if not raw_segments:
                tqdm.write("[エラー] Whisper の出力が空です。音声ファイルを確認してください。")
                sys.exit(1)

            # 【工程 3/5】フィラー（えっと・あの等）をタイムスタンプを維持したまま空文字に置換する
            cleaned_segments = clean_fillers_keep_timing(raw_segments, filler_list)
            pbar.update(1)

            # LLM が書き換える前の「フィラー除去済み生データ」をディープコピーして退避する
            # これが Whisper 限定版（校正前）の SRT 出力データになる
            whisper_only_segments = copy.deepcopy(cleaned_segments)

            # 【工程 4/5】LLM が前後の文脈をもとにバッチ校正する
            # LLM 校正をスキップするかどうかをここで一度だけ決定する
            skip_llm = (
                args.no_llm                         # --no-llm フラグが指定された場合
                or not os.path.exists(args.prompt)  # プロンプトファイルが存在しない場合
            )

            if args.no_llm:
                # プロンプトファイルが無い時は LLM 校正をスキップし、フィラー除去済み生データをそのまま使う
                tqdm.write("[*] --no-llm が指定されたため、LLM 校正工程をスキップします。")
            elif not os.path.exists(args.prompt):
                # --no-llm 指定時は LLM 校正をスキップし、フィラー除去済み生データをそのまま使う
                tqdm.write(f"[*] プロンプトファイルが見つからないため、LLM 校正をスキップします: {args.prompt}")

            # 【工程 4/5】
            if skip_llm:
                llm_refined_segments = copy.deepcopy(whisper_only_segments)
            else:
                try:
                    llm_refined_segments, llm_elapsed = refine_context_with_llm(
                        segments=cleaned_segments,
                        prompt_file_path=args.prompt,
                        batch_size=args.batch_size_llm
                    )
                except (PromptFileNotFoundError, InvalidConfigError) as e:
                    # プロンプト不正・設定値不正は回復不能のため処理を停止する
                    tqdm.write(f"[エラー] LLM 校正の設定に問題があります: {e}")
                    sys.exit(1)
                tqdm.write(f"[*] LLM 処理時間: {llm_elapsed:.2f} 秒")

            pbar.update(1)

            # 【工程 5/5】BudouX による文字数カット + ファイルの書き出し
            try:
                # 【どんな状態でも、ベースとなる Whisper に関する 3 ファイルは必ず生成されます】
                # srt/ フォルダへ
                write_srt_file(
                    whisper_only_segments,
                    paths["whisper_srt"],
                    min_char_len=args.min_char_len,  # main.py の引数から届いた値
                    max_char_len=args.max_char_len
                )
                save_segments_as_json(whisper_only_segments, paths["whisper_json"])  # transcript/ フォルダへ
                save_segments_as_plaintext(whisper_only_segments, paths["whisper_txt"]) # text/ フォルダへ

                # Whisperの出力ファイル群
                tqdm.write(f"[+] 【Whisper 版 SRT 】{paths['whisper_srt']}")
                tqdm.write(f"[+] 【Whisper 生データ】{paths['whisper_json']}")
                tqdm.write(f"[+] 【Whisper テキスト】{paths['whisper_txt']}")

                # 【追加で refined の 3 ファイルが生成され、計 6 ファイルになります】
                if not skip_llm:
                    # srt/ フォルダへ
                    write_srt_file(
                        llm_refined_segments,
                        paths["refined_srt"],
                        min_char_len=args.min_char_len,
                        max_char_len=args.max_char_len
                    )         
                    save_segments_as_json(llm_refined_segments, paths["refined_json"])  # transcript/ フォルダへ
                    save_segments_as_plaintext(llm_refined_segments, paths["refined_txt"]) # text/ フォルダへ

                    # LLMの出力ファイル群
                    tqdm.write(f"[+] 【LLM 校正版 SRT 】{paths['refined_srt']}")
                    tqdm.write(f"[+] 【LLM 生データ   】{paths['refined_json']}")
                    tqdm.write(f"[+] 【LLM 校正テキスト】{paths['refined_txt']}")
                    
            except FileWriteError as e:
                tqdm.write(f"[エラー] 成果物ファイルの書き出しに失敗しました: {e}")
                sys.exit(1)

            pbar.update(1)

    # -----------------------------------------------------------------
    # 【後始末クリーンアップ】
    # -----------------------------------------------------------------
    # 正常終了・エラー終了どちらでも必ずここが実行される。
    # REMOVE_TEMP_AUDIO = True の場合のみ output/audio/ から削除する。
    # REMOVE_TEMP_AUDIO = False の場合はそのまま output/audio/ に残す。
    finally:
        if is_video_input and REMOVE_TEMP_AUDIO:
            try:
                if os.path.exists(target_audio_file):
                    os.remove(target_audio_file)  # 一時音声ファイルを削除
                    tqdm.write(f"[*] 一時音声ファイルを削除しました: {target_audio_file}")
            except OSError as e:
                # クリーンアップ失敗は警告にとどめる（メイン処理は完了しているため）
                tqdm.write(f"[警告] 一時音声ファイルの削除中にエラーが発生しました: {e}")
        elif is_video_input and not REMOVE_TEMP_AUDIO:
            # 削除しない場合は保存先をログに残す
            tqdm.write(f"[*] 抽出した音声ファイルを保存しました: {target_audio_file}")

    tqdm.write(f"[*] 総処理時間: {time.perf_counter() - start_time:.2f} 秒")