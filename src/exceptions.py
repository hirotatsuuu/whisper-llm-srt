"""
exceptions.py
プロジェクト全体で使用するカスタム例外クラスを定義するモジュール。

例外を種類ごとに定義することで、呼び出し元が「何が原因で失敗したか」を
except の型で正確に識別・ハンドリングできるようになります。
握りつぶし（bare except）を防ぎ、デバッグを容易にする目的で作成しています。
"""


class WhisperSrtBaseError(Exception):
    """このプロジェクト固有の例外の基底クラス。

    すべてのカスタム例外はこのクラスを継承します。
    呼び出し元で「プロジェクト由来のエラー全体」をまとめて捕捉したい場合は
    このクラスを except で指定してください。
    """
    pass


# ------------------------------------------------------------------
# ファイル入出力系
# ------------------------------------------------------------------

class FileReadError(WhisperSrtBaseError):
    """ファイルの読み込みに失敗した場合に送出される例外。

    対象: テキストファイル（辞書・フィラー・プロンプト等）の open/read 失敗。
    """
    pass


class FileWriteError(WhisperSrtBaseError):
    """ファイルへの書き込みに失敗した場合に送出される例外。

    対象: SRT ファイル・テキスト出力ファイルの open/write 失敗。
    """
    pass


class PromptFileNotFoundError(WhisperSrtBaseError):
    """プロンプトファイルが存在しない、または空の場合に送出される例外。

    プロンプトは LLM 校正の動作に必須のため、フォールバックせず
    明示的にエラーとして扱います。
    """
    pass


# ------------------------------------------------------------------
# 音声・動画処理系
# ------------------------------------------------------------------

class AudioExtractionError(WhisperSrtBaseError):
    """動画ファイルからの音声抽出に失敗した場合に送出される例外。

    対象: ffmpeg による音声ストリーム抽出・再エンコードの失敗。
    """
    pass


class FfmpegNotFoundError(WhisperSrtBaseError):
    """ffmpeg コマンドがシステムに見つからない場合に送出される例外。

    winget install Gyan.FFmpeg でインストールできます。
    """
    pass


class WhisperTranscribeError(WhisperSrtBaseError):
    """Whisper による文字起こし処理に失敗した場合に送出される例外。

    対象: モデルのロード失敗・transcribe 実行中の予期せぬエラー。
    """
    pass


class WhisperModelLoadError(WhisperSrtBaseError):
    """Whisper モデルのメモリへの読み込みに失敗した場合に送出される例外。

    対象: MemoryError（メモリ不足）または load_model() の汎用的な失敗。
    """
    pass


# ------------------------------------------------------------------
# LLM 校正系
# ------------------------------------------------------------------

class LlmApiError(WhisperSrtBaseError):
    """Ollama API の呼び出しに失敗した場合に送出される例外。

    対象: ollama.chat() の通信エラー・タイムアウト・モデル未インストール等。
    このエラーはバッチ単位でキャッチし、Whisper 生データで代替して処理を継続します。
    """
    pass


class InvalidTimestampError(WhisperSrtBaseError):
    """セグメント内の単語タイムスタンプが不正な値を持つ場合に送出される例外。

    対象: start/end が None・空文字・数値変換不可能な値。
    このエラーはセグメント単位でキャッチし、LLM 校正をスキップして生データを維持します。
    """
    pass


# ------------------------------------------------------------------
# 設定系
# ------------------------------------------------------------------

class InvalidConfigError(WhisperSrtBaseError):
    """設定値が不正な場合に送出される例外。

    対象: BATCH_SIZE_LLM が 0 以下など、動作を保証できない設定値。
    """
    pass