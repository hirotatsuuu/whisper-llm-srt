# Whisper×ELYZA×BudouXでSRT字幕生成スクリプト

OpenAIの文字起こしAI「Whisper」を使用し、動画編集（Shotcut、DaVinci Resolve等）のテロップ作成に最適化されたSRT字幕ファイルを自動生成するPythonスクリプトです。

## 主な特徴

- **耳・頭脳・デザイナーの3階層モジュール設計**: データの流れを時系列（Whisperによる音声認識 ➔ LLMによる文脈校正 ➔ BudouXによる最終レイアウト整形）で完全一本道化。メンテナンス性に優れた洗練された構造です。
- **ミリ秒単位の完全シンクロ**: 等分（割り算）による時間配分を廃止し、実際の発話タイミング（ミリ秒）に合わせて字幕の表示時間を自動伸縮します。
- **タイムスタンプ維持型フィラー除去**: LLMに渡す前の段階で、「えっと」「あのー」といった言葉のヒゲ（フィラー）の時間データ（箱）は残したまま「空文字」に置換。音ズレを100%発生させずにノイズをカットします。除去する語は `filler.txt` で自由にカスタマイズできます。
- **文脈ベースのAI校正**: ローカルLLM（ELYZA）が各セグメントの前後の流れを読み取り、音声認識特有の同音異義語ミスや漢字の誤変換を自動補正します。あらすじや辞書は渡さず、前後テキストの文脈のみを手がかりとすることで、誤誘導のない高精度な補正を実現しています。
- **テロップ向けの文字数絶対厳守**: 1行の文字数を指定文字数間に自動調整。可能な限り指定文字数を超えないよう自動で安全に分割します。
- **自然な文節区切り**: Googleの文節区切りライブラリ「BudouX」を搭載。単語の途中で不自然にぶつ切りされるのを防ぎます。
- **クレンジング機能**: 画面テロップで邪魔になりやすい句読点（「、」や「。」）を自動で除去。ただし「。」や「、」が来た場合はそこで文脈の区切りと判断し、スマートにLLMへバッチ送信します。
- **動画ファイルに直接対応**: 動画（.mp4等）を指定すると、自動で同名の音声（.m4a）を切り出してから文字起こしを行います。
- **堅牢なエラーハンドリング**: メモリ不足やファイルの破損、ffmpegの未導入などが起きた際、クラッシュせずに分かりやすい日本語でエラーを通知し、最悪の場合でもWhisperの生データを維持して生存ルートを確保します。
- **処理時間の計測**: 実行完了時に、Whisper・LLM・総処理の各時間を小数点2桁でコンソールに出力します。

---

## 必要要件・インストール

あらかじめ Windows に Python 3.12 以上がインストールされている必要があります。

### 1. 外部ツール（ffmpeg / Ollama）のインストール

Windowsの標準機能（winget）を使い、ターミナル（PowerShell等）で以下を実行してインストールしてください。

動画から音声を抽出したり、音声フォーマットを変換するために ffmpeg が必須です。

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
```

また、文脈校正を行うために Ollama が必要です。

```powershell
winget install Ollama.Ollama
ollama --version
```

Ollama起動後、使用するモデル（ELYZA）をあらかじめダウンロードしておいてください。
※容量が大きいので時間が掛かります。

```powershell
`ollama pull hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf` 
```

※インストール後、設定を反映させるために一度ターミナル（またはPC）を再起動してください。

### 2. 環境構築とライブラリの同期

本プロジェクトでは、高速なパッケージ管理ツール `uv` を使用した環境管理を推奨しています。

1. `uv` が未インストールの場合は、以下を実行して導入してください。

```powershell
winget install Astral.uv
```

※インストール後は一度ターミナルを完全に閉じて、新しく開き直してください。

2. プロジェクトのルートフォルダで以下のコマンドを実行し、環境を同期します。

```powershell
uv sync
```

※初回実行時、指定したWhisperのモデル（標準では base）が自動でダウンロードされます。

---

## プロジェクトのディレクトリ構造

スクリプトをスムーズに動かすために、以下の構造でファイルを配置してください。

```text
├── src/
│   ├── __init__.py         #  Pythonプロジェクトのデフォルト
│   ├── pipeline.py         #  main.pyから実行されるスクリプト
│   ├── transcriber.py      # 【第1工程：耳】Whisperによる音声認識、辞書・フィラー読み込み
│   ├── refiner.py          # 【第2工程：頭脳】前後の文脈をもとにしたバッチ校正（ELYZA）
│   ├── formatter.py        # 【第3工程：デザイナー】BudouXによる整形、文字数カット、SRT出力
│   ├── utils.py            #  汎用ユーティリティ関数
│   ├── exceptions.py       #  カスタム例外クラス
│   └── config.py           #  初期値など後から変更可能な設定ファイル
├── resources/              # プログラム内で使用する各種データ
│   ├── sample/             # resourcesフォルダに格納するサンプルデータ      
│   ├── dictionary.txt      # 固有名詞・専門用語辞書（任意）
│   ├── filler.txt          # 除去したいフィラー語リスト（任意）
│   └── prompt.txt          # LLMへのプロンプト 
├── input/                  # 変換したいデータフォルダ（※.gitignoreにより、大容量メディアはGit管理外）
│   └── test.m4a            # 既定の音声ファイル（mp3, mp4等も可）      
├── output/                 # 変換後のデータフォルダ（※.gitignoreにより、大容量メディアはGit管理外）
│   ├── srt/                # SRT形式で出力したファイルの格納（目的の字幕データ）
│   ├── text/               # TXT形式で出力したファイルの格納（AIの出力のテキストデータ）
│   └── transcript/         # JSON形式で出力したファイルの格納（AIの出力の生データ）
├── docs/                   # マークダウン形式の書類の格納
│   ├── uv_manual.md        # uvコマンドの解説・導入マニュアル
│   ├── python_manual.md    # 通常の python / pip コマンドマニュアル
│   ├── git_manual.md       # Git / GitHub の基本操作・ワークフローマニュアル
│   └── troubleshooting.md  # エラー・不具合発生時の対処マニュアル
├── main.py                 # 全体を統括し一本道で処理を回すメインスクリプト
├── pyproject.toml          # プロジェクトの設定・ライブラリ管理ファイル（uv用）
├── uv.lock                 # 環境の完全同期用ロックファイル（手動編集不可・Git管理必須）
├── requirements.txt        # ライブラリ一括インストール用ファイル（pip用）
├── .gitignore              # Git管理除外設定ファイル
├── .gitattributes          # 改行コードの統一やバイナリ保護のためのGit属性設定ファイル
├── .editorconfig           # エディタのコード規約設定ファイル
├── README.md               # 本説明ファイル
└── LICENSE                 # ライセンス規約ファイル（MITライセンス）
```

---

## 使い方

プロジェクトのルートフォルダで実行してください。`uv` 環境であれば、手動で仮想環境に入り直す（Activateする）必要はありません。

### 1. 基本的な実行方法（デフォルト設定）

`./input/test.m4a` に音声ファイルを配置している場合、引数なしで実行するだけで自動的に文字起こしが始まり、`output`フォルダに成果物が出力されます。（計6ファイル生成される）

```powershell
uv run whisper-llm-srt
```

### 2. 特定の音声・動画ファイルを指定して実行

特定のファイルパスを指定したり、文字起こし精度を上げたい場合は、引数を使って実行できます。動画ファイルを指定した場合は、自動的に同フォルダ内に音声ファイル（.m4a）を抽出してから処理を行います。

```powershell
# 特定の音声ファイルを指定
uv run whisper-llm-srt ./input/audio.mp3

# 動画ファイルを指定して直接実行
uv run whisper-llm-srt ./input/input_movie.mp4

# 超高精度モデル（large）を指定
uv run whisper-llm-srt ./input/test.m4a -m large

# 別の単語辞書ファイルを指定
uv run whisper-llm-srt ./input/test.m4a -d ./resources/sample/dict_sample.txt

# 別のフィラーリストを指定
uv run whisper-llm-srt ./input/test.m4a -f ./resources/sample/dict_sample.txt

# LLMを使用せずWhisperのみで実行
uv run whisper-llm-srt --no-llm
```

##### 指定可能なWhisperモデルサイズ

右にいくほど精度が上がりますが、処理時間（PCのスペック）を要します。

```
tiny < base (デフォルト) < small < medium < large
```

---

## 詳細マニュアル（docsフォルダ）

より詳しい手順や、環境に合わせた使い方は `docs` フォルダ内の各ドキュメントを参照してください。

- **`docs/uv_manual.md`**: `uv` のインストール方法や、ライブラリの追加・削除など便利な応用コマンドの解説。
- **`docs/python_manual.md`**: `uv` を使用せず、従来の `python` や `pip` コマンド、`requirements.txt` を使って動かしたい場合の手順。
- **`docs/git_manual.md`**: 日常的なコミット・プッシュの流れや、間違えて動画を登録してしまった場合の対処法。
- **`docs/troubleshooting.md`**: 「ffmpegが見つからない」「メモリ不足で強制終了する」など、エラーが起きたときの自己解決手順。
- **`docs/ai_manual.md`**: 使用している3種類のAIモデルの簡単な説明。

---

## 固有名詞の登録（dictionary.txt）

認識率を上げたい固有の単語やYouTubeのチャンネル名などがある場合は、`resources` フォルダ内に `dictionary.txt` を作成し、1行に1単語ずつ記述してください。登録された単語は、Whisperの初期プロンプトとして渡されます。

```text
おたつ  #で始まる行はコメントです
ユーラシア大陸
自転車世界一周
```

## フィラー語の登録（filler.txt）

除去したいフィラー語がある場合は、`resources` フォルダ内に `filler.txt` を作成し、1行に1語ずつ記述してください。ファイルが存在しない場合は、デフォルトのフィラーリスト（えっと・あの・あのー・えー等）が自動で使われます。

```text
えっと  #で始まる行はコメントです
あのー
なんか
```

## LLMへ送るプロンプトの登録（llm_prompt.txt）

LLMへの指示書としてのプロンプトを記入する。この文章によって字幕の精度が大きく変わる。

```text
あなたは日本語字幕の校正専門家です。
【補正対象データ】の文脈を読み、文字の「誤変換」のみを自然な日本語に修正してください。

【厳守ルール】
1. 途中のIDを絶対に省略・削除せず、すべてのIDをそのまま出力すること。
2. 元の文章を勝手に要約したり、別の表現に言い換えたりしないこと。
3. TEXTが空欄のIDは、そのまま空欄（TEXT: ）で返すこと。
4. 挨拶や解説は一切出力せず、指定フォーマットのみを返すこと。
```

---

## スクリプト内の設定変更

既定のファイル名やモデルを変えたい場合は、`src/config.py` を直接書き換えてください。

```python
DEFAULT_AUDIO_FILE  = "./input/test.m4a" 
DEFAULT_DICT_FILE   = "./resources/dictionary.txt"
DEFAULT_FILLER_FILE = "./resources/filler.txt"

# --- Whisper ---
DEFAULT_MODEL_SIZE  = "base" # （tiny/base/small/medium/large）

# --- LLM ---
LLM_MODEL_NAME  = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"
LLM_PROMPT_FILE = "./data/llm_prompt.txt"

# --- 字幕レイアウト ---
MIN_CHAR_LEN = 10  
MAX_CHAR_LEN = 20  
```

---

## 出力サンプル (SRT形式)

```text
1
00:00:01,080 --> 00:00:02,560
はいどうもおたつです

2
00:00:03,140 --> 00:00:03,760
前回ですね
```

---

## ライセンス

本プロジェクトは **MIT ライセンス** の下で公開されています。商用利用、修正、配布、プライベート利用を含め、誰でも自由に無償で利用することができます。詳細は同梱の `LICENSE` ファイルをご確認ください。