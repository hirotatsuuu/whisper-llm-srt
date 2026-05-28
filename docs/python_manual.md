# 通常の Python / pip コマンドマニュアル (uv を使わない場合)

本プロジェクトは `uv` での管理を推奨していますが、通常の `python` および `pip` コマンドを使用して環境構築やスクリプトの実行を行う際の手順一覧です。

---

## 1. 仮想環境の構築とライブラリの導入

`uv` を使わない場合、手動で仮想環境（`.venv`）を作成し、アクティベート（有効化）した上でライブラリをインストールする必要があります。

### 🔧 1. 仮想環境の作成
プロジェクトのルートフォルダで以下を実行し、`.venv` フォルダを生成します。
```powershell
python -m venv .venv
```

### 🔓 2. 仮想環境のアクティベート（有効化）
作成した仮想環境をターミナルに認識させます。
```powershell
.venv\Scripts\Activate.ps1
```
* ※ アクティベートに成功すると、ターミナルの先頭に `(.venv)` と表示されます。
* ※ スクリプト実行エラー（セキュリティエラー）が出る場合は、事前に `Set-ExecutionPolicy RemoteSigned -Scope Process` を実行してください。

### 📥 3. ライブラリの一括インストール
仮想環境がアクティベートされた状態で、`requirements.txt` を使って必要なライブラリをまとめてインストールします。
```powershell
pip install -r requirements.txt
```

---

## 2. スクリプトの実行方法

必ず**仮想環境がアクティベートされていること**を確認してから実行してください。

### 🚀 1. 基本的な実行方法（デフォルト設定）
`./data/test.m4a` に音声ファイルを配置している場合、引数なしで実行します。
```powershell
python src/main.py
```

### 🎛️ 2. 引数を指定して実行
特定のファイルを指定したり、モデルのサイズを変更したりする場合のコマンド例です。

```powershell
# 特定の音声ファイルを指定
python src/main.py ./data/audio.mp3

# 動画ファイルを指定して直接実行
python src/main.py ./data/input_movie.mp4

# 高精度モデル（small）を指定
python src/main.py ./data/test.m4a -m small

# 別の単語辞書ファイルを指定
python src/main.py ./data/test.m4a -d my_dict.txt
```

---

## 3. 日常管理コマンド

### ➕ 1. ライブラリを個別に新しく追加したいとき
```powershell
pip install ライブラリ名
```
* ※ 追加した後は、他の環境でも再現できるように `pip freeze > requirements.txt` を実行して `requirements.txt` を手動で更新しておく必要があります。

### ❌ 2. 不要になったライブラリを削除したいとき
```powershell
pip uninstall ライブラリ名
```

### 📋 3. インストール済みライブラリの一覧確認
```powershell
pip list
```

---

## 4. 仮想環境の終了方法

作業が終わり、仮想環境から抜け出したい場合は以下のコマンドを叩きます。
```powershell
deactivate
```
ターミナルの先頭から `(.venv)` の表示が消え、通常のターミナル環境に戻ります。