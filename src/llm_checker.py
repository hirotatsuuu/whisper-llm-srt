import os      # プロンプトファイルのパス確認・読み込みに使うライブラリ
import ollama  # ローカルLLM実行エンジン「Ollama」と通信するためのライブラリ
import time    # 処理時間を小数点2桁まで精密に計測するためのライブラリ
from tqdm import tqdm  # tqdm.write を使って、進捗バーを破壊せずにログを出力するためのライブラリ


# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
# 将来モデルを変更したい場合や、軽量モデル（gemma2:2bなど）を試したい場合はここを書き換えてください。
LLM_MODEL_NAME = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"

# LLMへのプロンプトテンプレートファイルのパス
# ファイルの末尾に【補正対象データ】を自動で結合して使います。
# ファイルが存在しない場合は、下記のDEFAULT_PROMPTにフォールバックします。
LLM_PROMPT_FILE = "./data/llm_refine_prompt_template.txt"

# プロンプトファイルが見つからない場合のフォールバック用デフォルトプロンプト
DEFAULT_PROMPT = """あなたは日本語字幕の校正専門家です。
以下の【補正対象データ】は、音声認識AIが出力した日本語テキストです。
このデータの一番の絶対情報は「音（発音・響き）」であり、もともとのテキストが正しいことが大半です。

しかし、音声認識の特性上、稀に「同音異義語の誤変換」「漢字ミス」「音が近い言葉への聞き違い」によって変な文章になっている箇所があります。
各IDのTEXTを前後の文脈から読み取り、元の「音（発音）」を決して崩さない範囲で、日本語として最も自然な表現に書き直してください。

【最重要の補正ルール】
1. 音（発音）の維持【絶対厳守】
   元のテキストの「音（発音・響き）」から離れた言い換えや単語変更は一切NGです。
   修正する場合は、必ず「音が極めて近くて、前後の文脈に当てはまる正しい言葉」を推測して当てはめてください。
   （NG例：「難しい」を「しにくい」や「困難」に変えるのは絶対NG）

2. 勝手な文章削除の禁止【絶対厳守】
   変な表現でも、テキストを勝手に削除・省略することは絶対NGです。必ず元の音の長さに合わせて出力してください。

3. 基本は現状維持（過剰修正の禁止）
   一般的な会話として通じる部分は変更しないでください。明確に不自然な箇所のみ対象とします。

4. 空欄の維持
   TEXTが空欄（フィラー除去済み）のIDは、必ずそのまま空欄で返してください。

5. 出力の純粋性
   解説・前置き・後置きは一切不要です。指定フォーマットのみ出力してください。

【出力フォーマット】（このフォーマットを厳守してください）
ID: 番号 | TEXT: 校正後のテキスト"""
# =====================================================================


def load_prompt_template(file_path):
    """外部テキストファイルからプロンプトテンプレートを読み込む関数。

    ファイルが存在しない場合はDEFAULT_PROMPTにフォールバックします。
    【補正対象データ】の結合はこの関数では行わず、呼び出し元で行います。
    """
    if not file_path or not os.path.exists(file_path):
        tqdm.write(f"[*] 注意: プロンプトファイルが見つかりません: {file_path} (デフォルトプロンプトで続行します)")
        return DEFAULT_PROMPT

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            template = f.read().strip()

        if not template:
            tqdm.write(f"[*] 注意: プロンプトファイルが空です（デフォルトプロンプトで続行します）")
            return DEFAULT_PROMPT

        prompt_filename = os.path.basename(file_path)
        tqdm.write(f"[*] プロンプトテンプレート [{prompt_filename}] を読み込みました")
        return template

    except Exception as e:
        tqdm.write(f"[*] 警告: プロンプトファイルの読み込み中にエラーが発生しました（デフォルトプロンプトで続行します）: {e}")
        return DEFAULT_PROMPT


def refine_context_with_llm(segments: list) -> list:
    """
    【第2工程】
    各セグメントの前後の文脈をもとに、タイムスタンプを完全に維持したまま
    日本語として不自然な箇所をバッチ校正します。

    ※ seg["text"] のみを更新します。
       formatter.py は seg["text"] をBudouXで文節分割してSRTに書き出します。
       words の生タイムスタンプは一切触らず、時間情報を保護します。
    """
    tqdm.write("\n[*] ローカルLLMによる文脈バッチ校正を開始します...")

    # プロンプトテンプレートを外部ファイルから読み込む
    # 【補正対象データ】は末尾に結合する形で使うため、ここではテンプレート部分だけを取得する
    prompt_template = load_prompt_template(LLM_PROMPT_FILE)

    llm_start_time = time.perf_counter()  # LLM全体の処理時間計測開始（ループの外で1回だけ）

    refined_segments = []
    current_batch = []

    for index, seg in enumerate(segments):
        current_batch.append(seg)

        # 句読点（。、）があるか、10件溜まったか、最後の要素ならバッチ処理を実行
        has_punctuation = "。" in seg["text"] or "、" in seg["text"]
        is_batch_full = len(current_batch) >= 10 or index == len(segments) - 1

        if has_punctuation or is_batch_full:

            # バッチ内の各セグメントを「ID: X | TEXT: Y」形式の文字列に整形
            batch_prompt_text = ""
            for batch_seg in current_batch:
                batch_prompt_text += f"ID: {batch_seg['id']} | TEXT: {batch_seg['text']}\n"

            # テンプレートの末尾に【補正対象データ】を結合して完全なプロンプトを組み立てる
            prompt = prompt_template + "\n\n【補正対象データ】\n" + batch_prompt_text

            try:
                # Ollama APIを呼び出してローカルLLM（ELYZA等）を実行
                response = ollama.chat(
                    model=LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )

                # LLMからの返答を行ごとに分解して解析しやすくする
                llm_lines = response["message"]["content"].strip().split("\n")

                # 各セグメントに対して、LLMの修正結果をマッピング
                for batch_seg in current_batch:
                    target_id = batch_seg["id"]
                    old_text = batch_seg["text"]
                    corrected_text = old_text  # 一致するIDが見つからない場合のフォールバック（現状維持）

                    # タイムスタンプの健全性チェック（異常があれば生データのまま維持）
                    words = batch_seg.get("words", [])
                    is_timestamp_valid = True

                    for w in words:
                        start_val = w.get("start")
                        end_val = w.get("end")
                        # タイムスタンプがNone・空・数値変換不可の場合は異常と判定
                        if start_val is None or end_val is None or start_val == "" or end_val == "":
                            tqdm.write(f"\n[⚠ タイムスタンプ空エラー ID:{target_id}] 安全のため生データを維持します。")
                            is_timestamp_valid = False
                            break
                        try:
                            float(start_val)
                            float(end_val)
                        except (ValueError, TypeError):
                            tqdm.write(f"\n[⚠ タイムスタンプ数値不正 ID:{target_id}] start:{start_val}, end:{end_val}")
                            is_timestamp_valid = False
                            break

                    # タイムスタンプが正常な場合のみLLMの校正結果を反映する
                    if is_timestamp_valid:
                        for line in llm_lines:
                            if f"ID: {target_id} " in line or f"ID:{target_id}" in line:
                                if "TEXT:" in line:
                                    parsed_text = line.split("TEXT:", 1)[1].strip()
                                    # LLMが誤って空文字を返してきた場合は元のテキストを維持するガード
                                    if parsed_text == "" and old_text != "":
                                        corrected_text = old_text
                                    else:
                                        corrected_text = parsed_text
                                    break

                        # 修正が行われた場合のみ差分をログ出力
                        if old_text != corrected_text:
                            tqdm.write(f"\n[修正検出 ID:{target_id}]")
                            tqdm.write(f"  BEFORE: {old_text}")
                            tqdm.write(f"  AFTER : {corrected_text}")

                        # seg["text"] だけを更新する（wordsの生タイムスタンプには一切触れない）
                        batch_seg["text"] = corrected_text

                    refined_segments.append(batch_seg)

            except Exception as e:
                # 【堅牢なエラーハンドリング】Ollamaのエラーが起きてもクラッシュさせない。
                # このバッチはWhisperの生データをそのまま採用して生存ルートを確保する。
                tqdm.write(f"\n[*] ブロック処理中にエラーが発生しました（安全のため生データを維持します）: {e}")
                for batch_seg in current_batch:
                    refined_segments.append(batch_seg)

            # バッチをリセットして次のグループに備える
            current_batch = []

    # LLM全体の処理時間を計測終了・返却（main側で一元表示するため、ここでは出力しない）
    llm_elapsed_time = time.perf_counter() - llm_start_time

    return refined_segments, llm_elapsed_time