import ollama  # ローカルLLM実行エンジン「Ollama」と通信し、ELYZAなどの大規模言語モデルへ字幕修正指示を送るライブラリ
import time    # 処理時間をミリ秒単位で計測するためのライブラリ
from tqdm import tqdm  # tqdm.write を使用して進捗バーを破壊せずにログ出力するためのライブラリ

# =====================================================================
# 初期設定エリア：変更したい場合はここを書き換えてください
# =====================================================================
# 将来モデルを変更したい場合や、軽量モデル（gemma2:2bなど）を試したい場合はここを書き換えてください。
LLM_MODEL_NAME = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"

# LLMへの校正プロンプトテンプレート（{batch_prompt_text} は実行時に埋め込まれます）
# ※ あらすじは渡しません。あらすじ内の誤った表現をAIが「正しい」と誤認するリスクを排除するためです。
# ※ 辞書も渡しません。辞書の単語に引っ張られて誤補正が起きるケースがあるためです。
#   その代わり、各TEXTの前後の流れ（文脈）だけを手がかりに、日本語として自然な表現を予測・補正させます。
LLM_REFINE_PROMPT_TEMPLATE = """あなたは日本語字幕の校正専門家です。
以下の【補正対象データ】は、音声認識AIが「現実の音声（音）」を高い精度で文字起こしした日本語テキストです。
このデータの一番の絶対情報は「音（発音・響き）」であり、もともとのテキストが正しいことが大半です。

しかし、音声認識の特性上、稀に「同音異義語の誤変換」「漢字ミス」「音が近い言葉への聞き違い」によって、変な文章や不自然な日本語になっている箇所があります。
各IDのTEXTを前後の文脈から読み取り、元のテキストの「音（発音）」を決して崩さない範囲で、日本語として最も自然な表現に1対1で書き直してください。

【最重要の補正ルール】
1. 音（発音）の維持【絶対厳守】
   元のテキストの「音（発音・響き）」から完全に離れた勝手な言い換えや単語の変更は一切認められません。
   （NG例：意味が近くても音が異なる変更。例:「難しい」を「しにくい」や「困難」に変えるのは絶対にNG）
   修正する場合は、必ず「音が極めて近くて、前後の文脈に当てはまる正しい言葉」を推測して当てはめてください。

2. 勝手な文章削除の絶対禁止【絶対厳守】
   LLMが理解しにくい文章や、変な表現であっても、テキストを勝手に消去・省略することは絶対にNGです。必ず元の音の長さに合わせてテキストを出力してください。

3. 基本は現状維持（過剰修正の禁止）
   もともとのテキストが正しいことが多いため、一般的な会話や表現として通じる部分は、少し口語的であっても一切変更しないでください。不自然で変な日本語になっている箇所のみを対象とします。

4. 空欄の維持
   TEXTが空欄（フィラー除去済み）のIDは、必ずそのまま空欄で返してください。

5. 出力の純粋性
   解説・前置き・後置きは一切不要です。指定フォーマットのみ出力してください。

【出力フォーマット】（このフォーマットを厳守してください）
ID: 番号 | TEXT: 校正後のテキスト

【補正対象データ】
{batch_prompt_text}"""
# =====================================================================


def refine_context_with_llm(segments: list) -> list:
    """
    【第2工程】
    各セグメントの前後の文脈をもとに、タイムスタンプを完全に維持したまま
    日本語として不自然な箇所をバッチ校正します。
    """
    print("\n[*] ローカルLLMによる文脈バッチ校正を開始します...")

    refined_segments = []
    current_batch = []

    for index, seg in enumerate(segments):
        current_batch.append(seg)

        # 句読点（。、）があるか、または10件溜まったら、または最後の要素ならバッチ処理を実行
        has_punctuation = "。" in seg["text"] or "、" in seg["text"]
        is_batch_full = len(current_batch) >= 10 or index == len(segments) - 1

        if has_punctuation or is_batch_full:
            # バッチ用のプロンプトテキストを組み立て
            batch_prompt_text = ""
            for batch_seg in current_batch:
                batch_prompt_text += f"ID: {batch_seg['id']} | TEXT: {batch_seg['text']}\n"

            # テンプレートにバッチデータを埋め込み
            prompt = LLM_REFINE_PROMPT_TEMPLATE.format(batch_prompt_text=batch_prompt_text)

            try:
                # Ollama APIを呼び出してローカルLLM（ELYZA等）を実行
                response = ollama.chat(
                    model=LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )

                # LLMからの返答を行ごとに分解
                llm_lines = response["message"]["content"].strip().split("\n")

                # 各セグメントに対して、LLMの修正結果をマッピング
                for batch_seg in current_batch:
                    target_id = batch_seg["id"]
                    old_text = batch_seg["text"]
                    corrected_text = old_text  # フォールバック用（一致するIDがない場合は現状維持）

                    # 💡【エラー処理】セグメント全体の内部単語(words)データのタイムスタンプ健全性チェック
                    words = batch_seg.get("words", [])
                    is_timestamp_valid = True
                    
                    for w in words:
                        start_val = w.get("start")
                        end_val = w.get("end")
                        # タイムスタンプが None、空文字、または数値として不正な場合の異常検知
                        if start_val is None or end_val is None or start_val == "" or end_val == "":
                            print(f"\n[⚠️ タイムスタンプ空エラー ID:{target_id}] 単語データ内の一部タイムスタンプが空、またはNoneです。安全のため該当箇所の生データを維持します。")
                            is_timestamp_valid = False
                            break
                        try:
                            float(start_val)
                            float(end_val)
                        except (ValueError, TypeError):
                            print(f"\n[⚠️ タイムスタンプ数値不正エラー ID:{target_id}] タイムスタンプを数値に変換できません (start:{start_val}, end:{end_val})。")
                            is_timestamp_valid = False
                            break

                    # タイムスタンプが正常な場合のみLLMのテキスト反映処理へ進む
                    if is_timestamp_valid:
                        for line in llm_lines:
                            if f"ID: {target_id} " in line or f"ID:{target_id}" in line:
                                if "TEXT:" in line:
                                    parsed_text = line.split("TEXT:", 1)[1].strip()
                                    # 【改善①】LLMが誤って空文字を返してきた場合、元のテキストを維持するガード
                                    if parsed_text == "" and old_text != "":
                                        corrected_text = old_text
                                    else:
                                        corrected_text = parsed_text
                                    break

                        # 修正が行われた場合はログに出力
                        if old_text != corrected_text:
                            print(f"\n[修正検出 ID:{target_id}]")
                            print(f"  BEFORE: {old_text}")
                            print(f"  AFTER : {corrected_text}")

                        # セグメントのテキストを校正後のものにアップデート
                        batch_seg["text"] = corrected_text

                    # --- 【最重要：タイムスタンプを維持するための変更】 ---
                    # formatter.py は words を使って字幕を組み立てるため、こちらも更新が必須。
                    # 以前の「最初のwordにまとめて載せ、残りのwordは空文字化」する方式は、
                    # 後続の処理でタイムスタンプを消失させる致命的な原因となっていたため撤廃しました。
                    # ここでは、元のwordsデータ（生の時間情報）を1ミリ秒も汚さずにそのまま次段へ引き渡します。
                    # (words変数の定義位置をタイムスタンプ検証用に上部へ移動させ、整合性を確保しています)

                    refined_segments.append(batch_seg)

            except Exception as e:
                # 【堅牢なエラーハンドリング】
                # Ollamaのエラーやメモリ不足が起きても、タスクをクラッシュさせない。
                # このバッチはWhisperの生データをそのまま採用して生存ルートを確保する。
                print(f"\n[*] ブロック処理中にエラーが発生しました（安全のため生データを維持します）: {e}")
                for batch_seg in current_batch:
                    refined_segments.append(batch_seg)

            # バッチをリセット
            current_batch = []

    # 💡 main.py 側で正確に一元管理して出力するため、ここにあった重複 print は削除しました
    return refined_segments