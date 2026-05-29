import ollama  # ローカルLLM実行エンジン「Ollama」と通信し、ELYZAなどの大規模言語モデルへ字幕修正指示を送るライブラリ
import time    # 処理時間をミリ秒単位で計測するためのライブラリ


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
以下の【補正対象データ】は、音声認識AIが出力した日本語テキストです。
音声認識では「同音異義語の誤変換」「漢字ミス」「不自然なカタカナ語」などのエラーが多発します。
各IDのTEXTを前後の文脈から読み取り、日本語として最も自然な表現に1対1で書き直してください。

【補正ルール】
- 前後のTEXTの流れを必ず参照し、話の文脈に合った言葉を選んでください
- 日本語として不自然・不正確な箇所のみ修正し、正しい箇所は一切変えないでください
- TEXTが空欄（フィラー除去済み）のIDは、必ずそのまま空欄で返してください
- 解説・前置き・後置きは一切不要です。指定フォーマットのみ出力してください

【出力フォーマット】（このフォーマットを厳収してください）
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
    print("[*] ローカルLLMによる文脈バッチ校正を開始します...")

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

                    # --- 【最重要】内部の単語（words）データの更新（最重要） ---
                    # formatter.py は words を使って字幕を組み立てるため、こちらも更新が必須。
                    # LLMはセグメント単位で校正するため、単語単位の厳密な対応は取れない。
                    # そこで「最初のwordに校正済みテキストをまとめて載せ、残りのwordは空文字化」する方式を採用。
                    # タイムスタンプ（start/end）は元のままなので、発話タイミングはズレません。
                    words = batch_seg.get("words", [])
                    non_empty_words = [w for w in words if w.get("word", "").strip()]

                    if corrected_text and non_empty_words:
                        # 最初の有効なwordにLLM校正後のテキストをまとめて書き込み
                        non_empty_words[0]["word"] = corrected_text
                        # 2つ目以降の有効なwordは空文字化（タイムスタンプは維持）
                        for w in non_empty_words[1:]:
                            w["word"] = ""

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