import ollama

# =====================================================================
# 初期設定エリア：使用するローカルLLMモデルの定義
# =====================================================================
# 将来モデルを変更したい場合や、軽量モデル（gemma2:2bなど）を試したい場合はここを書き換えてください。
LLM_MODEL_NAME = "hf.co/mmnga/Llama-3-ELYZA-JP-8B-gguf"
# =====================================================================


def generate_summary(segments: list) -> str:
    """
    【第2工程 - 下準備】
    動画全体のタイムスタンプ付きテキストから、AIに全体のストーリー（文脈）を把握させるための
    簡単なあらすじ（要約）を自動生成します。
    
    これを行うことで、後半のバッチ処理（部分的な校正）の際に、AIが前後の文脈を見失わずに
    正確な補正（専門用語のハメ直しなど）ができるようになります。
    """
    print("[*] ローカルLLM（ELYZA）に動画全体の文脈（あらすじ）を学習させています...")
    
    # Whisperの全セグメントから、テキストだけを抽出して一本のタイムラインにする
    full_text_timeline = ""
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            full_text_timeline += f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {text}\n"
            
    # テキストが空の場合は、空の要約を返す
    if not full_text_timeline:
        return "（音声データが空のため、あらすじはありません）"
        
    # AIへのプロンプト（命令書）
    prompt = f"""
あなたは優秀な動画編集エディターです。
以下に提示する「タイムスタンプ付きの文字起こしデータ」の全体を読み、この動画が『何について話している動画なのか』を、300文字程度の日本語のあらすじ（要約）としてまとめてください。
この要約は、後ほど行う「誤字脱字の校正処理」の極めて重要な文脈データとして使用します。

【文字起こしデータ】
{full_text_timeline}

【出力フォーマット】
余計な挨拶や前置きは一切排除し、要約した文章だけをダイレクトに出力してください。
"""

    try:
        # 最上部で定義した定数 LLM_MODEL_NAME を使用してOllamaを呼び出し
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        summary = response["message"]["content"].strip()
        print("[+] 全体文脈の把握が完了しました。")
        return summary
        
    except Exception as e:
        # 万が一LLMの呼び出しに失敗しても、システム全体をクラッシュさせずに生存させる
        print(f"[*] 全体要約の生成中にエラーが発生しました（処理は継続します）: {e}")
        return "（エラーのため全体要約の生成をスキップしました）"


def refine_context_with_llm(segments: list, dictionary_terms: list, summary: str) -> list:
    """
    【第2工程 - 本番】
    動画全体のあらすじ（summary）と、専門用語辞書（dictionary_terms）を頭に入れたAIが、
    各セグメントのタイムスタンプ（ミリ秒のインデックス番号）を完全に維持したまま、
    日本語として不自然な聞き間違い、漢字の誤変換、固有名詞を超高精度にバッチ校正します。
    """
    print("[*] ローカルLLMによる文脈バッチ校正を開始します...")
    
    refined_segments = []
    current_batch = []
    
    # 辞書データをAIに分かりやすいテキスト形式に整形
    dict_text = "\n".join([f"- {term}" for term in dictionary_terms]) if dictionary_terms else "（登録なし）"
    
    # 全セグメントを1つずつ精査していく
    for index, seg in enumerate(segments):
        current_batch.append(seg)
        
        # 判定用フラグ：今回のセグメントに句読点が含まれているか
        has_punctuation = "。" in seg["text"] or "、" in seg["text"]
        # または、現在のバッチが10件溜まったか、あるいは最後のセグメントか
        is_batch_full = len(current_batch) >= 10 or index == len(segments) - 1
        
        # 「文節の区切り（句読点）」が来たタイミング、またはバッチ上限に達したらLLMへ送信
        if has_punctuation or is_batch_full:
            
            # AIが元のタイムスタンプの「箱（インデックス）」を見失わないよう、JSONライクな構造テキストを作る
            batch_prompt_text = ""
            for batch_seg in current_batch:
                batch_prompt_text += f"ID: {batch_seg['id']} | TEXT: {batch_seg['text']}\n"
                
            # AIへの超精密なプロンプト（命令書）
            prompt = f"""
あなたはテレビ番組やYouTubeのテロップ制作を行う、極めて優秀な編集エディターです。
提示された【補正対象データ】のTEXT部分に含まれる、音声認識特有の誤変換、漢字の間違い、同音異義語のミスを、動画の【全体あらすじ】および【専門用語辞書】をベースに文脈を考慮して美しく校正してください。

【動画の全体あらすじ（文脈）】
{summary}

【専門用語辞書（固有名詞の正解リスト）】
{dict_text}

【最重要ルール（厳守事項）】
1. 出力は、必ず元の【補正対象データ】と同じ「ID: 番号 | TEXT: 校正後の文字列」のフォーマットを1行ずつ維持してください。
2. IDの番号は、絶対に改変したり統合したりせず、そのまま返してください。
3. フィラー（えっと、あのー、等）が原因で元のTEXTが「空文字（空欄）」になっているIDは、時間を狂わせないための重要な空箱です。勝手に削除せず、そのまま「TEXT: 」（あるいは空欄）として出力してください。
4. 解説や「修正しました」などの前置き・後置きは一切出力せず、指定フォーマットのデータのみを出力してください。

【補正対象データ】
{batch_prompt_text}
"""

            try:
                # 最上部で定義した定数 LLM_MODEL_NAME を使用してOllamaで校正を実行
                response = ollama.chat(
                    model=LLM_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # AIからの返答を解析（行ごとにバラす）
                llm_lines = response["message"]["content"].strip().split("\n")
                
                # AIの出力を元のセグメント構造にハメ直す
                for batch_seg in current_batch:
                    target_id = batch_seg["id"]
                    corrected_text = batch_seg["text"] # 万が一見つからなかった場合のフォールバック（現状維持）

                    # テキスト修正前後の確認のため
                    old_text = batch_seg["text"] # 修正前のテキストを保存
                    new_text = old_text          # デフォルトは現状維持
                    
                    # AIの出力行から、該当するIDの行を探し出す
                    for line in llm_lines:
                        if f"ID: {target_id} " in line or f"ID:{target_id}" in line:
                            if "TEXT:" in line:
                                # 「TEXT:」より後ろの文字列を正解として抽出
                                corrected_text = line.split("TEXT:", 1)[1].strip()
                                break
                    
                    #  修正前後でテキストが変わっていた場合のみ出力
                    if old_text != new_text:
                        print(f"\n[修正検出 ID:{target_id}]", f"  BEFORE: {old_text}", f"  AFTER : {new_text}")
                                
                    # タイムスタンプやIDなどの重要データはそのままに、テキストだけをAIの綺麗な文字に差し替える
                    batch_seg["text"] = corrected_text
                    refined_segments.append(batch_seg)
                    
            except Exception as e:
                # 【堅牢なエラーハンドリング】
                # Ollamaの通信エラーやメモリ不足が起きても、文字起こしタスクそのものは絶対にクラッシュさせない。
                # 安全のため、このバッチ（最大10行分）はWhisperの生データをそのまま採用して生存ルートを確保する。
                print(f"[*] ブロック処理中にエラーが発生しました（安全のため生データを維持します）: {e}")
                for batch_seg in current_batch:
                    refined_segments.append(batch_seg)
                    
            # 処理が終わったバッチを空にして、次のグループ（10件分）に備える
            current_batch = []
            
    print("[+] ローカルLLMによる文脈校正がすべて完了しました。")
    return refined_segments