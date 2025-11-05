# ============================================================
# 🧠 utils.py — GPT Semantic Keyword Extraction Utilities
# ============================================================

import openai
import time

def extract_keywords(text: str, model: str = "gpt-4o-mini", retries: int = 3, sleep_time: float = 2.0):
    """
    Extract 10 semantic keywords (Chinese or English) using GPT API.
    Returns a clean keyword string separated by 、.
    """
    if not text or text.strip() == "":
        return ""

    prompt = f"請從以下摘要中萃取10個具語義代表性的學術關鍵詞，以繁體中文或英文均可，並用頓號（、）分隔：\n\n{text}"

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"❌ GPT error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(sleep_time)
    return ""
