from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import openai
import tempfile
import os
import re

app = FastAPI(
    title="TAAA Semantic Extractor",
    description="Adaptive multilingual semantic keyword extractor using GPT-4o-mini",
    version="4.0.0"
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/", response_class=HTMLResponse)
def home():
    html = open("index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...), bilingual: str = Form("false")):
    df = pd.read_csv(file.file)
    if "abstract" not in df.columns:
        return {"error": "Missing 'abstract' column."}

    bilingual_mode = (bilingual.lower() == "true")

    results = df["abstract"].apply(lambda x: extract_keywords_global(x, bilingual_mode))
    df["language"], df["keywords"] = zip(*results)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    return FileResponse(tmp.name, media_type="text/csv", filename="taaa_keywords_global.csv")

# 🌍 Language detection
def detect_language(text: str) -> str:
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_count = len(re.findall(r'[A-Za-z]', text))
    jp_count = len(re.findall(r'[\u3040-\u30ff]', text))
    kr_count = len(re.findall(r'[\uac00-\ud7af]', text))
    accented = len(re.findall(r'[áéíóúüñçàèùßαβγабвг]', text, flags=re.IGNORECASE))

    if zh_count > 0 and en_count > 0:
        return "mixed"

    lang_scores = {
        "chinese": zh_count,
        "english": en_count,
        "japanese": jp_count,
        "korean": kr_count,
        "other": accented
    }
    lang = max(lang_scores, key=lang_scores.get)
    return lang if lang_scores[lang] > 0 else "unknown"

# 🧠 Optimized multilingual prompt templates
PROMPTS = {
    "chinese": "請從以下中文摘要中萃取10個具語義代表性的學術關鍵詞，強調研究主題、方法與核心概念。以頓號（、）分隔。若為技術名詞，請保持原文或以英文標示。\n\n摘要：\n{text}",
    "english": "Extract 10 semantically representative academic keywords from the following English abstract. Focus on scientific themes, methods, and key terminology. Separate keywords by commas.\n\nAbstract:\n{text}",
    "japanese": "次の日本語の要約から、研究のテーマ、方法、主要な概念を表す代表的な学術キーワードを10個抽出してください。英語の専門用語が必要な場合は併記してください。キーワードは読点（、）で区切ってください。\n\n要約：\n{text}",
    "korean": "다음 한국어 초록에서 연구 주제, 방법, 핵심 개념을 대표하는 학술적 주요 키워드 10개를 추출하세요. 필요할 경우 영어 기술 용어를 함께 제시하세요. 키워드는 쉼표(,)로 구분하세요.\n\n초록:\n{text}",
    "spanish": "Extrae 10 palabras clave académicas representativas del siguiente resumen en español. Enfócate en el tema de investigación, metodología y conceptos principales. Separa las palabras clave con comas. Si existen términos técnicos, puedes mantenerlos en inglés.\n\nResumen:\n{text}",
    "french": "Extrayez 10 mots-clés académiques représentatifs du résumé suivant en français. Mettez l’accent sur le sujet de recherche, la méthode et les concepts clés. Séparez les mots-clés par des virgules. Les termes techniques peuvent rester en anglais.\n\nRésumé :\n{text}",
    "mixed": "The following abstract contains both Chinese and English text. Please extract 10 representative academic keywords in English only, summarizing the research focus and technical themes. Separate by commas.\n\nAbstract:\n{text}",
    "other": "The following abstract is written in {language}. Extract 10 representative academic keywords in the same language if possible. If technical or scientific, provide English equivalents in parentheses. Separate keywords by commas or the natural punctuation of the language.\n\nAbstract:\n{text}",
    "bilingual": "請根據以下摘要，分別以繁體中文與英文各萃取10個具語義代表性的學術關鍵詞。請輸出格式如下：\n中文關鍵詞：...(以頓號「、」分隔)\nEnglish keywords: ...(以逗號「,」分隔)\n\n摘要內容：\n{text}"
}

# ✨ Core function
def extract_keywords_global(text, bilingual_mode=False):
    if not text or pd.isna(text):
        return ("unknown", "")

    lang = detect_language(text)
    prompt = ""

    if bilingual_mode:
        prompt = PROMPTS["bilingual"].format(text=text)
    else:
        if lang in PROMPTS:
            prompt = PROMPTS[lang].format(text=text)
        else:
            prompt = PROMPTS["other"].format(language=lang, text=text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        keywords = response["choices"][0]["message"]["content"].strip()
        return (lang, keywords)
    except Exception as e:
        return (lang, f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
