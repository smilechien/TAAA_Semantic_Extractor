# ================================================================
# 🌐 TAAA Semantic Extractor
#  - Compatible with OpenAI Python SDK v1.x+
#  - FastAPI backend for Render deployment
#  - Multilingual semantic keyword extraction (auto-detect)
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
from openai import OpenAI
import tempfile
import os

# ----------------------------
# 🔧 Initialize
# ----------------------------
app = FastAPI(
    title="TAAA Semantic Extractor",
    description="Upload abstracts and extract 10 semantic keywords via GPT-4o-mini (multilingual auto-detect)",
    version="2.0.0"
)

# Instantiate OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# 🏠 Home route
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        html = open("index.html", "r", encoding="utf-8").read()
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Error loading page:</h3><p>{e}</p>")


# ----------------------------
# 📤 CSV upload route
# ----------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"❌ Unable to read CSV: {e}"}

    if "abstract" not in df.columns:
        return {"error": "Missing 'abstract' column. Please include a column named 'abstract'."}

    df["keywords"] = df["abstract"].apply(lambda x: extract_keywords(str(x)))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")

    return FileResponse(
        tmp.name,
        media_type="text/csv",
        filename="taaa_keywords.csv"
    )


# ----------------------------
# 🧠 Keyword extraction function
# ----------------------------
def extract_keywords(text: str) -> str:
    """Call GPT-4o-mini to extract 10 representative semantic keywords."""
    if not text or pd.isna(text):
        return ""

    prompt = (
        "請根據以下摘要內容，萃取 10 個具語義代表性的學術關鍵詞，"
        "可為繁體中文或英文（依原文語言自動判斷），並用頓號（、）分隔。"
        "若摘要為非中英文語言（如日文、西班牙文、法文等），請自動以相同語言回覆。\n\n"
        f"{text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"


# ----------------------------
# 🚀 Local dev entry point
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
