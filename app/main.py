from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import openai
import tempfile
import os

app = FastAPI(title="TAAA Semantic Extractor",
              description="Upload abstracts and extract 10 semantic keywords via GPT-4o-mini",
              version="1.0.0")

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/", response_class=HTMLResponse)
def home():
    html = open("index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "abstract" not in df.columns:
        return {"error": "Missing 'abstract' column."}

    df["keywords"] = df["abstract"].apply(lambda x: extract_keywords(x))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    return FileResponse(tmp.name, media_type="text/csv", filename="taaa_keywords.csv")

def extract_keywords(text):
    if not text or pd.isna(text):
        return ""
    prompt = f"請從以下摘要中萃取10個具語義代表性的學術關鍵詞，以繁體中文或英文均可，並用頓號（、）分隔：\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"
