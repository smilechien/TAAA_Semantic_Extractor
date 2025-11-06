# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v6.1)
# FastAPI + OpenAI + TF-IDF Fallback + Multilingual UI
# ============================================================

import os
import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langdetect import detect
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
import networkx as nx

# ------------------------------------------------------------
# ⚙️ App Initialization
# ------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

# Mount static folder for downloadable CSVs
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ------------------------------------------------------------
# 🧠 OpenAI Client Initialization (with safe retry logic)
# ------------------------------------------------------------
def init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️ No OpenAI API key found, TF-IDF mode only.")
        return None
    try:
        client = OpenAI(api_key=api_key)  # ✅ no proxies arg (compatible with v1.52+)
        _ = client.models.list()
        print("✅ OpenAI client initialized successfully.")
        return client
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")
        return None

client = init_openai_client()

# ------------------------------------------------------------
# 🧩 Routes
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the front-end HTML from templates/index.html.
    """
    index_path = BASE_DIR / "templates" / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html not found under app/templates/.</h3>")

@app.get("/health", response_class=HTMLResponse)
async def health_check():
    """Simple Render health endpoint"""
    return HTMLResponse("✅ OK")

# ------------------------------------------------------------
# 🧮 Utility Functions
# ------------------------------------------------------------
def detect_language(texts):
    """Detect language for labeling fonts."""
    for txt in texts:
        if isinstance(txt, str) and len(txt.strip()) > 10:
            try:
                return detect(txt)
            except Exception:
                continue
    return "en"

def safe_tfidf(texts, n_clusters=5):
    """Simple TF-IDF + KMeans fallback."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
    clusters = km.fit_predict(X)
    df = pd.DataFrame({
        "text": texts,
        "theme": [f"Theme_{c+1}" for c in clusters]
    })
    return df

def gpt_semantic_analysis(texts):
    """Use OpenAI GPT-4o mini model for semantic clustering."""
    results = []
    for t in texts:
        for attempt in range(3):  # auto-retry
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a research text classifier."},
                        {"role": "user", "content": f"Classify this abstract into a short theme label:\n\n{t}"}
                    ],
                    timeout=20
                )
                theme = response.choices[0].message.content.strip()
                results.append(theme)
                break
            except (APIConnectionError, RateLimitError, APITimeoutError):
                time.sleep(1 + attempt)
            except Exception:
                results.append("Uncategorized")
                break
    return pd.DataFrame({"text": texts, "theme": results})

# ------------------------------------------------------------
# 🎨 Visualization Function
# ------------------------------------------------------------
def plot_scatter_theme(df, lang, output_path):
    """Generate scatter plot of top 20 themes or terms."""
    plt.figure(figsize=(8, 6))
    matplotlib.rcParams["font.family"] = "Noto Sans TC" if lang.startswith("zh") else "Segoe UI"

    theme_counts = df["theme"].value_counts().head(20)
    colors = plt.cm.tab20(np.linspace(0, 1, len(theme_counts)))

    plt.scatter(range(len(theme_counts)), theme_counts.values, s=100, c=colors)
    for i, (theme, count) in enumerate(theme_counts.items()):
        plt.text(i, count + 0.2, theme, rotation=45, ha="right", fontsize=9)

    plt.title("Top 20 Themes" if lang.startswith("en") else "前 20 主題")
    plt.xlabel("Themes" if lang.startswith("en") else "主題")
    plt.ylabel("Frequency" if lang.startswith("en") else "出現次數")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------------------------------------
# 🚀 Main Analysis Endpoint
# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...), mode: str = Form("abstract")):
    """
    Core endpoint — performs semantic or co-word analysis.
    """
    file_path = BASE_DIR / "static" / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    # Text source column detection
    text_col = None
    for c in df.columns:
        if "abstract" in c.lower() or "text" in c.lower():
            text_col = c
            break
    if not text_col:
        return HTMLResponse("<h3>❌ No text/abstract column found.</h3>")

    texts = df[text_col].fillna("").tolist()
    lang = detect_language(texts)

    # Engine decision
    use_gpt = False
    if "smilechien" in file.filename.lower() and client:
        use_gpt = True
    elif "chatgpt" in os.getenv("RENDER", "").lower():
        use_gpt = True
    else:
        use_gpt = client is not None

    # Semantic analysis
    if use_gpt and client:
        print("🔹 Using GPT semantic engine")
        result_df = gpt_semantic_analysis(texts)
        engine_mode = "GPT"
    else:
        print("🔹 Using TF-IDF fallback engine")
        result_df = safe_tfidf(texts)
        engine_mode = "TF-IDF"

    # Save themed CSV
    themed_path = BASE_DIR / "static" / f"{Path(file.filename).stem}_themed.csv"
    result_df.to_csv(themed_path, index=False, encoding="utf-8-sig")

    # Plot scatter
    plot_path = BASE_DIR / "static" / f"{Path(file.filename).stem}_themes.png"
    plot_scatter_theme(result_df, lang, plot_path)

    return HTMLResponse(f"""
    <html><body style='font-family:Segoe UI,Noto Sans TC;text-align:center;margin:40px;'>
      <h2>✅ Analysis Complete</h2>
      <p>Engine: <b>{engine_mode}</b></p>
      <img src='/static/{plot_path.name}' width='600'><br><br>
      <a href='/static/{themed_path.name}' download>📥 Download Themed CSV</a><br><br>
      <a href='/'>🏠 Back to Home</a>
    </body></html>
    """)

# ------------------------------------------------------------
# 🏁 Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
