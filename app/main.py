# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer v7.0
# Auto-detects abstract vs co-word CSVs
# Handles UTF-8 / Big5 encoding automatically
# Adds built-in sample datasets
# ============================================================

import os, time, random
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langdetect import detect
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
import networkx as nx

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ------------------------------------------------------------
# 🔑 Safe OpenAI init
# ------------------------------------------------------------
def init_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️ No OpenAI key found, TF-IDF only.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        print("✅ OpenAI client ready.")
        return client
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")
        return None

client = init_openai_client()

# ------------------------------------------------------------
# 🧠 Utility helpers
# ------------------------------------------------------------
def read_csv_safely(path):
    """Try UTF-8 first, then Big5/CP950 fallback."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="cp950")
        except Exception as e:
            raise RuntimeError(f"CSV encoding error: {e}")

def detect_language(texts):
    for txt in texts:
        if isinstance(txt, str) and len(txt.strip()) > 10:
            try:
                return detect(txt)
            except Exception:
                continue
    return "en"

def safe_tfidf(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
    clusters = km.fit_predict(X)
    return pd.DataFrame({"text": texts, "theme": [f"Theme_{c+1}" for c in clusters]})

def gpt_semantic_analysis(texts):
    results = []
    for t in texts:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a research text classifier."},
                        {"role": "user", "content": f"Classify this abstract into a short theme label:\n\n{t}"}
                    ],
                    timeout=25
                )
                results.append(response.choices[0].message.content.strip())
                break
            except (APIConnectionError, RateLimitError, APITimeoutError):
                time.sleep(1 + attempt)
            except Exception:
                results.append("Uncategorized")
                break
    return pd.DataFrame({"text": texts, "theme": results})

def plot_scatter_theme(df, lang, output_path):
    plt.figure(figsize=(8,6))
    matplotlib.rcParams["font.family"] = "Noto Sans TC" if lang.startswith("zh") else "Segoe UI"
    theme_counts = df["theme"].value_counts().head(20)
    colors = plt.cm.tab20(np.linspace(0,1,len(theme_counts)))
    plt.scatter(range(len(theme_counts)), theme_counts.values, s=100, c=colors)
    for i, (theme, count) in enumerate(theme_counts.items()):
        plt.text(i, count+0.2, theme, rotation=45, ha="right", fontsize=9)
    plt.title("Top 20 Themes" if lang.startswith("en") else "前 20 主題")
    plt.xlabel("Themes" if lang.startswith("en") else "主題")
    plt.ylabel("Frequency" if lang.startswith("en") else "出現次數")
    plt.tight_layout(); plt.savefig(output_path); plt.close()

# ------------------------------------------------------------
# 🧮 Analyzer Core
# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    file_path = BASE_DIR / "static" / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        df = read_csv_safely(file_path)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    # Auto-detect mode
    if len(df.columns) == 1:
        mode = "abstract"
        text_col = df.columns[0]
    else:
        mode = "coword"

    # ---------------------- Abstract Mode ----------------------
    if mode == "abstract":
        texts = df[text_col].fillna("").astype(str).tolist()
        lang = detect_language(texts)
        use_gpt = client is not None
        result_df = gpt_semantic_analysis(texts) if use_gpt else safe_tfidf(texts)
        engine = "GPT" if use_gpt else "TF-IDF"

        themed_path = BASE_DIR / "static" / f"{Path(file.filename).stem}_themes.csv"
        result_df.to_csv(themed_path, index=False, encoding="utf-8-sig")
        plot_path = BASE_DIR / "static" / f"{Path(file.filename).stem}_plot.png"
        plot_scatter_theme(result_df, lang, plot_path)

        return HTMLResponse(f"""
        <html><body style='font-family:Segoe UI,Noto Sans TC;text-align:center;margin:40px;'>
        <h2>✅ Abstract Analysis Complete</h2>
        <p>Engine: <b>{engine}</b></p>
        <img src='/static/{plot_path.name}' width='600'><br><br>
        <a href='/static/{themed_path.name}' download>📥 Download Themes CSV</a><br><br>
        <a href='/'>🏠 Back to Home</a></body></html>
        """)

    # ---------------------- Co-Word Mode ----------------------
    else:
        cols = [c for c in df.columns if df[c].dtype == object]
        edges = []
        for _, row in df.iterrows():
            words = [str(w).strip() for w in row.dropna().tolist() if len(str(w)) > 1]
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    edges.append((words[i], words[j]))
        G = nx.Graph(); G.add_edges_from(edges)
        nx.write_weighted_edgelist(G, BASE_DIR / "static" / "coword_edges.txt")

        return HTMLResponse(f"""
        <html><body style='font-family:Segoe UI,Noto Sans TC;text-align:center;margin:40px;'>
        <h2>✅ Co-Word Network Generated</h2>
        <p>{len(G.nodes())} nodes, {len(G.edges())} edges</p>
        <a href='/static/coword_edges.txt' download>📥 Download Edge List</a><br><br>
        <a href='/'>🏠 Back to Home</a></body></html>
        """)

# ------------------------------------------------------------
# 🏠 Home & Sample Routes
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = BASE_DIR / "templates" / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html missing under app/templates/.</h3>")

@app.get("/samples", response_class=HTMLResponse)
async def download_samples():
    """Serve sample abstract & co-word CSVs for demo."""
    abstract_sample = BASE_DIR / "static" / "abstract_sample.csv"
    coword_sample = BASE_DIR / "static" / "coword_sample.csv"
    # Generate once if not exist
    if not abstract_sample.exists():
        pd.DataFrame({"abstract": [
            "Artificial intelligence improves clinical diagnosis accuracy.",
            "Machine learning aids cancer detection in radiology.",
            "Nursing management enhances patient safety and care quality."
        ]}).to_csv(abstract_sample, index=False, encoding="utf-8-sig")
    if not coword_sample.exists():
        pd.DataFrame({
            "Keyword1": ["AI","Machine Learning","Patient Safety"],
            "Keyword2": ["Diagnosis","Cancer","Nursing"],
            "Keyword3": ["Radiology","Prediction","Hospital"]
        }).to_csv(coword_sample, index=False, encoding="utf-8-sig")

    return HTMLResponse(f"""
    <html><body style='font-family:Segoe UI,Noto Sans TC;text-align:center;margin:40px;'>
    <h2>📊 Download Sample Datasets</h2>
    <a href='/static/abstract_sample.csv' download>🧠 Abstract Sample</a><br>
    <a href='/static/coword_sample.csv' download>🐾 Co-Word Sample</a><br><br>
    <a href='/'>🏠 Back</a>
    </body></html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
