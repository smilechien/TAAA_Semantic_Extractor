import os
import io
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import networkx as nx
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from guess_language import guess_language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
#  App Setup
# -------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
#  Smart CSV Reader (handles garbled encodings)
# -------------------------------------------------------------
def smart_read_csv(file_bytes: bytes):
    try:
        # detect encoding first
        enc = chardet.detect(file_bytes)["encoding"] or "utf-8"
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
    except Exception:
        # fallback to utf-8-sig or cp950 (Big5)
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="cp950", errors="ignore")
    return df

# -------------------------------------------------------------
#  Mode Detection (Abstract vs Co-Word)
# -------------------------------------------------------------
def detect_mode(df: pd.DataFrame):
    """Detect whether this is abstract (1 text col) or co-word (multiple cols)."""
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(text_cols) == 1:
        return "abstract", text_cols[0]
    return "coword", text_cols

# -------------------------------------------------------------
#  Language Detection Helper
# -------------------------------------------------------------
def detect_language(df: pd.DataFrame):
    sample = " ".join(df.iloc[0].astype(str).tolist())[:200]
    lang = guess_language(sample) or "unknown"
    if lang.startswith("zh"):
        return "zh"
    elif lang.startswith("en"):
        return "en"
    return "other"

# -------------------------------------------------------------
#  Route: Homepage
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = TEMPLATE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html not found under app/templates/.</h3>")

# -------------------------------------------------------------
#  Route: CSV Preview
# -------------------------------------------------------------
@app.post("/preview", response_class=HTMLResponse)
async def preview_csv(file: UploadFile = File(...)):
    bytes_data = await file.read()
    df = smart_read_csv(bytes_data)

    # preview first 5 rows
    preview = df.head(5).to_html(index=False, escape=False)

    mode, cols = detect_mode(df)
    lang = detect_language(df)

    # bilingual label auto-selection
    if lang == "zh":
        if mode == "abstract":
            mode_label = "🧠 摘要模式"
            mode_hint = "偵測到單欄檔案 → 將使用摘要分析"
        else:
            mode_label = "🐾 共詞模式"
            mode_hint = "偵測到多欄檔案 → 將使用共詞分析"
    else:
        if mode == "abstract":
            mode_label = "🧠 Abstract Mode"
            mode_hint = "Detected single-column file → will use abstract analysis"
        else:
            mode_label = "🐾 Co-Word Mode"
            mode_hint = "Detected multi-column file → will use co-word analysis"

    html = f"""
    <h3>📄 Preview / 預覽</h3>
    {preview}
    <h4 style='color:{"green" if mode=="abstract" else "blue"};'>
        {mode_label}
    </h4>
    <p>{mode_hint}</p>
    <form action="/analyze_csv" enctype="multipart/form-data" method="post">
        <input type="hidden" name="mode" value="{mode}">
        <input type="file" name="file" accept=".csv" required>
        <button type="submit">🚀 Analyze</button>
    </form>
    """
    return HTMLResponse(html)

# -------------------------------------------------------------
#  Route: Analyze CSV
# -------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    bytes_data = await file.read()
    df = smart_read_csv(bytes_data)

    mode, cols = detect_mode(df)
    out_dir = BASE_DIR / "static" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "abstract":
        # -------- ABSTRACT MODE: TF-IDF + Similarity Graph --------
        text_col = cols if isinstance(cols, str) else cols[0]
        texts = df[text_col].astype(str).tolist()

        # compute tf-idf & cosine similarity
        vec = TfidfVectorizer(max_features=500)
        X = vec.fit_transform(texts)
        sim = cosine_similarity(X)

        terms = vec.get_feature_names_out()
        G = nx.Graph()
        for i, term1 in enumerate(terms):
            for j, term2 in enumerate(terms[i+1:], i+1):
                if sim[i][j] > 0.25:
                    G.add_edge(term1, term2, weight=sim[i][j])

        # Top-degree terms
        deg = dict(G.degree())
        top_terms = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]

        plt.figure(figsize=(8,5))
        plt.barh([t[0] for t in top_terms], [t[1] for t in top_terms])
        plt.title("Top 20 Abstract Terms (Degree)")
        plt.tight_layout()
        bar_path = out_dir / "theme_bar.png"
        plt.savefig(bar_path, dpi=150)
        plt.close()

        df_vertices = pd.DataFrame(list(G.nodes), columns=["term"])
        df_relations = pd.DataFrame([(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                                    columns=["source", "target", "weight"])
        df_vertices.to_csv(out_dir / "vertices.csv", index=False)
        df_relations.to_csv(out_dir / "relations.csv", index=False)

        # Theme CSV (simplified)
        df_theme = pd.DataFrame({"theme": [t[0] for t in top_terms]})
        df_theme.to_csv(out_dir / "themes.csv", index=False)

    else:
        # -------- CO-WORD MODE: Network from multiple columns --------
        text_cols = cols if isinstance(cols, list) else [cols]
        edges = []
        for _, row in df.iterrows():
            words = [str(w).strip() for w in row[text_cols].dropna()]
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    edges.append((w1, w2))

        G = nx.Graph()
        for u, v in edges:
            G.add_edge(u, v)

        deg = dict(G.degree())
        top_terms = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]

        plt.figure(figsize=(8,5))
        plt.barh([t[0] for t in top_terms], [t[1] for t in top_terms])
        plt.title("Top 20 Co-Word Terms (Degree)")
        plt.tight_layout()
        bar_path = out_dir / "theme_bar.png"
        plt.savefig(bar_path, dpi=150)
        plt.close()

        df_vertices = pd.DataFrame(list(G.nodes), columns=["term"])
        df_relations = pd.DataFrame(G.edges, columns=["source", "target"])
        df_vertices.to_csv(out_dir / "vertices.csv", index=False)
        df_relations.to_csv(out_dir / "relations.csv", index=False)

        df_theme = pd.DataFrame({"theme": [t[0] for t in top_terms]})
        df_theme.to_csv(out_dir / "themes.csv", index=False)

    # ---------------------------------------------------------
    # Result HTML
    # ---------------------------------------------------------
    result_html = f"""
    <h2>✅ Analysis Complete</h2>
    <p>Detected Mode: <strong>{mode}</strong></p>
    <h3>Download Results</h3>
    <ul>
      <li><a href="/static/outputs/vertices.csv" download>🔹 Vertices</a></li>
      <li><a href="/static/outputs/relations.csv" download>🔸 Relations</a></li>
      <li><a href="/static/outputs/themes.csv" download>🧩 Themes</a></li>
      <li><a href="/static/outputs/theme_bar.png" download>📊 Theme Bar</a></li>
    </ul>
    <footer>© 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v9.4</footer>
    """
    return HTMLResponse(result_html)
