import os, io, random
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langcodes import Language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

# -------------------------------------------------------------
#  App setup
# -------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_OUT = BASE_DIR / "static" / "outputs"
STATIC_OUT.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# -------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------
def smart_read_csv(file_bytes: bytes):
    """Auto-detect encoding (UTF-8 / UTF-8-SIG / CP950)."""
    for enc in ["utf-8", "utf-8-sig", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            continue
    raise ValueError("Unable to decode CSV file.")

def detect_mode(df: pd.DataFrame):
    """Simplified rule: 1-column → Abstract; multi-column → Co-word."""
    return ("abstract", df.columns[0]) if df.shape[1] == 1 else ("coword", list(df.columns))

def detect_language(df: pd.DataFrame):
    """Rough bilingual detection using langcodes (Render-safe)."""
    sample = " ".join(df.iloc[0].astype(str).tolist())[:200]
    try:
        lang_code = str(Language.find(sample))
    except Exception:
        lang_code = "en"
    if "zh" in lang_code:
        return "zh"
    elif "en" in lang_code:
        return "en"
    return "other"

# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    path = TEMPLATE_DIR / "index.html"
    return HTMLResponse(path.read_text("utf-8")) if path.exists() else HTMLResponse("<h3>❌ Missing index.html</h3>")

# -------------------------------------------------------------
@app.post("/preview", response_class=HTMLResponse)
async def preview_csv(file: UploadFile = File(...)):
    df = smart_read_csv(await file.read())
    mode, cols = detect_mode(df)
    lang = detect_language(df)

    color = "green" if mode == "abstract" else "blue"
    label_zh = "🧠 摘要模式" if mode == "abstract" else "🐾 共詞模式"
    hint_zh  = "偵測到單欄 → 使用摘要分析" if mode == "abstract" else "偵測到多欄 → 使用共詞分析"
    label_en = "🧠 Abstract Mode" if mode == "abstract" else "🐾 Co-Word Mode"
    hint_en  = "Detected single column → Abstract analysis" if mode == "abstract" else "Detected multi-column → Co-Word analysis"
    label, hint = (label_zh, hint_zh) if lang == "zh" else (label_en, hint_en)

    preview = df.head(5).to_html(index=False, escape=False)
    return HTMLResponse(f"""
        <h3>📄 Preview</h3>{preview}
        <h4 style='color:{color}'>{label}</h4><p>{hint}</p>
        <form action="/analyze_csv" enctype="multipart/form-data" method="post">
          <input type="hidden" name="mode" value="{mode}">
          <input type="file" name="file" accept=".csv" required>
          <button type="submit">🚀 Analyze</button>
        </form>
    """)

# -------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    df = smart_read_csv(await file.read())
    mode, cols = detect_mode(df)

    # ---------- Build Graph ----------
    if mode == "abstract":
        text_col = cols if isinstance(cols, str) else cols[0]
        texts = df[text_col].astype(str).tolist()
        vec = TfidfVectorizer(max_features=500)
        X = vec.fit_transform(texts)
        sim = cosine_similarity(X)
        terms = vec.get_feature_names_out()
        G = nx.Graph()
        for i, t1 in enumerate(terms):
            for j, t2 in enumerate(terms[i+1:], i+1):
                if sim[i][j] > 0.25:
                    G.add_edge(t1, t2, weight=float(sim[i][j]))
    else:
        edges = []
        for _, row in df.iterrows():
            words = [str(w).strip() for w in row.dropna()]
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    edges.append((w1, w2))
        G = nx.Graph()
        for u, v in edges:
            G.add_edge(u, v)

    # ---------- Degree & Top Terms ----------
    deg = dict(G.degree())
    top_terms = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]

    # ---------- Bar Chart ----------
    plt.figure(figsize=(8,5))
    plt.barh([t[0] for t in top_terms], [t[1] for t in top_terms])
    plt.title(f"Top 20 Terms ({mode.title()} Mode)")
    plt.tight_layout()
    plt.savefig(STATIC_OUT / "theme_bar.png", dpi=150)
    plt.close()

    # ---------- Interactive Plotly Scatter ----------
    terms = [t[0] for t in top_terms]
    vec = TfidfVectorizer()
    X = vec.fit_transform(terms)
    coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
    n_clusters = min(6, max(2, len(terms)//4))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(coords)
    df_plot = pd.DataFrame({
        "x": coords[:,0],
        "y": coords[:,1],
        "term": terms,
        "cluster": [f"Theme {l+1}" for l in labels]
    })

    fig = px.scatter(
        df_plot, x="x", y="y", color="cluster", text="term",
        title=f"Theme Scatter ({mode.title()} Mode)",
        hover_data=["term"],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(textposition="top center", marker=dict(size=16, opacity=0.8))
    fig.update_layout(showlegend=True, height=600, width=700)
    fig.write_html(str(STATIC_OUT / "theme_scatter.html"), include_plotlyjs="cdn")

    # ---------- Save CSVs ----------
    pd.DataFrame(G.nodes, columns=["term"]).to_csv(STATIC_OUT / "vertices.csv", index=False)
    pd.DataFrame(G.edges, columns=["source","target"]).to_csv(STATIC_OUT / "relations.csv", index=False)
    pd.DataFrame({"theme":[t[0] for t in top_terms]}).to_csv(STATIC_OUT / "themes.csv", index=False)

    return HTMLResponse(f"""
        <h2>✅ Analysis Complete</h2>
        <p>Detected Mode: <b>{mode}</b></p>
        <h3>Download Results</h3>
        <ul>
          <li><a href="/static/outputs/vertices.csv" download>🔹 Vertices</a></li>
          <li><a href="/static/outputs/relations.csv" download>🔸 Relations</a></li>
          <li><a href="/static/outputs/themes.csv" download>🧩 Themes</a></li>
          <li><a href="/static/outputs/theme_bar.png" download>📊 Theme Bar</a></li>
          <li><a href="/static/outputs/theme_scatter.html" target="_blank">🌈 Theme Scatter (Interactive)</a></li>
        </ul>
        <footer>© 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v9.7 stable</footer>
    """)
