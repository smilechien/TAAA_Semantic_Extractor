# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v9.3)
# Handles multilingual CSVs with auto-mode detection.
# ================================================================

import io, os, csv, tempfile
import chardet, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ------------------------------------------------
# 1️⃣ App setup
# ------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------------------------
# 2️⃣ Safe OpenAI init
# ------------------------------------------------
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"⚠️ OpenAI init failed: {e}")
    client = None

# ------------------------------------------------
# 3️⃣ Smart CSV reader (encoding + delimiter)
# ------------------------------------------------
def smart_read_csv(content_bytes: bytes):
    enc = chardet.detect(content_bytes).get("encoding") or "utf-8"
    if enc.lower().startswith("big5"):
        enc = "cp950"           # Taiwanese Big5 fallback
    decoded = content_bytes.decode(enc, errors="ignore")
    buf = io.StringIO(decoded)
    sample = buf.read(2048)
    buf.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except csv.Error:
        sep = "\t" if sample.count("\t") > sample.count(",") else ","
    df = pd.read_csv(buf, sep=sep, engine="python")
    return df, enc

# ------------------------------------------------
# 4️⃣ Home route
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    idx = BASE_DIR / "templates" / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html not found under app/templates/.</h3>")

# ------------------------------------------------
# 5️⃣ Preview endpoint
# ------------------------------------------------
@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df, enc = smart_read_csv(content)
        df = df.dropna(how="all").head(5)
        return JSONResponse({
            "encoding": enc,
            "columns": list(df.columns),
            "rows": df.fillna("").astype(str).values.tolist()
        })
    except Exception as e:
        return JSONResponse({"error": f"Preview failed: {str(e)}"})

# ------------------------------------------------
# 6️⃣ Analysis endpoint
# ------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df, enc = smart_read_csv(content)
    except Exception as e:
        return HTMLResponse(f"<h3>CSV read error: {str(e)}</h3>")

    df = df.dropna(how="all")
    mode = "abstract" if len(df.columns) == 1 else "coword"

    tmpdir = Path(tempfile.mkdtemp())
    out_vertices = tmpdir / "vertices.csv"
    out_relations = tmpdir / "relations.csv"
    out_themes = tmpdir / "themes.csv"
    bar_path = tmpdir / "theme_bar.png"
    scatter_path = tmpdir / "theme_scatter.png"

    # ============================
    # 🧠 ABSTRACT MODE
    # ============================
    if mode == "abstract":
        col = df.columns[0]
        docs = df[col].astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(docs)
        km = KMeans(n_clusters=min(5, len(docs)), n_init=10, random_state=42)
        labels = km.fit_predict(X)

        df["theme"] = [f"Theme_{l+1}" for l in labels]
        df.to_csv(out_themes, index=False, encoding="utf-8-sig")

        # term weights (optional vertices)
        terms = vectorizer.get_feature_names_out()
        term_weights = X.toarray().sum(axis=0)
        vertices = pd.DataFrame({"term": terms, "weight": term_weights})
        vertices.to_csv(out_vertices, index=False, encoding="utf-8-sig")

        # simple bar plot
        theme_counts = df["theme"].value_counts()
        plt.figure(figsize=(6, 4))
        theme_counts.plot(kind="barh", color="#2ecc71")
        plt.title("Theme Distribution (Abstract Mode)")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()

        # scatter
        plt.figure(figsize=(5, 4))
        plt.scatter(range(len(docs)), labels, c=labels, cmap="viridis")
        plt.title("Theme Scatter (Abstract Mode)")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()

    # ============================
    # 🐾 CO-WORD MODE
    # ============================
    else:
        num_df = df.select_dtypes(include=["number"])
        if num_df.empty:
            num_df = df.apply(lambda c: pd.factorize(c)[0])
        corr = num_df.corr().fillna(0)
        terms = corr.columns.tolist()

        # vertices (degree)
        vertices = pd.DataFrame({"term": terms, "degree": corr.abs().sum()})
        vertices = vertices.sort_values("degree", ascending=False).head(20)

        # edges
        edges = []
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                w = corr.iat[i, j]
                if abs(w) > 0.3:
                    edges.append({"source": terms[i], "target": terms[j], "weight": round(w, 3)})
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(out_relations, index=False, encoding="utf-8-sig")

        # clusters
        G = nx.Graph()
        for e in edges:
            G.add_edge(e["source"], e["target"], weight=e["weight"])
        clusters = list(nx.connected_components(G))

        theme_map = []
        for idx, comp in enumerate(clusters):
            for t in comp:
                theme_map.append({"theme": f"Cluster_{idx+1}", "term": t})
        theme_df = pd.DataFrame(theme_map)

        vertices = vertices.merge(theme_df, on="term", how="left").fillna("Unassigned")
        vertices.to_csv(out_vertices, index=False, encoding="utf-8-sig")
        theme_df.to_csv(out_themes, index=False, encoding="utf-8-sig")

        # --- theme bar (aggregated) ---
        theme_bar = (
            vertices.groupby("theme", as_index=False)["degree"]
            .mean().sort_values("degree", ascending=True)
        )
        plt.figure(figsize=(8, 4))
        plt.barh(theme_bar["theme"], theme_bar["degree"], color="#27ae60")
        plt.title("Top Themes (Average Degree per Cluster)")
        plt.xlabel("Average Degree")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()

        # --- scatter with theme label ---
        plt.figure(figsize=(6, 4))
        plt.scatter(range(len(vertices)), vertices["degree"], c="#2980b9")
        for i, (term, theme) in enumerate(zip(vertices["term"], vertices["theme"])):
            plt.text(i, vertices["degree"].iloc[i], f"{theme}:{term}",
                     fontsize=8, rotation=45)
        plt.title("Theme Scatter (Co-Word Mode)")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()

    # ------------------------------------------------
    # Build HTML summary
    # ------------------------------------------------
    html = f"""
    <h2>✅ Analysis complete ({mode.upper()} MODE)</h2>
    <ul>
      <li>🔹 <a href="{out_vertices}">Vertices</a></li>
      <li>🔸 <a href="{out_relations}">Relations</a></li>
      <li>🧩 <a href="{out_themes}">Themes</a></li>
    </ul>
    <img src="{bar_path}" width="480"><br>
    <img src="{scatter_path}" width="480">
    """
    return HTMLResponse(html)

# ------------------------------------------------
# 7️⃣ Health check
# ------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ------------------------------------------------
# Run locally
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
