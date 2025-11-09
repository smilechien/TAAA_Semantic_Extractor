# ============================================================
# 🌐 main.py — TAAA Semantic–Co-Word Analyzer v21.1 (GPT + TF-IDF Hybrid)
# Author: Smile
# Description: Semantic extraction using GPT (if API key provided)
#              or TF-IDF fallback (offline). Compatible with Render.
# ============================================================

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd, numpy as np, io, os, openai, networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from community import community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------
# ⚙️ Initialize app and folders
# ------------------------------------------------------------
app = FastAPI()
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# ============================================================
# 🧩 Helper functions
# ============================================================

def safe_read_csv(file: UploadFile) -> pd.DataFrame:
    """Auto-detect UTF-8 or Big5 encoding."""
    data = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8")
    except Exception:
        df = pd.read_csv(io.BytesIO(data), encoding="big5", errors="ignore")
    return df.fillna("")

def compute_aac(values):
    """AAC dominance metric from top 3 values."""
    vals = sorted(values, reverse=True)[:3]
    if len(vals) < 3 or vals[1] == 0 or vals[2] == 0:
        return 0
    v1, v2, v3 = vals
    aac = ((v1 / v2) / (v2 / v3)) / (1 + ((v1 / v2) / (v2 / v3)))
    return round(aac, 2)

# ============================================================
# 🧠 GPT Semantic Extraction
# ============================================================

async def extract_semantics_with_gpt(api_key: str, text: str, n_terms: int = 10):
    """Send text to GPT and return extracted key concepts."""
    openai.api_key = api_key
    prompt = (
        f"Extract the top {n_terms} concise academic keywords or themes "
        f"from the following text:\n\n{text}\n\nReturn a comma-separated list."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response["choices"][0]["message"]["content"]
        return [t.strip() for t in content.split(",") if t.strip()]
    except Exception as e:
        print("⚠️ GPT extraction failed:", e)
        return []

# ============================================================
# 🧮 TF-IDF Fallback Extraction
# ============================================================

def extract_semantics_with_tfidf(texts, n_terms=10):
    """Extract top TF-IDF terms per document."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    results = []
    for i in range(X.shape[0]):
        row = X[i].toarray().flatten()
        top_idx = row.argsort()[-n_terms:][::-1]
        results.append(feature_names[top_idx].tolist())
    return results

# ============================================================
# 🧭 Main Analysis Route
# ============================================================

@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile, api_key: str = Form(None)):
    df = safe_read_csv(file)
    ncols = len(df.columns)
    mode = 1 if ncols == 1 else 2
    file_base = os.path.splitext(file.filename)[0]

    # ========================================================
    # MODE 1 — ABSTRACT MODE (Semantic Extraction)
    # ========================================================
    if mode == 1:
        texts = [str(row.iloc[0]) for _, row in df.iterrows()]
        use_gpt = "smilechien" in file.filename.lower() and api_key

        if use_gpt:
            print("🔑 Using GPT for semantic extraction.")
            terms = [await extract_semantics_with_gpt(api_key, t) for t in texts]
        else:
            print("⚙️ Using local TF-IDF fallback.")
            terms = extract_semantics_with_tfidf(texts)

        # Output CSV
        max_len = max(len(x) for x in terms)
        padded = [x + [""] * (max_len - len(x)) for x in terms]
        df_terms = pd.DataFrame(padded)
        out_path = "static/semantic_terms.csv"
        df_terms.to_csv(out_path, index=False, encoding="utf-8-sig")

        note = (
            "✅ GPT Semantic Extraction completed (using your API key)."
            if use_gpt else "✅ TF-IDF Semantic Extraction completed (offline mode)."
        )
        return HTMLResponse(
            f"<h3>{note}</h3><a href='/{out_path}'>Download Terms CSV</a><br>"
            "<a href='/'>Return Home</a>"
        )

    # ========================================================
    # MODE 2 — CO-WORD NETWORK ANALYSIS
    # ========================================================
    else:
        pairs = []
        for _, row in df.iterrows():
            terms = [str(x).strip() for x in row if str(x).strip()]
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    pairs.append(tuple(sorted([terms[i], terms[j]])))
        edges = pd.DataFrame(pairs, columns=["term1", "term2"])
        edges["connection_count"] = 1
        edges = edges.groupby(["term1", "term2"], as_index=False)["connection_count"].sum()

        # Vertex table
        all_terms = pd.Series(pd.concat([edges["term1"], edges["term2"]]))
        vertices = pd.DataFrame({
            "name": all_terms.value_counts().index,
            "value": all_terms.value_counts().values
        })
        strength = edges.groupby("term1")["connection_count"].sum().add(
            edges.groupby("term2")["connection_count"].sum(), fill_value=0)
        vertices["value2"] = vertices["name"].map(strength).fillna(0)

        # Louvain clustering
        G = nx.from_pandas_edgelist(edges, "term1", "term2", ["connection_count"])
        cluster = community_louvain.best_partition(G, weight="connection_count", random_state=42)
        vertices["cluster"] = vertices["name"].map(cluster)

        leaders = (
            vertices.sort_values(["cluster", "value2"], ascending=[True, False])
            .groupby("cluster").head(1)[["cluster", "name"]]
            .rename(columns={"name": "leader_label"})
        )
        vertices = vertices.merge(leaders, on="cluster", how="left")

        # Theme summary
        theme = vertices.groupby("cluster").agg(
            leader=("leader_label", "first"),
            member_count=("name", "count"),
            members=("name", lambda x: ", ".join(x))
        ).reset_index(drop=True)
        theme.to_csv("static/theme.csv", index=False, encoding="utf-8-sig")

        # AAC + H metrics
        aac = compute_aac(vertices["value"].tolist())
        theme = theme.sort_values("member_count", ascending=False)
        h = max(i + 1 for i, c in enumerate(theme["member_count"]) if c >= i + 1)

        # Plots
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(theme) + 1), theme["member_count"], color="#69b3a2")
        plt.axhline(y=h, color="red", linestyle="--")
        plt.axvline(x=h, color="red", linestyle="--")
        plt.title(f"H-Theme Bar (h={h})")
        plt.tight_layout()
        plt.savefig("static/h_theme_bar.png", dpi=150)
        plt.close()

        fig = px.scatter(vertices, x="value2", y="value", color=vertices["cluster"].astype(str),
                         size="value", hover_name="name",
                         title=f"Theme Scatter — AAC={aac}")
        fig.write_html("static/theme_scatter.html")

        # Outputs
        vertices.to_csv("static/vertices.csv", index=False, encoding="utf-8-sig")
        edges.to_csv("static/relations.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame([{
            "AAC": aac, "H": h,
            "Clusters": len(theme), "Nodes": len(vertices), "Edges": len(edges)
        }])
        summary.to_csv("static/summary.csv", index=False, encoding="utf-8-sig")

        html = """
        <h3>✅ Co-Word Analysis Completed</h3>
        <p><a href='/static/vertices.csv'>vertices.csv</a> |
        <a href='/static/relations.csv'>relations.csv</a> |
        <a href='/static/theme.csv'>theme.csv</a> |
        <a href='/static/summary.csv'>summary.csv</a></p>
        <p><a href='/static/h_theme_bar.png'>📊 H-Theme Bar</a> |
        <a href='/static/theme_scatter.html'>🌐 Theme Scatter</a></p>
        <p><a href='/'>Return Home</a></p>
        """
        return HTMLResponse(html)

# ============================================================
# 🏠 Home Route
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ============================================================
# 🧱 Static Mount
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")
