# ============================================================
# 🌐 main.py — TAAA Semantic–Co-Word Analyzer (Render-Ready v21.0)
# Author: Smile
# Description: Hybrid GPT/TF-IDF semantic extraction + co-word clustering
# ============================================================

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd, numpy as np, io, os, openai, networkx as nx, matplotlib.pyplot as plt, plotly.express as px
from community import community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()
os.makedirs("static", exist_ok=True)

# ============================================================
# 🧩 Helper Functions
# ============================================================

def safe_read_csv(file: UploadFile) -> pd.DataFrame:
    data = file.file.read()
    for enc in ["utf-8", "big5", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc).fillna("")
        except Exception:
            continue
    raise ValueError("CSV encoding not recognized")

def compute_aac(values):
    vals = sorted(values, reverse=True)[:3]
    if len(vals) < 3 or vals[1] == 0 or vals[2] == 0:
        return 0
    v1, v2, v3 = vals
    return round(((v1 / v2) / (v2 / v3)) / (1 + ((v1 / v2) / (v2 / v3))), 2)

def extract_terms_tfidf(texts, topn=10):
    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    return [terms[np.argsort(-vec.toarray()).ravel()[:topn]].tolist() for vec in tfidf]

async def extract_terms_gpt(texts, api_key, topn=10):
    openai.api_key = api_key
    results = []
    for text in texts:
        prompt = f"Extract the top {topn} meaningful keywords (2–3 words each) from the text:\n{text}"
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            keywords = [w.strip() for w in response.choices[0].message.content.split(",")[:topn]]
            results.append(keywords)
        except Exception:
            results.append([])
    return results

# ============================================================
# 🧭 Analysis Route
# ============================================================

@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile, api_key: str = Form(None)):
    df = safe_read_csv(file)
    ncols = len(df.columns)
    mode = 1 if ncols == 1 else 2
    fname = os.path.splitext(file.filename)[0]

    # =======================================================
    # 🧠 Mode 1: Semantic Extraction (GPT or TF-IDF)
    # =======================================================
    if mode == 1:
        texts = df.iloc[:, 0].astype(str).tolist()
        if api_key and "sk-" in api_key:
            terms = await extract_terms_gpt(texts, api_key)
            method = "GPT Semantic Extraction"
        else:
            terms = extract_terms_tfidf(texts)
            method = "TF-IDF Fallback"

        max_len = max(len(t) for t in terms)
        padded = [t + [""] * (max_len - len(t)) for t in terms]
        df_terms = pd.DataFrame(padded)
        df_terms.to_csv("static/semantic_terms.csv", index=False, encoding="utf-8-sig")
        return HTMLResponse(
            f"<h3>🧠 {method} Completed</h3>"
            f"<p><a href='/static/semantic_terms.csv'>Download terms.csv</a></p>"
            f"<a href='/'>Return Home</a>"
        )

    # =======================================================
    # 🧩 Mode 2: Co-Word Network Analysis
    # =======================================================
    pairs = []
    for _, row in df.iterrows():
        terms = [str(x).strip() for x in row if str(x).strip()]
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append(tuple(sorted([terms[i], terms[j]])))

    edges = pd.DataFrame(pairs, columns=["term1", "term2"])
    edges["weight"] = 1
    edges = edges.groupby(["term1", "term2"], as_index=False)["weight"].sum()

    all_terms = pd.Series(pd.concat([edges["term1"], edges["term2"]]))
    vertices = pd.DataFrame({"name": all_terms.value_counts().index, "value": all_terms.value_counts().values})
    strength = edges.groupby("term1")["weight"].sum().add(edges.groupby("term2")["weight"].sum(), fill_value=0)
    vertices["value2"] = vertices["name"].map(strength).fillna(0)

    G = nx.from_pandas_edgelist(edges, "term1", "term2", ["weight"])
    cluster = community_louvain.best_partition(G, weight="weight", random_state=42)
    vertices["cluster"] = vertices["name"].map(cluster)

    leaders = vertices.sort_values(["cluster", "value2"], ascending=[True, False])\
                      .groupby("cluster").head(1)[["cluster", "name"]]\
                      .rename(columns={"name": "leader"})
    vertices = vertices.merge(leaders, on="cluster", how="left")

    theme = vertices.groupby("cluster").agg(
        leader=("leader", "first"),
        member_count=("name", "count"),
        members=("name", lambda x: ", ".join(x))
    ).reset_index(drop=True)
    theme.to_csv("static/theme.csv", index=False, encoding="utf-8-sig")

    aac = compute_aac(vertices["value"].tolist())
    mean_x, mean_y = vertices["value2"].mean(), vertices["value"].mean()
    fig = px.scatter(vertices, x="value2", y="value", color=vertices["cluster"].astype(str),
                     size="value", hover_name="name", title=f"Theme Scatter — AAC={aac}")
    fig.add_vline(x=mean_x, line_dash="dot", line_color="red")
    fig.add_hline(y=mean_y, line_dash="dot", line_color="red")
    fig.write_html("static/theme_scatter.html")

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(theme) + 1), theme["member_count"], color="#69b3a2")
    h = max(i + 1 for i, c in enumerate(theme["member_count"]) if c >= i + 1)
    plt.axhline(y=h, color="red", linestyle="--")
    plt.title(f"H-Theme Bar (h={h})")
    plt.savefig("static/h_theme_bar.png", dpi=150)
    plt.close()

    vertices.to_csv("static/vertices.csv", index=False, encoding="utf-8-sig")
    edges.to_csv("static/relations.csv", index=False, encoding="utf-8-sig")

    return HTMLResponse(f"""
        <h3>✅ Co-Word Analysis Completed</h3>
        <p><a href='/static/vertices.csv'>vertices.csv</a> |
           <a href='/static/relations.csv'>relations.csv</a> |
           <a href='/static/theme.csv'>theme.csv</a></p>
        <p><a href='/static/h_theme_bar.png'>📊 H-Theme Bar</a> |
           <a href='/static/theme_scatter.html'>🌐 Theme Scatter</a></p>
        <a href='/'>Return Home</a>
    """)

# ============================================================
# 🏠 Home Route
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

app.mount("/static", StaticFiles(directory="static"), name="static")
