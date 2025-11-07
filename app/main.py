# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer  v15.9
# ============================================================
# Robust, Render-ready build with:
#   • UTF-8 / Big5 CSV resilience
#   • DOI semantic extraction (GPT / OpenAlex)
#   • Cluster detection + TAAA theme assignment
#   • Static download outputs
#   • HTML served from /templates/index.html
# ============================================================

import os, io, time, json, requests, pandas as pd, networkx as nx, plotly.express as px
from collections import Counter
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import matplotlib.pyplot as plt

# ============================================================
# 0️⃣ App setup
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="TAAA Semantic–Co-Word Analyzer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ============================================================
# 1️⃣ Safe CSV loader (UTF-8 / Big5 fallback)
# ============================================================
def load_csv(uploaded_file: UploadFile):
    raw = uploaded_file.file.read()
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Unable to decode CSV with UTF-8/Big5 encodings")

# ============================================================
# 2️⃣ Dual semantic extractor (GPT Plus → OpenAlex fallback)
# ============================================================
def extract_semantic_phrases(dois):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_gpt = bool(api_key)
    results = []

    if use_gpt:
        print("🔹 Using ChatGPT Plus semantic extraction")
        client = OpenAI(api_key=api_key)
        for doi in dois:
            try:
                prompt = f"Extract 10 concise scientific keywords or phrases from the article with DOI {doi}. Return them separated by semicolons."
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                kw = resp.choices[0].message.content.strip()
                results.append({"doi": doi, "source": "GPT", "keywords": kw})
            except Exception as e:
                results.append({"doi": doi, "source": "GPT_error", "keywords": str(e)})
            time.sleep(1)
    else:
        print("🔹 Using OpenAlex semantic extraction")
        for doi in dois:
            try:
                encoded = requests.utils.quote(doi, safe="")
                url = f"https://api.openalex.org/works/https://doi.org/{encoded}"
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    results.append({"doi": doi, "source": "OpenAlex_error", "keywords": None})
                    continue
                d = r.json()
                kw = []
                if "concepts" in d:
                    kw += [c["display_name"] for c in sorted(
                        d["concepts"], key=lambda x: x.get("score", 0), reverse=True)[:10]]
                if "abstract_inverted_index" in d:
                    flat = d["abstract_inverted_index"]
                    recon = [""] * (max(sum(flat.values(), [])) + 1)
                    for w, pos in flat.items():
                        for p in pos:
                            recon[p] = w
                    text = " ".join(recon)
                    toks = [t for t in text.lower().split() if len(t) > 3]
                    kw += [w for w, _ in Counter(toks).most_common(10)]
                kw = "; ".join(sorted(set(kw))) if kw else "(no keywords)"
                results.append({"doi": doi, "source": "OpenAlex", "keywords": kw})
            except Exception as e:
                results.append({"doi": doi, "source": "OpenAlex_error", "keywords": str(e)})
            time.sleep(1)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(STATIC_DIR, "doi_semantic_keywords.csv"),
              index=False, encoding="utf-8-sig")
    return df

# ============================================================
# 3️⃣ Main analysis
# ============================================================
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = load_csv(file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    mode = "abstract" if df.shape[1] == 1 else "coword"
    print(f"🔍 Detected mode: {mode}")

    # --- DOI detection ---
    if mode == "abstract" and df.iloc[:, 0].astype(str).str.contains("10\\.").any():
        dois = df.iloc[:, 0].dropna().unique().tolist()
        extract_semantic_phrases(dois)

    # --- Build relation pairs ---
    rels = []
    if mode == "coword":
        for _, row in df.iterrows():
            terms = [str(x).strip() for x in row if pd.notna(x) and str(x).strip()]
            for i, a in enumerate(terms):
                for b in terms[i + 1 :]:
                    rels.append(tuple(sorted([a, b])))
    else:
        for _, row in df.iterrows():
            words = str(row.iloc[0]).split()
            for i, a in enumerate(words):
                for b in words[i + 1 :]:
                    rels.append(tuple(sorted([a, b])))

    if not rels:
        return HTMLResponse("<h3>❌ No valid relations found.</h3>")

    rel = (pd.DataFrame(rels, columns=["source", "target"])
           .value_counts().reset_index(name="weight"))
    rel.to_csv(os.path.join(STATIC_DIR, "relations.csv"),
               index=False, encoding="utf-8-sig")

    # --- Vertices ---
    vertices = pd.DataFrame(pd.concat([rel["source"], rel["target"]]).unique(),
                            columns=["name"])
    freq = pd.concat([rel["source"], rel["target"]]).value_counts().reset_index()
    freq.columns = ["name", "value2"]
    vertices = vertices.merge(freq, on="name", how="left")
    vertices["value2"] = pd.to_numeric(vertices["value2"], errors="coerce").fillna(1)
    vertices["value"] = vertices["value2"]
    vertices.to_csv(os.path.join(STATIC_DIR, "vertices.csv"),
                    index=False, encoding="utf-8-sig")

    # --- Cluster detection (Louvain) ---
    G = nx.from_pandas_edgelist(rel, "source", "target", "weight")
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
    except Exception:
        comms = [list(G.nodes())]
    cluster_map = {n: i + 1 for i, c in enumerate(comms) for n in c}
    vertices["cluster"] = vertices["name"].map(cluster_map).fillna(0).astype(int)

    # --- Theme summary ---
    theme_summary = (
        vertices.groupby("cluster", as_index=False)
        .apply(lambda g: pd.Series({
            "cluster_label": g.sort_values("value2", ascending=False)["name"].iloc[0],
            "member_count": len(g),
            "semantic_label": g.sort_values("value2", ascending=False)["name"].iloc[0],
            "members": ", ".join(g["name"].tolist())
        }))
    )
    theme_summary.to_csv(os.path.join(STATIC_DIR, "theme.csv"),
                         index=False, encoding="utf-8-sig")

    # --- Article–Theme assignment ---
    article_records = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        terms = [v for v in row if str(v).strip()]
        clusters = vertices.loc[vertices["name"].isin(terms), "cluster"].tolist()
        theme_num = min(pd.Series(clusters).mode().values) if clusters else -1
        article_records.append({"article_id": i, "theme": theme_num})
    art_df = pd.DataFrame(article_records)
    theme_map = dict(zip(theme_summary["cluster"], theme_summary["cluster_label"]))
    art_df["theme_label"] = art_df["theme"].map(theme_map)
    art_df.to_csv(os.path.join(STATIC_DIR, "article_theme_assign.csv"),
                  index=False, encoding="utf-8-sig")

    # --- Visuals ---
    top20 = vertices.sort_values("value2", ascending=False).head(20)
    fig_scatter = px.scatter(
        top20,
        x="value2",
        y=top20.index + 1,
        text="name",
        color=top20["cluster"].astype(str),
        size="value2",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=f"Theme Scatter (Top 20 Terms, H = {len(theme_summary)})",
        labels={"x": "Frequency", "y": "Rank", "color": "Cluster"},
    )
    fig_scatter.update_traces(textposition="top center", textfont=dict(color="black"))
    fig_scatter.write_html(os.path.join(STATIC_DIR, "theme_scatter.html"),
                           include_plotlyjs="cdn")

    plt.figure(figsize=(6, 4))
    theme_summary.plot.barh(x="cluster_label", y="member_count", legend=False)
    plt.title(f"Top H-Theme Distribution (H = {len(theme_summary)})")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "h_theme_bar.png"), dpi=150)
    plt.close()

    # --- HTML response ---
    return HTMLResponse(f"""
    <h2>✅ Analysis Complete ({mode} mode)</h2>
    <p>Detected {len(vertices)} vertices and {len(rel)} relations.</p>
    <ul>
      <li><a href='/static/vertices.csv'>Vertices (CSV)</a></li>
      <li><a href='/static/relations.csv'>Relations (CSV)</a></li>
      <li><a href='/static/theme.csv'>Theme Summary (CSV)</a></li>
      <li><a href='/static/article_theme_assign.csv'>Article–Theme Assignment (CSV)</a></li>
      <li><a href='/static/h_theme_bar.png'>📊 H-Theme Bar (PNG)</a></li>
      <li><a href='/static/theme_scatter.html'>🎨 Theme Scatter (Interactive)</a></li>
      <li><a href='/static/doi_semantic_keywords.csv'>🔗 DOI Semantic Keywords (CSV)</a></li>
    </ul>
    """)

# ============================================================
# 4️⃣ Home page served from /templates/
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>⚠️ index.html not found under /templates.</h3>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

# Generic safe file route
@app.get("/{filename}")
async def serve_static(filename: str):
    path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse('{"detail":"Not Found"}', status_code=404)
