# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (Render-Safe, Numeric-Safe)
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import io, base64, os, re, requests
from langdetect import detect
from openai import OpenAI

app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Auto-detect mode from uploaded CSV; multilingual semantic and co-word analysis.",
    version="2.4.0"
)

# ------------------------------------------------
# 🧠 Safe GPT client
# ------------------------------------------------
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(var, None)
    return OpenAI(api_key=key)

# ------------------------------------------------
# 🏠 Home
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = open("index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

# ------------------------------------------------
# 📤 Upload + Analyze
# ------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    filename = file.filename
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ Unable to read CSV: {e}</h3>")

    mode = "mode1" if df.shape[1] == 1 else "mode2"

    # ---------- Mode 1 ----------
    if mode == "mode1":
        all_terms = []
        for _, row in df.iterrows():
            text = str(row.iloc[0]).strip()
            if re.match(r"^10\.\d{4,9}/", text):
                text = fetch_abstract_from_doi(text)
            lang = detect_language(text)
            terms = extract_terms_gpt(filename, text, lang)
            term_list = [t.strip() for t in re.split(r"[;,]", terms) if t.strip()]
            all_terms.append(term_list)
        df_terms = pd.DataFrame(all_terms)
        edges = build_edges(df_terms)
        lang_display = lang
    # ---------- Mode 2 ----------
    else:
        lang_display = detect_language(" ".join(df.columns))
        edges = build_edges(df)

    if edges.empty:
        return HTMLResponse("<h3>❌ No valid term pairs found in your CSV.</h3>")

    # ensure numeric edges
    edges["edge"] = pd.to_numeric(edges["edge"], errors="coerce").fillna(1).astype(float)

    top20, rels = louvain_top20(edges)
    if top20.empty:
        return HTMLResponse("<h3>❌ Unable to build network: no valid nodes or edges.</h3>")

    img64 = plot_network(top20, rels)

    tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix="_vertices.csv")
    tmp_r = tempfile.NamedTemporaryFile(delete=False, suffix="_relations.csv")
    top20.to_csv(tmp_v.name, index=False, encoding="utf-8-sig")
    rels.to_csv(tmp_r.name, index=False, encoding="utf-8-sig")

    html = f"""
    <h2>✅ Analysis complete ({mode})</h2>
    <p>Detected language: <b>{lang_display}</b></p>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={tmp_v.name}">📥 Download vertices CSV</a><br>
    <a href="/download?path={tmp_r.name}">📥 Download relations CSV</a>
    """
    return HTMLResponse(content=html)

# ------------------------------------------------
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# ------------------------------------------------
# GPT extraction
# ------------------------------------------------
def extract_terms_gpt(filename: str, text: str, lang: str):
    if "smilechien" not in filename.lower():
        words = re.findall(r"[A-Za-z\u4e00-\u9fff\-]+", text)
        return ", ".join(sorted(set(words))[:10])
    if not text.strip():
        return ""
    prompt = f"Extract 10 key semantic terms in {lang}, comma-separated, from:\n{text}"
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ------------------------------------------------
def detect_language(text: str):
    try:
        code = detect(text)
    except Exception:
        code = "unknown"
    return {
        "zh-cn": "Chinese", "zh-tw": "Chinese", "en": "English",
        "ja": "Japanese", "ko": "Korean", "fr": "French", "es": "Spanish"
    }.get(code.lower(), code)

# ------------------------------------------------
def fetch_abstract_from_doi(doi: str):
    for u in [f"https://api.crossref.org/works/{doi}",
              f"https://api.openalex.org/works/doi:{doi}"]:
        try:
            r = requests.get(u, timeout=8)
            if r.status_code == 200:
                j = r.json()
                abs_ = j.get("message", {}).get("abstract") or j.get("abstract", {}).get("value")
                if abs_:
                    return re.sub(r"<[^>]+>", "", abs_)
        except Exception:
            pass
    return ""

# ------------------------------------------------
def build_edges(df: pd.DataFrame):
    pairs = []
    for _, row in df.iterrows():
        terms = [t.strip() for t in row if isinstance(t, str) and t.strip()]
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    edges = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    edges["edge"] = edges.groupby(["Source", "Target"])["edge"].transform("sum")
    return edges.drop_duplicates()

# ------------------------------------------------
def louvain_top20(edges: pd.DataFrame):
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    clusters = nx.community.louvain_communities(G, seed=42)
    cmap = {n: i + 1 for i, c in enumerate(clusters) for n in c}
    deg = pd.Series(dict(G.degree(weight="edge")), name="count").reset_index()
    deg.columns = ["term", "count"]
    deg["cluster"] = deg["term"].map(cmap)
    top20 = deg.sort_values("count", ascending=False).head(20)
    rels = edges.query("Source in @top20.term and Target in @top20.term")
    return top20, rels

# ------------------------------------------------
def plot_network(top20, rels):
    if top20.empty or rels.empty:
        return ""
    plt.figure(figsize=(7, 6))
    G = nx.from_pandas_edgelist(rels, "Source", "Target", "edge")
    pos = nx.spring_layout(G, seed=42)
    mean_e, mean_c = rels["edge"].mean(), top20["count"].mean()

    nx.draw_networkx_edges(G, pos, alpha=0.3,
                           width=[e["edge"] for _, _, e in G.edges(data=True)])
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[top20.set_index("term").loc[n, "count"] * 50 for n in G.nodes()],
        node_color=[top20.set_index("term").loc[n, "cluster"] for n in G.nodes()],
        cmap="tab10", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axvline(mean_e, color="gray", ls="--", lw=1)
    plt.axhline(mean_c, color="gray", ls="--", lw=1)
    plt.text(mean_e * 1.05, plt.ylim()[1]*0.9, f"μ(edge)={mean_e:.2f}", color="gray", fontsize=8)
    plt.text(plt.xlim()[1]*0.8, mean_c * 1.05, f"μ(count)={mean_c:.2f}", color="gray", fontsize=8)
    plt.title("Top-20 Network (Louvain)", fontsize=12)
    plt.xlabel("Edge count")
    plt.ylabel("Node count")
    plt.grid(True, linestyle=":", alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
