# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (Stable Render Version)
# Author: Smile
# Description:
#   • Mode 1  → Abstract / DOI column (GPT/DOI extraction)
#   • Mode 2  → Multi-column co-word terms
#   • Louvain clustering (categorical edges only)
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tempfile, io, base64, os, re, requests
from langdetect import detect
from openai import OpenAI

# ------------------------------------------------------------
# 🚀 FastAPI setup
# ------------------------------------------------------------
app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Multilingual categorical Louvain clustering (Mode 1 / Mode 2)",
    version="2.6.0"
)

# ------------------------------------------------------------
# 🔑 GPT client
# ------------------------------------------------------------
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(var, None)
    return OpenAI(api_key=key)

# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = open("index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

# ------------------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    fn = file.filename
    try:
        # auto-detect delimiter
        df = pd.read_csv(file.file, sep=None, engine="python")
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    df = df.dropna(how="all")
    if df.empty:
        return HTMLResponse("<h3>❌ No usable data rows found in your file.</h3>")

    mode = "mode1" if df.shape[1] == 1 else "mode2"

    # --------------------------------------------------------
    # 🧠 Mode 1 – Abstracts / DOI
    # --------------------------------------------------------
    if mode == "mode1":
        all_terms = []
        for _, r in df.iterrows():
            text = str(r.iloc[0]).strip()
            if not text:
                continue
            if re.match(r"^10\.\d{4,9}/", text):
                text = fetch_abstract_from_doi(text)
            lang = detect_language(text)
            terms = extract_terms_gpt(fn, text, lang)
            all_terms.append(split_terms(terms))
        df_terms = pd.DataFrame(all_terms)
        edges = build_edges(df_terms)

    # --------------------------------------------------------
    # 🧩 Mode 2 – Co-word terms
    # --------------------------------------------------------
    else:
        lang = detect_language(" ".join(df.columns))
        edges = build_edges(df)

    # --------------------------------------------------------
    if edges.empty:
        preview_html = df.head().to_html(index=False)
        return HTMLResponse(
            f"<h3>❌ No valid term pairs found.</h3>"
            f"<p>👉 Each row must contain at least two non-empty terms.</p>"
            f"<h4>File preview:</h4>{preview_html}"
        )

    vertices, rels = louvain_cluster(edges)
    if vertices.empty:
        return HTMLResponse("<h3>❌ No cluster detected.</h3>")

    img64 = plot_network(vertices, rels)

    tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix="_vertices.csv")
    tmp_r = tempfile.NamedTemporaryFile(delete=False, suffix="_relations.csv")
    vertices.to_csv(tmp_v.name, index=False, encoding="utf-8-sig")
    rels.to_csv(tmp_r.name, index=False, encoding="utf-8-sig")

    html = f"""
    <h2>✅ Analysis Complete ({mode})</h2>
    <p>Detected language: <b>{lang}</b></p>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={tmp_v.name}">📥 Vertices CSV</a><br>
    <a href="/download?path={tmp_r.name}">📥 Relations CSV</a>
    """
    return HTMLResponse(content=html)

# ------------------------------------------------------------
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# ------------------------------------------------------------
# 🧠 GPT keyword extraction
# ------------------------------------------------------------
def extract_terms_gpt(filename: str, text: str, lang: str):
    # for non-developer users (no "smilechien" in file name)
    if "smilechien" not in filename.lower():
        words = re.findall(r"[A-Za-z\u4e00-\u9fff\-]+", text)
        return ", ".join(sorted(set(words))[:10])
    if not text.strip():
        return ""
    prompt = f"Extract 10 key semantic terms in {lang}, comma-separated:\n{text}"
    try:
        c = get_client()
        r = c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# ------------------------------------------------------------
# 🈚 helpers
# ------------------------------------------------------------
def split_terms(t):
    return [x.strip() for x in re.split(r"[;,、，]", str(t)) if x.strip()]

def detect_language(text):
    try:
        code = detect(text)
    except Exception:
        code = "unknown"
    mapping = {
        "zh": "Chinese", "en": "English", "ja": "Japanese",
        "ko": "Korean", "fr": "French", "es": "Spanish"
    }
    return mapping.get(code[:2], code)

def fetch_abstract_from_doi(doi: str):
    for u in [
        f"https://api.crossref.org/works/{doi}",
        f"https://api.openalex.org/works/doi:{doi}"
    ]:
        try:
            r = requests.get(u, timeout=8)
            if r.status_code == 200:
                j = r.json()
                abs_ = (
                    j.get("message", {}).get("abstract")
                    or j.get("abstract", {}).get("value")
                )
                if abs_:
                    return re.sub(r"<[^>]+>", "", abs_)
        except Exception:
            pass
    return ""

# ------------------------------------------------------------
# 🧩 Edge builder (robust)
# ------------------------------------------------------------
def build_edges(df):
    if df.empty:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    pairs = []
    for _, row in df.iterrows():
        # convert all to strings and remove empties/dupes
        terms = [str(t).strip() for t in row if str(t).strip() not in ["", "nan", "None"]]
        terms = list(dict.fromkeys(terms))
        if len(terms) < 2:
            continue
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    e = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    e = e.groupby(["Source", "Target"], as_index=False)["edge"].sum()
    return e

# ------------------------------------------------------------
# 🧮 Louvain clustering (categorical)
# ------------------------------------------------------------
def louvain_cluster(edges):
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    comms = nx.community.louvain_communities(G, seed=42)
    cmap = {n: i + 1 for i, c in enumerate(comms) for n in c}
    deg = pd.Series(dict(G.degree(weight="edge")), name="count").reset_index()
    deg.columns = ["term", "count"]
    deg["cluster"] = deg["term"].map(cmap)
    top20 = deg.sort_values("count", ascending=False).head(20)
    rels = edges.query("Source in @top20.term and Target in @top20.term")
    return top20, rels

# ------------------------------------------------------------
# 🖼️ Plot network
# ------------------------------------------------------------
def plot_network(vertices, rels):
    plt.figure(figsize=(7, 6))
    G = nx.from_pandas_edgelist(rels, "Source", "Target", "edge")
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[vertices.set_index("term").loc[n, "count"] * 50 for n in G.nodes()],
        node_color=[vertices.set_index("term").loc[n, "cluster"] for n in G.nodes()],
        cmap="tab10", alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Top-20 Louvain Network (Categorical Relations)", fontsize=12)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
