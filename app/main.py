# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v2.9, Robust Chinese CSV Reader)
# Author: Smile
# Description:
#   • Mode 1 → Abstract / DOI column (GPT/DOI extraction)
#   • Mode 2 → Multi-column co-word terms
#   • Full encoding detection (UTF-8-SIG / Big5 / CP950)
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tempfile, io, base64, os, re, requests, chardet
from langdetect import detect
from openai import OpenAI

app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Multilingual Louvain clustering with robust CSV decoding",
    version="2.9.0"
)

# ------------------------- GPT setup ----------------------------
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OPENAI_API_KEY not set in environment")
    # Remove proxy interference (Render safe)
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(var, None)
    return OpenAI(api_key=key)


# ------------------------- Safe CSV Reader ----------------------
def safe_read_csv(uploaded: UploadFile) -> pd.DataFrame:
    """Safely read CSV with auto-detection + Chinese encoding fallback."""
    raw = uploaded.file.read()              # read bytes once
    uploaded.file.seek(0)                   # reset pointer
    guess = chardet.detect(raw).get("encoding") or "utf-8"

    # Prioritize common East-Asian encodings
    encodings = [guess, "utf-8-sig", "utf-8", "big5", "cp950", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
            if not df.empty:
                print(f"✅ CSV decoded successfully using: {enc}")
                return df
        except Exception as e:
            print(f"⚠️ Failed decoding with {enc}: {e}")
            continue

    raise UnicodeDecodeError(
        "utf-8", raw, 0, 1, "All encoding attempts failed. Try saving file as UTF-8-SIG."
    )


# ------------------------- Helpers -------------------------------
def is_text_column(series: pd.Series) -> bool:
    """Return True if column likely contains text."""
    sample = series.dropna().astype(str).head(10)
    if sample.empty:
        return False
    numeric_ratio = sum(s.replace('.', '', 1).isdigit() for s in sample) / len(sample)
    return numeric_ratio < 0.5


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


def split_terms(t):
    return [x.strip() for x in re.split(r"[;,、，\t ]+", str(t)) if x.strip()]


def fetch_abstract_from_doi(doi):
    """Try to fetch abstract using CrossRef or OpenAlex."""
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
            continue
    return ""


# ------------------------- GPT Extractor -------------------------
def extract_terms_gpt(filename, text, lang):
    """Extract 10 key semantic terms (fallback = regex if no GPT trigger)."""
    if not text.strip():
        return ""
    # Only use GPT when file name contains 'smilechien' to save tokens
    if "smilechien" not in filename.lower():
        words = re.findall(r"[A-Za-z\u4e00-\u9fff\-]+", text)
        return ", ".join(sorted(set(words))[:10])

    prompt = f"Extract 10 key semantic terms in {lang}, comma-separated:\n{text[:1500]}"
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


# ------------------------- Edge Builder --------------------------
def build_edges(df):
    """Construct Source–Target pairs for all co-occurring terms."""
    pairs = []
    for _, row in df.iterrows():
        terms = [str(t).strip() for t in row if str(t).strip() not in ["", "nan", "None"]]
        terms = list(dict.fromkeys(terms))  # deduplicate
        if len(terms) < 2:
            continue
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    e = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    return e.groupby(["Source", "Target"], as_index=False)["edge"].sum()


# ------------------------- Louvain Cluster -----------------------
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


# ------------------------- Plot Network --------------------------
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


# ------------------------- Routes --------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(open("index.html", "r", encoding="utf-8").read())


@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    fn = file.filename
    try:
        df = safe_read_csv(file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    df = df.dropna(how="all")
    if df.empty:
        return HTMLResponse("<h3>❌ Your file has no usable rows.</h3>")

    # keep only mostly text columns
    text_cols = [c for c in df.columns if is_text_column(df[c])]
    df = df[text_cols]
    if df.empty:
        return HTMLResponse("<h3>❌ No text-like columns detected (numeric-only file).</h3>")

    mode = "mode1" if df.shape[1] == 1 else "mode2"

    # ---------- Mode 1 ----------
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

    # ---------- Mode 2 ----------
    else:
        lang = detect_language(" ".join(df.columns))
        edges = build_edges(df)

    if edges.empty:
        preview_html = df.head().to_html(index=False)
        return HTMLResponse(
            f"<h3>❌ No valid term pairs found.</h3>"
            f"<p>👉 Ensure each row has ≥2 non-empty text terms.</p>"
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
    <p>Kept text columns: <b>{', '.join(text_cols)}</b></p>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={tmp_v.name}">📥 Vertices CSV</a><br>
    <a href="/download?path={tmp_r.name}">📥 Relations CSV</a>
    """
    return HTMLResponse(content=html)


@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))


# ------------------------- Run locally ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
