# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (Louvain + Top20 Logic)
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io, base64, os, re, json, requests
from langdetect import detect
from openai import OpenAI

app = FastAPI()
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ============================================================
# 🔐 GPT Client Setup
# ============================================================
def get_gpt_client(filename: str):
    """Use system key only if 'smilechien' in filename."""
    if "smilechien" in filename.lower():
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Server OPENAI_API_KEY not set")
        return OpenAI(api_key=key)
    else:
        return None  # skip GPT for non-developer users

# ============================================================
# 🌏 Language Detection Helper
# ============================================================
def detect_language(text):
    try:
        code = detect(text)
    except Exception:
        code = "unknown"
    lang_map = {
        "zh-cn": "Chinese", "zh-tw": "Chinese", "en": "English",
        "ja": "Japanese", "ko": "Korean", "fr": "French", "es": "Spanish"
    }
    return lang_map.get(code, code)

# ============================================================
# 🔎 DOI to Abstract via CrossRef/OpenAlex
# ============================================================
def fetch_abstract_from_doi(doi):
    try:
        url = f"https://api.crossref.org/works/{doi}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            js = r.json()
            abs_ = js["message"].get("abstract", "")
            if abs_:
                return re.sub(r"<[^>]+>", "", abs_)
        url = f"https://api.openalex.org/works/doi:{doi}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            js = r.json()
            return js.get("abstract", {}).get("value", "")
    except Exception:
        pass
    return ""

# ============================================================
# 🧠 GPT Term Extraction
# ============================================================
def extract_terms_gpt(client, text, lang):
    if not client:
        words = re.findall(r"[\w\-]+", text)
        return ", ".join(sorted(set(words))[:10])
    prompt = f"Extract 10 key semantic terms (no duplicates) from this {lang} abstract. Output comma-separated terms only.\n\n{text}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a multilingual bibliometric assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        out = resp.choices[0].message.content.strip()
        return out
    except Exception as e:
        return f"error_{e}"

# ============================================================
# 🧩 Build Co-word Edges
# ============================================================
def build_coword_edges(df_terms):
    pairs = []
    for _, row in df_terms.iterrows():
        terms = [t.strip() for t in row if isinstance(t, str) and t.strip()]
        if len(terms) < 2:
            continue
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j]))
    edges = pd.DataFrame(pairs, columns=["Source", "Target"])
    edges = edges.groupby(["Source", "Target"]).size().reset_index(name="edge")
    return edges

# ============================================================
# 🧮 Louvain Clustering + Top20 Selection
# ============================================================
def cluster_and_top20(edges):
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    clusters = nx.community.louvain_communities(G, seed=42)
    cluster_map = {}
    for i, c in enumerate(clusters, 1):
        for n in c:
            cluster_map[n] = i

    counts = edges.groupby("Source")["edge"].sum().add(
        edges.groupby("Target")["edge"].sum(), fill_value=0
    ).reset_index()
    counts.columns = ["term", "count"]
    counts["cluster"] = counts["term"].map(cluster_map)

    # Apply frequency-based Top20 logic (R-style)
    freq = counts["cluster"].value_counts()
    selected = []
    remaining = 20

    def pick(freq_cond, per_cluster, cap):
        nonlocal remaining
        picks = []
        for cl, f in freq[freq_cond].items():
            subset = counts[counts["cluster"] == cl].nlargest(per_cluster, "count")
            picks.append(subset)
            if sum(len(df) for df in picks) >= cap:
                break
        return pd.concat(picks) if picks else pd.DataFrame(columns=counts.columns)

    lvl1 = pick(freq >= 4, 4, 20)
    selected.append(lvl1)
    remaining = 20 - len(lvl1)
    if remaining > 0:
        lvl2 = pick(freq == 3, 3, 22)
        selected.append(lvl2)
    if remaining > 0:
        lvl3 = pick(freq == 2, 2, 21)
        selected.append(lvl3)
    if remaining > 0:
        lvl4 = pick(freq == 1, 1, 20)
        selected.append(lvl4)

    top20 = pd.concat(selected).drop_duplicates("term").head(20)
    rels = edges.query("Source in @top20.term and Target in @top20.term")
    return top20, rels

# ============================================================
# 📊 Enhanced Network Plot with Means and Axes Labels
# ============================================================
def plot_network(top20, rels):
    plt.figure(figsize=(7, 6))
    G = nx.from_pandas_edgelist(rels, "Source", "Target", "edge")
    pos = nx.spring_layout(G, seed=42)

    mean_count = top20["count"].mean()
    mean_edge = rels["edge"].mean()

    # Scatter-like axes layout
    plt.axhline(mean_count, color="gray", linestyle="--", linewidth=1)
    plt.axvline(mean_edge, color="gray", linestyle="--", linewidth=1)

    plt.text(mean_edge * 1.02, plt.ylim()[1] * 0.9, f"μ(edge)={mean_edge:.2f}", color="gray", fontsize=8)
    plt.text(plt.xlim()[1] * 0.8, mean_count * 1.02, f"μ(count)={mean_count:.2f}", color="gray", fontsize=8)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=rels["edge"])
    nx.draw_networkx_nodes(
        G, pos,
        node_size=top20.set_index("term").loc[list(G.nodes), "count"] * 50,
        node_color=top20.set_index("term").loc[list(G.nodes), "cluster"],
        cmap="tab10", alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.xlabel("Edge count", fontsize=10)
    plt.ylabel("Node count", fontsize=10)
    plt.title("Top 20 Network (Louvain Clusters)", fontsize=12)
    plt.axis("on")
    plt.grid(True, linestyle=":", alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ============================================================
# 📤 Main Analysis Endpoint
# ============================================================
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(mode: str = Form(...), file: UploadFile = File(...)):
    filename = file.filename
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ Could not read file: {e}</h3>")

    # === MODE 1: Abstracts + GPT ===
    if mode == "mode1":
        if "abstract" not in df.columns:
            return HTMLResponse("<h3>❌ CSV must contain 'abstract' column.</h3>")
        client = get_gpt_client(filename)
        all_terms = []
        for _, row in df.iterrows():
            text = str(row["abstract"]).strip()
            if re.match(r"^10\.\d{4,9}/", text):
                abs_text = fetch_abstract_from_doi(text)
            else:
                abs_text = text
            lang = detect_language(abs_text)
            terms = extract_terms_gpt(client, abs_text, lang)
            term_list = [t.strip() for t in re.split(r"[;,]", terms) if t.strip()]
            all_terms.append(term_list)
        df_terms = pd.DataFrame(all_terms)
        edges = build_coword_edges(df_terms)

    # === MODE 2: Co-word CSV ===
    elif mode == "mode2":
        lang = detect_language(" ".join(df.columns))
        edges = build_coword_edges(df)
    else:
        return HTMLResponse("<h3>❌ Invalid mode selected.</h3>")

    # === Cluster and Plot ===
    top20, rels = cluster_and_top20(edges)
    img64 = plot_network(top20, rels)

    # Save CSVs in memory
    vbuf = io.StringIO(); rbuf = io.StringIO()
    top20.to_csv(vbuf, index=False)
    rels.to_csv(rbuf, index=False)
    app.state.vertices = io.BytesIO(vbuf.getvalue().encode("utf-8"))
    app.state.relations = io.BytesIO(rbuf.getvalue().encode("utf-8"))

    html = f"""
    <h2>✅ Analysis Complete ({mode})</h2>
    <p>Detected Language: <b>{lang}</b></p>
    <p>Top 20 nodes by R-style selection logic (Louvain clusters).</p>
    <img src="data:image/png;base64,{img64}" alt="Network Plot"
         style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download_vertices" target="_blank">📥 Download Vertices CSV</a><br>
    <a href="/download_relations" target="_blank">📥 Download Relations CSV</a>
    """
    return HTMLResponse(html)

# ============================================================
# 📥 Download Endpoints
# ============================================================
@app.get("/download_vertices")
async def download_vertices():
    buf = getattr(app.state, "vertices", None)
    if not buf: return HTMLResponse("<h3>No vertex data available.</h3>")
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vertices.csv"})

@app.get("/download_relations")
async def download_relations():
    buf = getattr(app.state, "relations", None)
    if not buf: return HTMLResponse("<h3>No relation data available.</h3>")
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=relations.csv"})
