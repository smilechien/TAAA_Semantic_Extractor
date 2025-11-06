# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer  v4.8
# Author: Smile
# ------------------------------------------------------------
#  • GPT / TF-IDF hybrid semantic engine with retry
#  • Multilingual fonts & bilingual labels
#  • Scatter (Top-20 terms) + h-bar (Top-h themes)
#  • Adds *_themed.csv  → theme per document (TAAA)
#  • Red theme boxes on scatter for publication visuals
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import tempfile, io, os, re, base64, chardet, time
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# ------------------------------------------------------------
# 🔧 App setup
# ------------------------------------------------------------
app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Hybrid GPT/TF-IDF multilingual semantic analyzer with TAAA theme assignment",
    version="4.8"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------
# 🩺 Health check
# ------------------------------------------------------------
@app.get("/health")
async def health():
    key_ok = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "gpt_key_available": key_ok}

# ------------------------------------------------------------
# 🧩 Safe CSV reader
# ------------------------------------------------------------
def safe_read_csv(uploaded: UploadFile) -> pd.DataFrame:
    raw = uploaded.file.read(); uploaded.file.seek(0)
    guess = chardet.detect(raw).get("encoding") or "utf-8"
    for enc in [guess, "utf-8-sig", "utf-8", "big5", "cp950", "latin1"]:
        try:
            text = raw.decode(enc, errors="ignore")
            buf = io.BytesIO(text.encode("utf-8-sig"))
            df = pd.read_csv(buf, sep=None, engine="python")
            if not df.empty:
                return df
        except Exception:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "Unable to decode CSV")

# ------------------------------------------------------------
# 🧠 GPT/TF-IDF hybrid keyword extractor (with retry)
# ------------------------------------------------------------
def extract_keywords(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    # --- GPT first ---
    if os.getenv("OPENAI_API_KEY"):
        prompt = (
            "Extract 10 representative academic keywords from the following text "
            "and separate them by commas:\n\n" + text
        )
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2)
                kw = r.choices[0].message.content.strip()
                kw = re.sub(r"[、;；，\s]+", ", ", kw)
                return kw
            except Exception:
                time.sleep(1)
    # --- TF-IDF fallback ---
    try:
        vec = TfidfVectorizer(max_features=10, stop_words="english")
        vec.fit([text])
        return ", ".join(vec.get_feature_names_out())
    except Exception:
        return ""

# ------------------------------------------------------------
# 🔗 Build edges
# ------------------------------------------------------------
def build_edges(df_kw):
    pairs = []
    for _, row in df_kw.iterrows():
        terms = [t.strip() for t in re.split(r"[,，;；\s]+", str(row["keywords"])) if t.strip()]
        terms = list(dict.fromkeys(terms))
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    e = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    return e.groupby(["Source", "Target"], as_index=False)["edge"].sum()

# ------------------------------------------------------------
# 🧩 Louvain cluster
# ------------------------------------------------------------
def louvain_cluster(edges):
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    comms = nx.community.louvain_communities(G, seed=42)
    cmap = {n: i + 1 for i, c in enumerate(comms) for n in c}
    deg = pd.Series(dict(G.degree(weight="edge")), name="count").reset_index()
    deg.columns = ["term", "count"]
    deg["cluster"] = deg["term"].map(cmap)
    return deg, edges

# ------------------------------------------------------------
# 🎨 Enhanced Scatter Plot (Top-20 Terms + Red Theme Labels)
# ------------------------------------------------------------
def plot_scatter(vertices, topic_labels=None, lang="en"):
    """Top-20 term scatter with multilingual fonts & red theme boxes."""
    plt.rcParams["font.family"] = (
        ["Noto Sans TC", "Microsoft JhengHei", "SimHei"]
        if lang.startswith(("zh", "ja", "ko")) else ["Segoe UI", "Arial"]
    )

    plt.figure(figsize=(7, 6))
    x = vertices["count"].rank().values
    y = vertices["edge"].values
    clusters = vertices["cluster"].values

    plt.scatter(x, y, s=vertices["count"]*45, c=clusters,
                cmap="tab10", alpha=0.9, edgecolor="k")

    plt.axvline(x.mean(), color="red", ls="--", lw=1)
    plt.axhline(y.mean(), color="red", ls="--", lw=1)

    # --- term labels ---
    for idx, row in enumerate(vertices.itertuples()):
        plt.text(x[idx]+0.15, y[idx], row.term, fontsize=8, color="black")

    # --- red theme boxes ---
    if topic_labels is not None and not topic_labels.empty:
        centroids = (
            vertices.groupby("cluster")[["count", "edge"]]
            .mean()
            .reset_index()
            .merge(topic_labels, on="cluster", how="left")
        )
        for _, r in centroids.iterrows():
            if pd.notna(r.get("topic_label", "")):
                plt.text(
                    r["count"], r["edge"] + 0.8, r["topic_label"],
                    color="red", fontsize=11, fontweight="bold",
                    ha="center",
                    bbox=dict(facecolor="white", alpha=0.65,
                              edgecolor="red", boxstyle="round,pad=0.35")
                )

    xlabel = "詞彙排名" if lang.startswith(("zh","ja","ko")) else "Term Rank"
    ylabel = "共現頻率" if lang.startswith(("zh","ja","ko")) else "Co-occurrence Frequency"
    title  = "關鍵詞散佈圖（含主題）" if lang.startswith(("zh","ja","ko")) else "Top-20 Terms Scatter Plot (with Themes)"

    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=0.3, ls=":")
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------
# 📊 h-bar Chart
# ------------------------------------------------------------
def plot_hbar(theme_freq, h_index, topic_labels):
    plt.figure(figsize=(7,5))
    merged = pd.merge(theme_freq, topic_labels, on="cluster", how="left").sort_values("freq", ascending=False)
    plt.bar(merged["cluster"].astype(str), merged["freq"], color="skyblue", edgecolor="k")
    plt.axhline(y=h_index, color="red", ls="--", lw=1.2)
    for i, r in enumerate(merged.itertuples()):
        plt.text(i, r.freq+0.5, str(r.topic_label), rotation=90, ha="center", fontsize=8)
    plt.title(f"Top-h Core Themes (h={h_index}) – TAAA")
    plt.xlabel("Theme (Cluster ID)"); plt.ylabel("Assigned Rows Count")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------
# 🧮 TAAA theme assignment
# ------------------------------------------------------------
def assign_themes(df, vertices, topic_labels):
    mapping = dict(vertices[["term","cluster"]].values)
    cluster_label = dict(topic_labels[["cluster","topic_label"]].values)
    results=[]
    for _, row in df.iterrows():
        text=" ".join(map(str,row.values))
        clusters=[mapping[t] for t in mapping if t in text]
        if not clusters:
            theme="Unassigned"
        else:
            mode=pd.Series(clusters).mode().iloc[0]
            theme=cluster_label.get(mode,f"Cluster {mode}")
        results.append(theme)
    df_theme=df.copy()
    df_theme.insert(0,"theme",results)
    return df_theme

# ------------------------------------------------------------
# 🏠 Home
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(open("index.html", encoding="utf-8").read())

# ------------------------------------------------------------
# 📤 Analyze CSV
# ------------------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = safe_read_csv(file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    if df.shape[1]==1:
        df.columns=["abstract"]
        df["keywords"]=df["abstract"].apply(lambda x: extract_keywords(str(x)))
    else:
        df.columns=[c.strip() for c in df.columns]
        df["keywords"]=df.apply(lambda r:", ".join([str(v) for v in r if pd.notna(v)]),axis=1)

    edges=build_edges(df)
    if edges.empty:
        return HTMLResponse("<h3>❌ No co-word pairs found.</h3>")

    vertices,edges=louvain_cluster(edges)
    vertices["edge"]=vertices["term"].map(lambda t: edges.query("Source==@t or Target==@t")["edge"].sum())
    top20=vertices.sort_values("count",ascending=False).head(20)

    # --- h-core metrics ---
    theme_freq=vertices.groupby("cluster")["term"].count().reset_index(name="freq").sort_values("freq",ascending=False)
    theme_freq["rank"]=range(1,len(theme_freq)+1)
    h_index=int(max(theme_freq["rank"][theme_freq["freq"]>=theme_freq["rank"]],default=0))
    topic_labels=vertices.groupby("cluster")["term"].apply(lambda x:", ".join(x.head(2))).reset_index(name="topic_label")

    # --- plots ---
    img_scatter=plot_scatter(top20, topic_labels)
    img_bar=plot_hbar(theme_freq,h_index,topic_labels)

    # --- TAAA assignment ---
    df_theme=assign_themes(df,vertices,topic_labels)

    # --- Save outputs ---
    tmp_v=tempfile.NamedTemporaryFile(delete=False,suffix="_vertices.csv")
    tmp_e=tempfile.NamedTemporaryFile(delete=False,suffix="_relations.csv")
    tmp_t=tempfile.NamedTemporaryFile(delete=False,suffix="_themed.csv")
    vertices.to_csv(tmp_v.name,index=False,encoding="utf-8-sig")
    edges.to_csv(tmp_e.name,index=False,encoding="utf-8-sig")
    df_theme.to_csv(tmp_t.name,index=False,encoding="utf-8-sig")

    html=f"""
    <h2>✅ Analysis Complete</h2>
    <h4>Scatter Plot (Top-20 Terms)</h4>
    <img src="data:image/png;base64,{img_scatter}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <h4>Bar Chart (Top-h Core Themes)</h4>
    <img src="data:image/png;base64,{img_bar}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={tmp_v.name}">📥 Vertices CSV</a><br>
    <a href="/download?path={tmp_e.name}">📥 Relations CSV</a><br>
    <a href="/download?path={tmp_t.name}">📥 Themed CSV (TAAA assignments)</a>
    """
    return HTMLResponse(html)

# ------------------------------------------------------------
# 📥 Download route
# ------------------------------------------------------------
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# ------------------------------------------------------------
# 🚀 Local / Render run
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, locale
    locale.setlocale(locale.LC_ALL, "")
    port=int(os.environ.get("PORT",10000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
