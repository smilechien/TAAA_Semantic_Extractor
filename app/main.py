# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v3.0)
# Author: Smile
# Description:
#   • Reads Chinese/English CSVs (Big5 / CP950 / UTF-8-SIG)
#   • GPT-4o-mini keyword extraction
#   • Louvain clustering with mean-reference scatter plot
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import tempfile, io, os, re, chardet
from openai import OpenAI

# ------------------------------------------------
# 🔧 Initialize
# ------------------------------------------------
app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Multilingual keyword extraction + network clustering",
    version="3.0.0"
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ------------------------------------------------
# 🧩 Safe CSV Reader (Chinese compatible)
# ------------------------------------------------
def safe_read_csv(uploaded: UploadFile) -> pd.DataFrame:
    """Detect and read CSV with multiple encoding fallbacks."""
    raw = uploaded.file.read()
    uploaded.file.seek(0)
    guess = chardet.detect(raw).get("encoding") or "utf-8"
    encodings = [guess, "utf-8-sig", "utf-8", "big5", "cp950", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
            if not df.empty:
                print(f"✅ Decoded using {enc}")
                return df
        except Exception as e:
            print(f"⚠️ Failed with {enc}: {e}")
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "All encodings failed.")


# ------------------------------------------------
# 🧠 Keyword Extraction
# ------------------------------------------------
def extract_keywords(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    prompt = (
        "請根據以下摘要內容，萃取 10 個具語義代表性的學術關鍵詞，"
        "可為繁體中文或英文（依原文語言自動判斷），並用頓號（、）分隔。\n\n"
        f"{text}"
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        kw = r.choices[0].message.content.strip()
        kw = re.sub(r"[、;；\|／/，、\s]+", ", ", kw)
        kw = re.sub(r",\s*,+", ", ", kw)
        return kw.strip(" ,")
    except Exception as e:
        return f"Error: {e}"


# ------------------------------------------------
# 🔗 Edge Builder
# ------------------------------------------------
def build_edges(df_kw):
    pairs = []
    for _, row in df_kw.iterrows():
        terms = [t.strip() for t in re.split(r"[,，、;；\s]+", str(row["keywords"])) if t.strip()]
        terms = list(dict.fromkeys(terms))
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    e = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    return e.groupby(["Source", "Target"], as_index=False)["edge"].sum()


# ------------------------------------------------
# 🎨 Network Plot (with x/y axes & mean lines)
# ------------------------------------------------
def plot_network(vertices):
    plt.figure(figsize=(7, 6))
    # 2-D scatter layout for visual reference
    x = vertices["count"].rank().values
    y = vertices["cluster"]
    plt.scatter(x, y, s=vertices["count"] * 8, c=vertices["cluster"], cmap="tab10", alpha=0.8)

    # Mean reference lines
    xm, ym = x.mean(), y.mean()
    plt.axvline(x=xm, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=ym, color="red", linestyle="--", linewidth=1)

    # Labels
    for i, row in vertices.iterrows():
        plt.text(x[i] + 0.1, y[i], row["term"], fontsize=8)

    plt.xlabel("Term Rank (Count order)")
    plt.ylabel("Cluster ID")
    plt.title("Louvain Keyword Network (Mean Reference Lines in Red)")
    plt.grid(alpha=0.3, linestyle=":")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ------------------------------------------------
# 🏠 Home Route
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        html = open("index.html", "r", encoding="utf-8").read()
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(f"<h3>Error loading page:</h3><p>{e}</p>")


# ------------------------------------------------
# 📤 CSV Upload Route
# ------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = safe_read_csv(file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    if "abstract" not in df.columns:
        return HTMLResponse("<h3>❌ Missing 'abstract' column in CSV.</h3>")

    # Extract keywords
    df["keywords"] = df["abstract"].apply(lambda x: extract_keywords(str(x)))
    edges = build_edges(df)

    if edges.empty:
        return HTMLResponse("<h3>❌ No valid co-word pairs detected.</h3>")

    # Build Louvain clusters
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    comms = nx.community.louvain_communities(G, seed=42)
    cmap = {n: i + 1 for i, c in enumerate(comms) for n in c}
    deg = pd.Series(dict(G.degree(weight="edge")), name="count").reset_index()
    deg.columns = ["term", "count"]
    deg["cluster"] = deg["term"].map(cmap)
    vertices = deg.sort_values("count", ascending=False).head(30)

    # Plot network
    import base64
    img64 = plot_network(vertices)

    tmp_v = tempfile.NamedTemporaryFile(delete=False, suffix="_vertices.csv")
    tmp_r = tempfile.NamedTemporaryFile(delete=False, suffix="_relations.csv")
    vertices.to_csv(tmp_v.name, index=False, encoding="utf-8-sig")
    edges.to_csv(tmp_r.name, index=False, encoding="utf-8-sig")

    html = f"""
    <h2>✅ Analysis Complete</h2>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={tmp_v.name}">📥 Vertices CSV</a><br>
    <a href="/download?path={tmp_r.name}">📥 Relations CSV</a>
    """
    return HTMLResponse(html)


# ------------------------------------------------
# 📥 Download Route
# ------------------------------------------------
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))


# ------------------------------------------------
# 🚀 Local Dev Entry Point
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn, base64
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
