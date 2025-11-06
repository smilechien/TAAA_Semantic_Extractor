# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v4.8 Render-Ready)
# Author: Smile
# Description:
#   • Multilingual keyword extraction (GPT-4o / TF-IDF fallback)
#   • Louvain clustering + TAAA theme assignment
#   • Auto font + bilingual labels
#   • Render health check & static CSV output
# ================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import tempfile, io, os, re, time, base64, chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import openai
from openai.error import APIError, RateLimitError, APITimeoutError

# ================================================================
# 🧩  App Initialization
# ================================================================
app = FastAPI(
    title="TAAA Semantic–Co-Word Analyzer",
    description="Multilingual GPT/TF-IDF hybrid semantic analyzer with TAAA assignment",
    version="4.8.0"
)

# --- Safe OpenAI client (Render compatible) ---
openai.api_key = os.getenv("OPENAI_API_KEY")


# ================================================================
# 🧠 GPT call with auto-retry and fallback
# ================================================================
def call_gpt_with_retry(prompt, model="gpt-4o-mini", max_retries=3, delay=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except (APIError, RateLimitError, APITimeoutError) as e:
            print(f"⚠️ API error ({attempt}/{max_retries}): {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"❌ Non-retryable error: {e}")
            break
    return None


# ================================================================
# 🧩  CSV Reader (robust multi-encoding)
# ================================================================
def safe_read_csv(uploaded: UploadFile) -> pd.DataFrame:
    raw = uploaded.file.read()
    uploaded.file.seek(0)
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
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "All encodings failed")


# ================================================================
# 🧮  TF-IDF Keyword Extraction (offline fallback)
# ================================================================
def extract_keywords_tfidf(docs, top_k=10):
    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vect.fit_transform(docs)
    names = vect.get_feature_names_out()
    keywords = []
    for row in X:
        top_idx = row.toarray().flatten().argsort()[-top_k:][::-1]
        kw = ", ".join([names[i] for i in top_idx])
        keywords.append(kw)
    return keywords


# ================================================================
# 🧠  GPT Semantic Extraction
# ================================================================
def extract_keywords_gpt(text: str, lang: str):
    prompt = (
        f"請根據以下摘要內容，以{lang}萃取10個具語義代表性的學術關鍵詞，"
        f"並用頓號（、）分隔：\n\n{text}"
    )
    out = call_gpt_with_retry(prompt)
    if not out:
        return ""
    out = re.sub(r"[、;；\|／/，、\s]+", ", ", out)
    return out.strip(" ,")


# ================================================================
# 🔗  Edge Builder
# ================================================================
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


# ================================================================
# 🎨  Scatter Plot (with theme boxes)
# ================================================================
def plot_scatter(vertices, theme_labels, lang="en"):
    font_family = "Noto Sans TC" if lang.startswith("zh") else "Segoe UI"
    plt.rcParams["font.family"] = font_family

    plt.figure(figsize=(7, 6))
    x = vertices["edge"]
    y = vertices["count"].rank()
    colors = vertices["cluster"]
    plt.scatter(x, y, s=vertices["count"] * 10, c=colors, cmap="tab10", alpha=0.8, edgecolor="k")

    for i, row in vertices.iterrows():
        plt.text(x[i] + 0.05, y[i], row["term"], fontsize=8)

    xm, ym = x.mean(), y.mean()
    plt.axvline(x=xm, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=ym, color="red", linestyle="--", linewidth=1)

    # vertical theme boxes
    for cid, theme in theme_labels.items():
        yc = vertices.loc[vertices["cluster"] == cid, "count"].rank().mean()
        plt.text(xm * 1.2, yc, theme, fontsize=12, color="red",
                 bbox=dict(facecolor="lightcoral", alpha=0.3, boxstyle="round,pad=0.4"))

    plt.xlabel("Edge (Co-occurrence Frequency)" if lang.startswith("en") else "共現頻率")
    plt.ylabel("Term Rank by Count" if lang.startswith("en") else "詞彙重要性排名")
    plt.title("Louvain Keyword Network" if lang.startswith("en") else "Louvain 關鍵詞網絡圖")
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ================================================================
# 🏠  Home Route
# ================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("<h2>Welcome to TAAA Semantic Analyzer</h2><p>Use /analyze_csv to upload your data.</p>")


# ================================================================
# 📈  Analysis Endpoint
# ================================================================
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    fn = file.filename
    try:
        df = safe_read_csv(file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ CSV read error: {e}</h3>")

    # --- Choose mode ---
    use_gpt = "smilechien" in fn.lower() or os.getenv("CHATGPT_ENV") == "true"
    lang = "Chinese"
    if "abstract" in df.columns:
        docs = df["abstract"].astype(str).fillna("")
        df["keywords"] = (
            [extract_keywords_gpt(t, lang) for t in docs]
            if use_gpt else extract_keywords_tfidf(docs)
        )
    else:
        return HTMLResponse("<h3>❌ Missing 'abstract' column.</h3>")

    edges = build_edges(df)
    if edges.empty:
        return HTMLResponse("<h3>❌ No valid co-word pairs detected.</h3>")

    # Louvain clustering
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "edge")
    comms = nx.community.louvain_communities(G, seed=42)
    cmap = {n: i + 1 for i, c in enumerate(comms) for n in c}
    deg = pd.Series(dict(G.degree(weight="edge")), name="count").reset_index()
    deg.columns = ["term", "count"]
    deg["cluster"] = deg["term"].map(cmap)
    deg["edge"] = deg["term"].map(lambda t: edges.query("Source == @t or Target == @t")["edge"].sum())
    vertices = deg.sort_values("count", ascending=False).head(20)

    # Auto theme names (TF-IDF style)
    theme_labels = {
        i + 1: "Theme " + str(i + 1) for i in range(len(comms))
    }

    img64 = plot_scatter(vertices, theme_labels, lang="zh" if "zh" in lang.lower() else "en")

    # --- Save outputs ---
    out_v = tempfile.NamedTemporaryFile(delete=False, suffix="_vertices.csv")
    out_r = tempfile.NamedTemporaryFile(delete=False, suffix="_relations.csv")
    out_t = tempfile.NamedTemporaryFile(delete=False, suffix="_themed.csv")

    vertices.to_csv(out_v.name, index=False, encoding="utf-8-sig")
    edges.to_csv(out_r.name, index=False, encoding="utf-8-sig")

    df_theme = df.copy()
    df_theme.insert(0, "theme", "Theme_Assigned")
    df_theme.to_csv(out_t.name, index=False, encoding="utf-8-sig")

    html = f"""
    <h2>✅ Analysis Complete ({'GPT' if use_gpt else 'TF-IDF'} Mode)</h2>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={out_v.name}">📥 Vertices CSV</a><br>
    <a href="/download?path={out_r.name}">📥 Relations CSV</a><br>
    <a href="/download?path={out_t.name}">📥 Themed CSV</a>
    """
    return HTMLResponse(html)


# ================================================================
# 📥  Download Route
# ================================================================
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))


# ================================================================
# 💓  Health Check
# ================================================================
@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# ================================================================
# 🚀  Local Entry Point
# ================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running TAAA Semantic Analyzer on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
