# ================================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (v5.2, Render-ready)
# ================================================================
# Author: Smile Chien
# Description:
#   - Serves index.html UI for file upload & mode selection
#   - Detects language and uses GPT or TF-IDF engine automatically
#   - Performs co-word or abstract-theme analysis
#   - Generates scatter & bar plots and downloadable CSVs
# ================================================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tempfile, os, io, re, base64, random
from openai import OpenAI
from openai._exceptions import APIError, RateLimitError, APITimeoutError
import chardet, time

# ------------------------------------------------
# 🔧 App setup
# ------------------------------------------------
app = FastAPI(title="TAAA Semantic–Co-Word Analyzer v5.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Create static folder if not exist
os.makedirs("app/static", exist_ok=True)

# ------------------------------------------------
# 🧠 Engine initialization (GPT safe client)
# ------------------------------------------------
def init_openai_client():
    """Initialize OpenAI client safely."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No API key found — fallback to TF-IDF mode.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        # simple heartbeat
        _ = client.models.list()
        return client
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")
        return None

client = init_openai_client()

# ------------------------------------------------
# 🧩 Safe CSV reader
# ------------------------------------------------
def safe_read_csv(uploaded: UploadFile) -> pd.DataFrame:
    """Read CSV with automatic encoding detection."""
    raw = uploaded.file.read()
    uploaded.file.seek(0)
    guess = chardet.detect(raw).get("encoding") or "utf-8"
    for enc in [guess, "utf-8-sig", "utf-8", "big5", "cp950", "latin1"]:
        try:
            text = raw.decode(enc, errors="ignore")
            buf = io.StringIO(text)
            df = pd.read_csv(buf)
            if not df.empty:
                print(f"✅ Decoded with {enc}")
                return df
        except Exception:
            continue
    raise ValueError("❌ Unable to decode CSV. Save as UTF-8-SIG and retry.")

# ------------------------------------------------
# 🧩 Keyword extraction (GPT or TF-IDF)
# ------------------------------------------------
def extract_keywords_tfidf(texts, top_k=10):
    vect = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf = vect.fit_transform(texts)
    terms = vect.get_feature_names_out()
    keywords = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().ravel()
        top_terms = [terms[idx] for idx in row.argsort()[-top_k:][::-1]]
        keywords.append(", ".join(top_terms))
    return keywords

def extract_keywords_gpt(text):
    if not client:
        return "Error: GPT unavailable"
    prompt = (
        "請根據以下摘要內容，萃取10個具語義代表性的學術關鍵詞，"
        "以原文語言呈現，並用逗號分隔：\n\n" + text
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            out = r.choices[0].message.content.strip()
            return re.sub(r"[、，;；\s]+", ", ", out)
        except (APIError, RateLimitError, APITimeoutError):
            time.sleep(2 ** attempt)
        except Exception as e:
            print("GPT error:", e)
            break
    return "Error: GPT timeout"

# ------------------------------------------------
# 🔗 Build co-word edges
# ------------------------------------------------
def build_edges(df_kw):
    pairs = []
    for kw_str in df_kw["keywords"]:
        terms = [t.strip() for t in re.split(r"[,，;；\s]+", str(kw_str)) if t.strip()]
        terms = list(dict.fromkeys(terms))
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                pairs.append((terms[i], terms[j], 1))
    if not pairs:
        return pd.DataFrame(columns=["Source", "Target", "edge"])
    e = pd.DataFrame(pairs, columns=["Source", "Target", "edge"])
    return e.groupby(["Source", "Target"], as_index=False)["edge"].sum()

# ------------------------------------------------
# 🎨 Scatter plot
# ------------------------------------------------
def plot_scatter(vertices, themes=None, lang="en"):
    plt.figure(figsize=(7, 6))
    x = vertices["edge"]
    y = vertices["count"].rank(ascending=False)
    cmap = plt.cm.tab10

    for _, row in vertices.iterrows():
        plt.scatter(row["edge"], row["count"], s=row["count"] * 6,
                    c=[cmap(row["cluster"] % 10)], edgecolor="k", alpha=0.8)
        plt.text(row["edge"] + 0.05, row["count"], row["term"], fontsize=8)

    plt.axvline(x.mean(), color="red", linestyle="--", lw=1)
    plt.axhline(y.mean(), color="red", linestyle="--", lw=1)

    if lang.startswith("zh"):
        plt.xlabel("共現頻率 (Edge Number)", fontsize=12, family="Noto Sans TC")
        plt.ylabel("詞語排名 (Term Rank by Count)", fontsize=12, family="Noto Sans TC")
        plt.title("主題語意網絡圖 (Louvain 聚類)", fontsize=14, family="Noto Sans TC")
    else:
        plt.xlabel("Edge Number (Co-occurrence Frequency)", fontsize=12, family="Segoe UI")
        plt.ylabel("Term Rank by Total Count", fontsize=12, family="Segoe UI")
        plt.title("Keyword Co-Word Network (Louvain Clusters)", fontsize=14, family="Segoe UI")

    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------
# 🏠 Home route
# ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("app/templates/index.html")

# ------------------------------------------------
# 📤 CSV upload and analysis
# ------------------------------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...), mode: str = Form("abstract")):
    df = safe_read_csv(file)
    if "abstract" not in df.columns and mode == "abstract":
        return HTMLResponse("<h3>❌ Missing 'abstract' column.</h3>")

    # choose language & engine
    try:
        sample_text = " ".join(df.iloc[0].astype(str))
        lang = detect(sample_text)
    except Exception:
        lang = "en"

    engine = "GPT" if client else "TF-IDF"

    if mode == "abstract":
        if engine == "GPT":
            df["keywords"] = df["abstract"].apply(extract_keywords_gpt)
        else:
            df["keywords"] = extract_keywords_tfidf(df["abstract"])
    else:
        df["keywords"] = df.iloc[:, 0].astype(str)

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
    edge_count = edges.groupby("Source")["edge"].sum().reset_index()
    deg = deg.merge(edge_count, left_on="term", right_on="Source", how="left").fillna(0)
    deg.rename(columns={"edge": "edge"}, inplace=True)

    img64 = plot_scatter(deg.head(20), lang=lang)

    # Write downloadable results
    out_v = f"app/static/{file.filename}_vertices.csv"
    out_r = f"app/static/{file.filename}_relations.csv"
    deg.to_csv(out_v, index=False, encoding="utf-8-sig")
    edges.to_csv(out_r, index=False, encoding="utf-8-sig")

    html = f"""
    <h2>✅ Analysis Complete ({engine} Engine)</h2>
    <img src="data:image/png;base64,{img64}" style="max-width:95%;border:1px solid #ccc"/><br><br>
    <a href="/download?path={out_v}">📥 Download Vertices CSV</a><br>
    <a href="/download?path={out_r}">📥 Download Relations CSV</a><br>
    <a href="/">🏠 Back to Home</a>
    """
    return HTMLResponse(html)

# ------------------------------------------------
# 📥 Download route
# ------------------------------------------------
@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# ------------------------------------------------
# 💓 Health check route
# ------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "engine": "GPT" if client else "TF-IDF"}

# ------------------------------------------------
# 🚀 Run locally
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
