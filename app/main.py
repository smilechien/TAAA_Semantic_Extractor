# ============================================================
# 🧠 TAAA Semantic–Co-Word Analyzer (v17.0)
# Author: Smile Chien · 2025
# ============================================================

import os, io, re, json, requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ============================================================
# ⚙️ CONFIG
# ============================================================
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ============================================================
# 🧩 HELPERS
# ============================================================
def safe_read_csv(upload_file: UploadFile):
    """Try multiple encodings for CSV files."""
    content = upload_file.file.read()
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc)
            return df
        except Exception:
            continue
    raise ValueError("❌ Cannot decode CSV file. Try UTF-8 or Big5 encoding.")

def fetch_openalex_abstract(doi):
    """Get abstract text from OpenAlex."""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json()
        if data.get("abstract_inverted_index"):
            abstract = data["abstract_inverted_index"]
            flat_pos = []
            for word, idxs in abstract.items():
                for i in idxs:
                    flat_pos.append((i, word))
            text = " ".join([w for i, w in sorted(flat_pos)])
            return text
        return None
    except Exception:
        return None

def fetch_openalex_keywords(doi):
    """Get OpenAlex top concepts as keywords."""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            return None
        data = res.json()
        if data.get("concepts"):
            return "; ".join([c["display_name"] for c in data["concepts"][:10]])
        return None
    except Exception:
        return None

def normalize_headers(df):
    """Standardize potential source/target column names."""
    df.columns = [c.strip().lower() for c in df.columns]
    alias_map = {
        "term1": "source", "node1": "source", "word1": "source", "keyword1": "source",
        "term2": "target", "node2": "target", "word2": "target", "keyword2": "target",
        "from": "source", "to": "target"
    }
    df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)
    return df

# ============================================================
# 🏠 HOME PAGE
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# 📊 MAIN ANALYSIS
# ============================================================
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile):
    try:
        df = safe_read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        mode = "abstract" if len(df.columns) == 1 else "coword"

        # --------------------------------------------------------
        # 🔍 ABSTRACT MODE (DOI Extraction)
        # --------------------------------------------------------
        if mode == "abstract":
            col = df.columns[0]
            if not df[col].astype(str).str.contains("/").any():
                return HTMLResponse("<h3>❌ No DOI format detected in file.</h3>")

            df["doi"] = df[col].astype(str)
            abstracts, keywords = [], []

            OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
            use_openai = bool(OPENAI_KEY)

            for doi in df["doi"]:
                text = fetch_openalex_abstract(doi)
                abstracts.append(text or "")
                kw = fetch_openalex_keywords(doi) or ""
                keywords.append(kw)

            df["abstract"] = abstracts
            df["keywords"] = keywords
            out_csv = os.path.join(STATIC_DIR, "doi_semantic_keywords.csv")
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            html = f"""
            <h2>✅ Abstract Mode Completed</h2>
            <p>Extracted {len(df)} records.</p>
            <ul>
              <li><a href="/static/doi_semantic_keywords.csv" target="_blank">📄 DOI Semantic Keywords CSV</a></li>
            </ul>
            <div style='margin-top:40px;'>
              <button onclick="window.location.href='/'"
                style="background:#007ACC;color:white;border:none;padding:10px 24px;
                       border-radius:6px;cursor:pointer;font-size:15px;">
                ⬅️ Return to Home
              </button>
            </div>
            <footer style="margin-top:25px;font-size:13px;color:#888;">
              © 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v17.0
            </footer>
            """
            return HTMLResponse(html)

        # --------------------------------------------------------
        # 🔗 CO-WORD MODE
        # --------------------------------------------------------
        if mode == "coword":
            edges = normalize_headers(df)

            # validate required columns
            if "source" not in edges.columns or "target" not in edges.columns:
                cols = ", ".join(df.columns)
                return HTMLResponse(f"<h3>❌ Missing 'source' or 'target' column.<br>"
                                    f"Detected columns: {cols}</h3>")

            edges["weight"] = pd.to_numeric(edges.get("weight", 1), errors="coerce").fillna(1)
            G = nx.from_pandas_edgelist(edges, "source", "target", ["weight"])
            vertices = pd.DataFrame({"term": list(G.nodes())})
            vertices["degree"] = [val for (node, val) in G.degree()]

            # clustering (Louvain)
            import community
            partition = community.best_partition(G)
            vertices["cluster"] = vertices["term"].map(partition)

            theme_df = (
                vertices["cluster"].value_counts()
                .reset_index()
                .rename(columns={"index": "cluster", "cluster": "member_count"})
            )
            theme_df.to_csv(os.path.join(STATIC_DIR, "theme.csv"), index=False, encoding="utf-8-sig")

            # bar chart
            plt.figure(figsize=(7, 5))
            theme_df.sort_values("member_count", ascending=True).plot.barh(
                x="cluster", y="member_count", legend=False)
            plt.title(f"Top H-Theme Distribution (H = {len(theme_df)})")
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_DIR, "h_theme_bar.png"), dpi=150)
            plt.close()

            # scatter plot with red mean lines
            mean_x = vertices["degree"].mean()
            mean_y = vertices["degree"].mean()
            fig = px.scatter(vertices.head(20), x="degree", y="degree",
                             color=vertices["cluster"].astype(str),
                             text="term",
                             title="Theme Scatter (Top 20 Terms, by Category)",
                             height=600)
            fig.add_shape(type="line", x0=mean_x, x1=mean_x,
                          y0=vertices["degree"].min(), y1=vertices["degree"].max(),
                          line=dict(color="red", dash="dot"))
            fig.add_shape(type="line", x0=vertices["degree"].min(), x1=vertices["degree"].max(),
                          y0=mean_y, y1=mean_y,
                          line=dict(color="red", dash="dot"))
            fig.write_html(os.path.join(STATIC_DIR, "theme_scatter.html"))

            html = f"""
            <h2>✅ Co-Word Analysis Completed</h2>
            <p>Detected {len(vertices)} vertices and {len(edges)} unique relations.</p>
            <ul>
              <li><a href="/static/theme.csv" target="_blank">Theme CSV</a></li>
              <li><a href="/static/h_theme_bar.png" target="_blank">H-Theme Bar (PNG)</a></li>
              <li><a href="/static/theme_scatter.html" target="_blank">Theme Scatter (Interactive)</a></li>
            </ul>
            <div style="margin-top:40px;">
              <button onclick="window.location.href='/'"
                style="background:#007ACC;color:white;border:none;padding:10px 24px;
                       border-radius:6px;cursor:pointer;font-size:15px;">
                ⬅️ Return to Home
              </button>
            </div>
            <footer style="margin-top:25px;font-size:13px;color:#888;">
              © 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v17.0
            </footer>
            """
            return HTMLResponse(html)

    except Exception as e:
        return HTMLResponse(f"<h3>Internal Error:<br>{e}</h3>")
