# ============================================================
# 🧠 TAAA Semantic–Co-Word Analyzer (v17.6)
# Robust handling for 1–3+ column data and disconnected graphs
# ============================================================

import os, io, requests
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

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_read_csv(upload_file: UploadFile):
    """Read uploaded CSV safely using multiple encodings."""
    content = upload_file.file.read()
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Cannot decode CSV file. Try UTF-8 or Big5 encoding.")

def normalize_headers(df):
    """Normalize header names (lowercase + trim)."""
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {
        "term1": "source", "node1": "source", "word1": "source", "keyword1": "source",
        "term2": "target", "node2": "target", "word2": "target", "keyword2": "target",
        "from": "source", "to": "target", "value": "weight"
    }
    df.rename(columns={k: v for k, v in alias.items() if k in df.columns}, inplace=True)
    return df

def wide_to_edges(df):
    """Convert wide co-word/affiliation table into edge list."""
    cols = list(df.columns)
    src_col = cols[0]
    melt_df = df.melt(id_vars=[src_col], var_name="target_col", value_name="target")
    melt_df.rename(columns={src_col: "source"}, inplace=True)
    melt_df = melt_df[melt_df["target"].astype(str).str.strip().notna()]
    melt_df = melt_df[melt_df["target"].astype(str).str.strip() != ""]
    melt_df = melt_df[~melt_df["target"].astype(str).isin(["0", "nan", "None"])]
    melt_df["weight"] = 1
    return melt_df[["source", "target", "weight"]]

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile):
    try:
        df = safe_read_csv(file)
        df.columns = [c.strip() for c in df.columns]
        mode = "abstract" if len(df.columns) == 1 else "coword"

        # --------------------------------------------------------
        # 🔗 CO-WORD MODE
        # --------------------------------------------------------
        if mode == "coword":
            df = normalize_headers(df)
            ncols = len(df.columns)

            # CASE 1️⃣: 3 columns = weighted edges
            if ncols == 3:
                cols = list(df.columns)
                edges = df.copy()
                edges.rename(columns={cols[0]: "source", cols[1]: "target", cols[2]: "weight"}, inplace=True)
                edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1)

            # CASE 2️⃣: 2 columns = unweighted edges
            elif ncols == 2:
                cols = list(df.columns)
                edges = df.copy()
                edges.rename(columns={cols[0]: "source", cols[1]: "target"}, inplace=True)
                edges["weight"] = 1

            # CASE 3️⃣: ≥4 columns = wide co-word table
            else:
                edges = wide_to_edges(df)

            # --- Build graph ---
            if edges.empty:
                return HTMLResponse("<h3>❌ No valid edge data found.</h3>")
            G = nx.from_pandas_edgelist(edges, "source", "target", ["weight"])
            if G.number_of_nodes() == 0:
                return HTMLResponse("<h3>❌ No valid nodes in data.</h3>")

            vertices = pd.DataFrame({"term": list(G.nodes())})
            vertices["degree"] = [d for _, d in G.degree()]
            vertices["strength"] = [
                sum(w for _, _, w in G.edges(n, data="weight")) for n in G.nodes()
            ]

            # --- Louvain clustering ---
            try:
                import community
                partition = community.best_partition(G)
                vertices["cluster"] = vertices["term"].map(partition)
            except Exception:
                # fallback for disconnected or small networks
                vertices["cluster"] = 0

            # --- Safety fallback if cluster missing ---
            if "cluster" not in vertices.columns:
                vertices["cluster"] = 0

            # --- Theme summary ---
            theme_df = (
                vertices.groupby("cluster").size()
                .reset_index(name="member_count")
                .sort_values("member_count", ascending=False)
            )
            theme_df.to_csv(os.path.join(STATIC_DIR, "theme.csv"), index=False, encoding="utf-8-sig")

            # --- Bar chart ---
            plt.figure(figsize=(7, 5))
            theme_df.plot.barh(x="cluster", y="member_count", legend=False)
            plt.title(f"Top H-Theme Distribution (H = {len(theme_df)})")
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_DIR, "h_theme_bar.png"), dpi=150)
            plt.close()

            # --- Scatter chart ---
            vertices["value2"] = vertices["degree"]
            vertices["value"] = vertices["strength"]
            mean_x = vertices["value2"].mean()
            mean_y = vertices["value"].mean()

            fig = px.scatter(
                vertices.head(20),
                x="value2",
                y="value",
                color=vertices["cluster"].astype(str),
                text="term",
                title="Theme Scatter (Top 20 Terms by Degree & Strength)",
                labels={"value2": "X: Degree", "value": "Y: Strength"},
                height=600,
            )
            fig.add_shape(
                type="line",
                x0=mean_x,
                x1=mean_x,
                y0=vertices["value"].min(),
                y1=vertices["value"].max(),
                line=dict(color="red", dash="dot"),
            )
            fig.add_shape(
                type="line",
                x0=vertices["value2"].min(),
                x1=vertices["value2"].max(),
                y0=mean_y,
                y1=mean_y,
                line=dict(color="red", dash="dot"),
            )
            fig.update_traces(textposition="top center", marker=dict(size=12, opacity=0.8))
            fig.write_html(os.path.join(STATIC_DIR, "theme_scatter.html"))

            html = f"""
            <h2>✅ Co-Word Analysis Completed</h2>
            <p>Detected {len(vertices)} vertices and {len(edges)} relations.</p>
            <ul>
              <li><a href="/static/theme.csv" target="_blank">Theme CSV</a></li>
              <li><a href="/static/h_theme_bar.png" target="_blank">H-Theme Bar (PNG)</a></li>
              <li><a href="/static/theme_scatter.html" target="_blank">Theme Scatter (Interactive)</a></li>
            </ul>
            <footer style="margin-top:25px;font-size:13px;color:#888;">
              © 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v17.6
            </footer>
            """
            return HTMLResponse(html)

        # --------------------------------------------------------
        # 🔍 ABSTRACT MODE (1 column)
        # --------------------------------------------------------
        else:
            return HTMLResponse("<h3>Abstract mode (DOI) detected — feature available separately.</h3>")

    except Exception as e:
        return HTMLResponse(f"<h3>Internal Error:<br>{e}</h3>")
