# =============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer (main.py v15.6)
# =============================================================
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os, io, chardet
from community import community_louvain

# -------------------------------------------------------------
# ⚙️ Basic setup
# -------------------------------------------------------------
app = FastAPI(title="TAAA Semantic–CoWord Analyzer v15.6")
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# Chinese-safe font setup
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Noto Sans CJK TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# -------------------------------------------------------------
# 🧩 Helper: robust CSV reader (UTF-8 / Big5 fallback)
# -------------------------------------------------------------
def read_csv_safely(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    enc_guess = chardet.detect(raw).get("encoding", "utf-8")
    for enc in [enc_guess, "utf-8-sig", "big5", "cp950"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            upload.file.close()
            return df.fillna("")
        except Exception:
            continue
    upload.file.close()
    return pd.DataFrame()


# -------------------------------------------------------------
# 🧮 Main analysis route
# -------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    df = read_csv_safely(file)
    if df.empty:
        return HTMLResponse("<h3>❌ Unable to read CSV file.</h3>")

    # --- Treat any DOI-like strings safely ---
    def safe_str(x):
        x = str(x).strip()
        if "/" in x and not x.startswith("http"):
            return f"DOI:{x}"  # prefix to avoid type confusion
        return x

    df = df.applymap(safe_str)
    mode = "abstract" if df.shape[1] == 1 else "coword"
    preview_html = df.head(5).to_html(index=False, escape=False)

    # ---------------------------------------------------------
    # 🔗 Generate pairwise relations
    # ---------------------------------------------------------
    rel = []
    if mode == "abstract":
        for _, row in df.iterrows():
            terms = [t.strip() for t in str(row[0]).split(",") if t.strip()]
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    rel.append((terms[i], terms[j]))
    else:
        for _, row in df.iterrows():
            terms = [str(v).strip() for v in row if str(v).strip()]
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    rel.append((terms[i], terms[j]))

    rel = pd.DataFrame(rel, columns=["source", "target"])
    rel["weight"] = 1
    rel = rel.groupby(["source", "target"], as_index=False).size().rename(columns={"size": "weight"})
    rel.to_csv("static/relations.csv", index=False, encoding="utf-8-sig")

    vertices = pd.DataFrame(
        pd.concat([rel["source"], rel["target"]]).unique(), columns=["name"]
    )

    # Safe numeric conversion — ensures later sorting won't fail
    for col in vertices.columns:
        if col not in ["name"]:
            vertices[col] = pd.to_numeric(vertices[col], errors="coerce").fillna(0)

    # Add placeholder value2 for sorting
    if "value2" not in vertices.columns:
        vertices["value2"] = 0

    vertices.to_csv("static/vertices.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 🕸️ Build network and Louvain clusters
    # ---------------------------------------------------------
    G = nx.from_pandas_edgelist(rel, "source", "target", ["weight"])
    partition = community_louvain.best_partition(G, weight="weight")
    cluster_df = pd.DataFrame({"term": list(partition.keys()), "cluster": list(partition.values())})

    # --- Full cluster summary ---
    cluster_summary = cluster_df.groupby("cluster").size().reset_index(name="member_count")
    centrality = nx.degree_centrality(G)
    leaders, members_all = [], []
    for cid in cluster_summary["cluster"]:
        members = cluster_df.loc[cluster_df["cluster"] == cid, "term"].tolist()
        leader = max(members, key=lambda t: centrality.get(t, 0))
        leaders.append(leader)
        members_all.append(", ".join(members))

    cluster_summary["cluster_label"] = leaders
    cluster_summary["semantic_label"] = leaders
    cluster_summary["member_terms"] = members_all
    cluster_summary["carac"] = cluster_summary["cluster"]
    cluster_summary.to_csv("static/cluster_theme_full.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 🧠 Article–Theme assignment (TAAA)
    # ---------------------------------------------------------
    article_theme_records = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        terms = [v for v in row if str(v).strip()]
        clusters = cluster_df.loc[cluster_df["term"].isin(terms), "cluster"].tolist()
        theme = min(pd.Series(clusters).mode().values) if clusters else -1
        article_theme_records.append({"article_id": i, "theme": theme})
    pd.DataFrame(article_theme_records).to_csv("static/article_theme_assign.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 📊 H-Theme bar chart (semantic labels instead of cluster numbers)
    # ---------------------------------------------------------
    theme_counts = (
        cluster_summary[["semantic_label", "member_count"]]
        .sort_values("member_count", ascending=False)
        .reset_index(drop=True)
    )
    H = sum(theme_counts["member_count"] >= (theme_counts.index + 1))

    plt.figure(figsize=(8, 6))
    bars = plt.barh(theme_counts["semantic_label"], theme_counts["member_count"], color="steelblue")
    plt.xlabel("Member Count")
    plt.ylabel("Theme (Semantic Label)")
    plt.title(f"H-Theme Distribution by Semantic Label (H = {H})", fontsize=12, fontweight="bold")
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("static/theme_bar.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 🎨 Theme scatter (Top 20 categorical cluster colors)
    # ---------------------------------------------------------
    top20 = cluster_df.groupby("cluster").head(20)
    unique_clusters = sorted(top20["cluster"].unique())
    color_map = {cid: col for cid, col in zip(unique_clusters, px.colors.qualitative.Plotly[:len(unique_clusters)])}

    fig = go.Figure()
    for cid in unique_clusters:
        subset = top20[top20["cluster"] == cid]
        fig.add_trace(go.Scatter(
            x=list(range(len(subset))),
            y=[cid]*len(subset),
            mode="markers+text",
            name=f"Cluster {cid}",
            text=subset["term"],
            textposition="top center",
            marker=dict(size=10, color=color_map[cid]),
        ))

    fig.update_layout(
        title="Theme Scatter (Top 20 Terms per Cluster)",
        xaxis_title="Index",
        yaxis_title="Cluster (Categorical)",
        legend_title="Clusters",
        showlegend=True,
        height=600
    )
    fig.write_html("static/theme_scatter.html")

    # ---------------------------------------------------------
    # ✅ Result Page
    # ---------------------------------------------------------
    links_html = """
    <h2>✅ Analysis Complete</h2>
    <div class='downloads'>
      <p><strong>📁 Download Results</strong></p>
      <a href='/static/cluster_theme_full.csv' target='_blank'>📘 Full Theme Table (CSV)</a><br>
      <a href='/static/article_theme_assign.csv' target='_blank'>🧩 Article–Theme Assignment (CSV)</a><br>
      <a href='/static/theme_bar.png' target='_blank'>📊 H-Theme Bar (PNG)</a><br>
      <a href='/static/theme_scatter.html' target='_blank'>🎨 Theme Scatter (Interactive)</a><br>
      <a href='/static/relations.csv' target='_blank'>🔗 Relation Edges (CSV)</a><br>
      <a href='/static/vertices.csv' target='_blank'>🧱 Vertices (CSV)</a><br>
    </div>
    """

    html = f"""
    <html><head><meta charset='utf-8'>
    <title>TAAA Semantic–CoWord Analyzer</title>
    <style>
    body {{font-family:'Segoe UI','Noto Sans TC',sans-serif;background:#fafafa;margin:40px;}}
    .container{{background:#fff;padding:30px;border-radius:14px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}
    table{{border-collapse:collapse;width:100%;font-size:13px;}}
    th,td{{border:1px solid #ccc;padding:4px;}}
    </style></head>
    <body><div class='container'>
    <h1>🌐 TAAA Semantic–Co-Word Analyzer</h1>
    <h2>Results Summary</h2>
    <h3>📄 Preview of Uploaded CSV (Top 5 Rows)</h3>
    {preview_html}
    {links_html}
    </div></body></html>
    """
    return HTMLResponse(content=html)


# -------------------------------------------------------------
# 📂 Serve static files explicitly
# -------------------------------------------------------------
@app.get("/static/{filename}", response_class=FileResponse)
async def serve_static_file(filename: str):
    """Serve generated CSV/PNG/HTML files explicitly."""
    file_path = Path("static") / filename
    if file_path.exists():
        return FileResponse(path=file_path, filename=filename)
    else:
        return HTMLResponse(f"<h3>❌ File not found: {filename}</h3>", status_code=404)


# -------------------------------------------------------------
# 🏠 Home page
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    index_html = Path("static/index.html")
    if index_html.exists():
        return HTMLResponse(index_html.read_text(encoding="utf-8"))
    else:
        return HTMLResponse("<h2>Upload interface missing. Please redeploy with index.html.</h2>")
