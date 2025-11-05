# ============================================================
# 🧭 TAAA Semantic Extractor — AAC Network Analyzer (FastAPI)
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io, base64, os

app = FastAPI()

# ------------------------------------------------------------
# Serve static assets (index.html, style.css, sample CSVs)
# ------------------------------------------------------------
app.mount("/", StaticFiles(directory="app", html=True), name="static")

# ------------------------------------------------------------
# Compute AAC and draw simple weighted network
# ------------------------------------------------------------
def compute_aac(df):
    df = df.copy()
    df["edge"] = df["edge"].astype(float)
    total_edge = df["edge"].sum()

    # Normalize edge weights
    df["weight"] = df["edge"] / total_edge

    # Compute node AAC (sum of normalized edge weights per node)
    strength = (
        pd.concat([
            df.groupby("Source")["weight"].sum(),
            df.groupby("Target")["weight"].sum()
        ])
        .groupby(level=0)
        .sum()
        .reset_index()
    )
    strength.columns = ["node", "AAC"]
    return df, strength


def draw_network(df):
    G = nx.from_pandas_edgelist(df, "Source", "Target", edge_attr="edge", create_using=nx.Graph())
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_color="#7FB3D5", node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ------------------------------------------------------------
# POST endpoint: analyze uploaded CSV
# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        return HTMLResponse(f"<h3>❌ 無法讀取檔案: {e}</h3>")

    data.columns = [c.strip() for c in data.columns]
    required = {"Source", "Target", "edge"}
    if not required.issubset(data.columns):
        return HTMLResponse(f"<h3>❌ CSV 必須包含欄位: {', '.join(required)}</h3>")

    df_edges, df_nodes = compute_aac(data)
    img_base64 = draw_network(df_edges)

    # Prepare downloadable CSV
    out_buf = io.StringIO()
    df_nodes.to_csv(out_buf, index=False)
    csv_bytes = io.BytesIO(out_buf.getvalue().encode("utf-8"))
    app.state.csv_bytes = csv_bytes

    html = f"""
    <h2>✅ 分析完成 / Analysis Complete</h2>
    <p>Sum of edges = {df_edges['edge'].sum():.2f}</p>
    <img src="data:image/png;base64,{img_base64}" alt="AAC Network Graph" style="max-width:90%;border:1px solid #ccc"/><br><br>
    <a href="/download_csv" target="_blank">📥 下載結果 (CSV)</a>
    """
    return HTMLResponse(html)


# ------------------------------------------------------------
# GET endpoint: download CSV
# ------------------------------------------------------------
@app.get("/download_csv")
async def download_csv():
    csv_bytes = getattr(app.state, "csv_bytes", None)
    if not csv_bytes:
        return HTMLResponse("<h3>❌ No CSV available. Run analysis first.</h3>")
    csv_bytes.seek(0)
    return StreamingResponse(
        csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )
