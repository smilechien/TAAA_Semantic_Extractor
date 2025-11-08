# ============================================================
# 🌐 main.py — TAAA Semantic–Co-Word Analyzer v20.2
# Author: Smile
# Description: 5-Module pipeline for semantic / co-word clustering
# ============================================================

from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd, numpy as np, io, os
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from community import community_louvain

app = FastAPI()
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# ============================================================
# 🧩 Helper functions
# ============================================================

def safe_read_csv(file: UploadFile) -> pd.DataFrame:
    """Auto-detect UTF-8 or Big5"""
    data = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8")
    except Exception:
        df = pd.read_csv(io.BytesIO(data), encoding="big5", errors="ignore")
    return df.fillna("")

def compute_aac(values):
    """AAC dominance metric from top 3 values"""
    vals = sorted(values, reverse=True)[:3]
    if len(vals) < 3 or vals[1] == 0 or vals[2] == 0:
        return 0
    v1, v2, v3 = vals
    aac = ((v1 / v2) / (v2 / v3)) / (1 + ((v1 / v2) / (v2 / v3)))
    return round(aac, 2)

# ============================================================
# 🧭 Analysis Route
# ============================================================

@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile):
    df = safe_read_csv(file)
    ncols = len(df.columns)

    # --------------------------------------------------------
    # Mode 1 → Abstract (1-column)   Mode 2 → Co-Word (multi)
    # --------------------------------------------------------
    mode = 1 if ncols == 1 else 2
    file_base = os.path.splitext(file.filename)[0]

    # ============================================================
    # 🧠 MODE 1: ABSTRACT — Semantic Terms (placeholder GPT/TF-IDF)
    # ============================================================
    if mode == 1:
        terms = []
        for _, row in df.iterrows():
            text = str(row.iloc[0])
            # placeholder tokenization (simulate GPT extraction)
            toks = [t.strip(" ,.;") for t in text.split() if len(t) > 2]
            terms.append(list(set(toks[:10])))
        # convert to dataframe (article × terms)
        max_len = max(len(x) for x in terms)
        padded = [x + [""] * (max_len - len(x)) for x in terms]
        df_terms = pd.DataFrame(padded)
        df_terms.to_csv("static/semantic_terms.csv", index=False, encoding="utf-8-sig")
        return HTMLResponse(
            f"<h3>🧠 Abstract Mode completed — semantic terms extracted.</h3>"
            f"<a href='/static/semantic_terms.csv'>Download terms</a><br>"
            f"<a href='/'>Return Home</a>"
        )

    # ============================================================
    # 🐾 MODE 2: CO-WORD — Build network and cluster
    # ============================================================
    else:
        # build all co-occurrence pairs
        pairs = []
        for _, row in df.iterrows():
            terms = [str(x).strip() for x in row if str(x).strip()]
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    pairs.append(tuple(sorted([terms[i], terms[j]])))
        edges = pd.DataFrame(pairs, columns=["term1", "term2"])
        edges["connection_count"] = 1
        edges = edges.groupby(["term1", "term2"], as_index=False)["connection_count"].sum()

        # build vertex list
        all_terms = pd.Series(pd.concat([edges["term1"], edges["term2"]]))
        vertices = pd.DataFrame({"name": all_terms.value_counts().index,
                                 "value": all_terms.value_counts().values})

        # edge strength per node
        strength = edges.groupby("term1")["connection_count"].sum().add(
            edges.groupby("term2")["connection_count"].sum(), fill_value=0)
        vertices["value2"] = vertices["name"].map(strength).fillna(0)

        # enforce at least 20 vertices
        if len(vertices) < 20:
            extra = 20 - len(vertices)
            add = pd.DataFrame({
                "name": [f"dummy_{i}" for i in range(extra)],
                "value": 1, "value2": 1})
            vertices = pd.concat([vertices, add], ignore_index=True)

        # --------------------------------------------------------
        # Louvain clustering
        # --------------------------------------------------------
        G = nx.from_pandas_edgelist(edges, "term1", "term2", ["connection_count"])
        cluster = community_louvain.best_partition(G, weight="connection_count", random_state=42)
        vertices["cluster"] = vertices["name"].map(cluster)
        # mark cluster leaders
        leaders = vertices.sort_values(["cluster", "value2"], ascending=[True, False]) \
                          .groupby("cluster").head(1)[["cluster", "name"]] \
                          .rename(columns={"name": "leader_label"})
        vertices = vertices.merge(leaders, on="cluster", how="left")

        # --------------------------------------------------------
        # Theme summary (theme.csv)
        # --------------------------------------------------------
        theme = vertices.groupby("cluster").agg(
            leader=("leader_label", "first"),
            member_count=("name", "count"),
            members=("name", lambda x: ", ".join(x))
        ).reset_index(drop=True)
        theme.to_csv("static/theme.csv", index=False, encoding="utf-8-sig")

        # --------------------------------------------------------
        # TAAA theme assignment: dominant cluster per row
        # --------------------------------------------------------
        def assign_theme(row):
            row_terms = [str(x).strip() for x in row if str(x).strip()]
            clusters = [cluster[t] for t in row_terms if t in cluster]
            if not clusters: return "Unclassified"
            vals, cnts = np.unique(clusters, return_counts=True)
            top = vals[np.argmax(cnts)]
            return theme.loc[theme["leader"] == leaders.loc[leaders["cluster"] == top, "leader_label"].iloc[0],
                             "leader"].values[0]

        theme_assign = pd.DataFrame({
            "article_id": range(1, len(df) + 1),
            "theme": df.apply(assign_theme, axis=1)
        })
        theme_assign.to_csv("static/theme_assignment.csv", index=False, encoding="utf-8-sig")

        # --------------------------------------------------------
        # AAC + H-theme computation
        # --------------------------------------------------------
        aac = compute_aac(vertices["value"].tolist())
        theme = theme.sort_values("member_count", ascending=False)
        h = max(i + 1 for i, c in enumerate(theme["member_count"]) if c >= i + 1)

        # h-theme bar
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(theme) + 1), theme["member_count"], color="#69b3a2")
        plt.axhline(y=h, color="red", linestyle="--")
        plt.axvline(x=h, color="red", linestyle="--")
        plt.title(f"H-Theme Bar (h={h})")
        plt.xlabel("Theme Rank")
        plt.ylabel("Member Count")
        plt.tight_layout()
        plt.savefig("static/h_theme_bar.png", dpi=150)
        plt.close()

        # theme scatter (interactive)
        mean_x, mean_y = vertices["value2"].mean(), vertices["value"].mean()
        fig = px.scatter(vertices, x="value2", y="value", color=vertices["cluster"].astype(str),
                         size="value", hover_name="name",
                         title=f"Theme Scatter — AAC={aac}")
        fig.add_vline(x=mean_x, line_dash="dot", line_color="red")
        fig.add_hline(y=mean_y, line_dash="dot", line_color="red")
        fig.write_html("static/theme_scatter.html")

        # --------------------------------------------------------
        # Save all CSV outputs
        # --------------------------------------------------------
        vertices.to_csv("static/vertices.csv", index=False, encoding="utf-8-sig")
        edges.to_csv("static/relations.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame([{
            "AAC": aac,
            "Total_Clusters": theme.shape[0],
            "Nodes": len(vertices),
            "Edges": len(edges)
        }])
        summary.to_csv("static/summary.csv", index=False, encoding="utf-8-sig")

        # --------------------------------------------------------
        # Final HTML output
        # --------------------------------------------------------
        links = """
        <h3>✅ Analysis Completed</h3>
        <p><b>5 CSVs:</b><br>
        <a href='/static/vertices.csv'>vertices.csv</a> |
        <a href='/static/relations.csv'>relations.csv</a> |
        <a href='/static/theme.csv'>theme.csv</a> |
        <a href='/static/theme_assignment.csv'>theme_assignment.csv</a> |
        <a href='/static/summary.csv'>summary.csv</a></p>
        <p><b>2 Plots:</b><br>
        <a href='/static/h_theme_bar.png'>📊 H-Theme Bar</a> |
        <a href='/static/theme_scatter.html'>🌐 Theme Scatter</a></p>
        <p><a href='/'>Return Home</a></p>
        """
        return HTMLResponse(links)

# ============================================================
# 🏠 Home Route
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ============================================================
# Static Mount
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")
