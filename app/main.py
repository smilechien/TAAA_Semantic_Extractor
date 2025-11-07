# ============================================================
# 🧠 TAAA Semantic–Co-Word Analyzer (v18.2)
# Adds R-style Top20+ cluster-balanced node selection
# ============================================================

import os, io
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
import community.community_louvain as community_louvain

# ------------------------------------------------------------
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ------------------------------------------------------------
def safe_read_csv(upload_file: UploadFile):
    content = upload_file.file.read()
    for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            continue
    raise ValueError("❌ Cannot decode CSV file.")

def normalize_headers(df):
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {
        "term1":"source","node1":"source","word1":"source","keyword1":"source",
        "term2":"target","node2":"target","word2":"target","keyword2":"target",
        "from":"source","to":"target","value":"weight"
    }
    df.rename(columns={k:v for k,v in alias.items() if k in df.columns}, inplace=True)
    return df

def wide_to_edges(df):
    cols=list(df.columns)
    src_col=cols[0]
    melt_df=df.melt(id_vars=[src_col], var_name="target_col", value_name="target")
    melt_df.rename(columns={src_col:"source"}, inplace=True)
    melt_df=melt_df[melt_df["target"].astype(str).str.strip().notna()]
    melt_df=melt_df[melt_df["target"].astype(str).str.strip()!=""]
    melt_df=melt_df[~melt_df["target"].astype(str).isin(["0","nan","None"])]
    melt_df["weight"]=1
    return melt_df[["source","target","weight"]]

def compute_aac(values):
    values=sorted([v for v in values if v>0], reverse=True)
    if len(values)<3: return None
    v1,v2,v3=values[:3]
    try:
        ratio=(v1/v2)/(v2/v3)
        aac=ratio/(1+ratio)
        return round(aac,2)
    except ZeroDivisionError:
        return None

# ---------- R-style top-term selector ----------
def select_top20_like_R(vertices, cap_limit=20):
    """Mimic the R script’s frequency-tiered selection rule."""
    nodes0 = vertices.copy()
    nodes0 = nodes0.assign(freq=nodes0.groupby("cluster")["cluster"].transform("count"))
    selected = pd.DataFrame(columns=nodes0.columns)
    remaining = cap_limit

    def pick_level(df, freq_pred, per_cluster, remaining):
        if remaining <= 0: return df.iloc[0:0]
        eligible = df.query(freq_pred)
        if eligible.empty: return df.iloc[0:0]
        cluster_summary = (eligible.groupby("cluster")
                           .agg(freq=("freq","first"), sum_value=("strength","sum"))
                           .reset_index()
                           .sort_values(["freq","sum_value"], ascending=[False,False]))
        max_clusters = max(0, remaining // per_cluster)
        chosen = cluster_summary.head(max_clusters)["cluster"].tolist()
        chosen_rows = (eligible[eligible["cluster"].isin(chosen)]
                       .sort_values(["cluster","strength"], ascending=[True,False])
                       .groupby("cluster").head(per_cluster))
        return chosen_rows

    # Level 1
    lvl1 = pick_level(nodes0, "freq >= 4", 4, remaining)
    selected = pd.concat([selected, lvl1])
    remaining = cap_limit - len(selected)

    # Level 2
    if remaining > 0:
        cap_limit = max(cap_limit, 22)
        lvl2 = pick_level(nodes0, "freq == 3", 3, remaining)
        selected = pd.concat([selected, lvl2])
        remaining = cap_limit - len(selected)

    # Level 3
    if remaining > 0:
        cap_limit = max(cap_limit, 21)
        lvl3 = pick_level(nodes0, "freq == 2", 2, remaining)
        selected = pd.concat([selected, lvl3])
        remaining = cap_limit - len(selected)

    # Level 4
    if remaining > 0:
        lvl4 = pick_level(nodes0, "freq == 1", 1, remaining)
        selected = pd.concat([selected, lvl4])
        remaining = cap_limit - len(selected)

    selected = (selected.drop_duplicates(subset=["term","cluster"])
                        .sort_values("strength", ascending=False)
                        .head(cap_limit))
    return selected

# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile):
    try:
        df=safe_read_csv(file)
        df.columns=[c.strip() for c in df.columns]
        mode="abstract" if len(df.columns)==1 else "coword"

        if mode=="coword":
            df=normalize_headers(df)
            ncols=len(df.columns)

            if ncols==3:
                cols=list(df.columns)
                edges=df.copy()
                edges.rename(columns={cols[0]:"source",cols[1]:"target",cols[2]:"weight"},inplace=True)
                edges["weight"]=pd.to_numeric(edges["weight"],errors="coerce").fillna(1)
            elif ncols==2:
                cols=list(df.columns)
                edges=df.copy()
                edges.rename(columns={cols[0]:"source",cols[1]:"target"},inplace=True)
                edges["weight"]=1
            else:
                edges=wide_to_edges(df)

            if edges.empty:
                return HTMLResponse("<h3>❌ No valid edges found.</h3>")

            G=nx.from_pandas_edgelist(edges,"source","target",["weight"])
            if G.number_of_nodes()==0:
                return HTMLResponse("<h3>❌ No valid nodes.</h3>")

            vertices=pd.DataFrame({"term":list(G.nodes())})
            vertices["degree"]=[d for _,d in G.degree()]
            vertices["strength"]=[sum(w for _,_,w in G.edges(n,data="weight")) for n in G.nodes()]

            partition=community_louvain.best_partition(G, weight="weight", resolution=1.0)
            vertices["cluster"]=vertices["term"].map(partition)

            aac=compute_aac(vertices["strength"].tolist())
            aac_label=f"AAC = {aac}" if aac is not None else "AAC unavailable"

            # ---- R-style Top20+ term filtering ----
            top_nodes = select_top20_like_R(vertices, cap_limit=20)
            terms_selected = top_nodes["term"].tolist()
            edges = edges[edges["source"].isin(terms_selected) & edges["target"].isin(terms_selected)]
            vertices = vertices[vertices["term"].isin(terms_selected)]

            # ---- CSV outputs ----
            vertices.to_csv(os.path.join(STATIC_DIR,"vertices.csv"),index=False,encoding="utf-8-sig")
            edges.to_csv(os.path.join(STATIC_DIR,"relations.csv"),index=False,encoding="utf-8-sig")
            theme_assign = vertices[["term","cluster"]].merge(edges,left_on="term",right_on="source",how="left")
            theme_assign.to_csv(os.path.join(STATIC_DIR,"theme_assignment.csv"),index=False,encoding="utf-8-sig")

            # ---- Bar chart ----
            theme_df = vertices.groupby("cluster").size().reset_index(name="member_count")
            plt.figure(figsize=(7,5))
            theme_df.plot.barh(x="cluster",y="member_count",legend=False)
            plt.title(f"H-Theme Distribution (Top 20+) — {aac_label}")
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_DIR,"h_theme_bar.png"),dpi=150)
            plt.close()

            # ---- Scatter ----
            vertices["value2"]=vertices["degree"]
            vertices["value"]=vertices["strength"]
            mean_x,mean_y=vertices["value2"].mean(),vertices["value"].mean()

            fig=px.scatter(vertices,x="value2",y="value",
                           color=vertices["cluster"].astype(str),
                           text="term",
                           title=f"Theme Scatter (Top 20+ Balanced Selection) — {aac_label}",
                           labels={"value2":"Degree","value":"Strength"},height=600)
            fig.add_shape(type="line",x0=mean_x,x1=mean_x,
                          y0=vertices["value"].min(),y1=vertices["value"].max(),
                          line=dict(color="red",dash="dot"))
            fig.add_shape(type="line",x0=vertices["value2"].min(),x1=vertices["value2"].max(),
                          y0=mean_y,y1=mean_y,line=dict(color="red",dash="dot"))
            fig.update_traces(textposition="top center",marker=dict(size=12,opacity=0.8))
            fig.write_html(os.path.join(STATIC_DIR,"theme_scatter.html"))

            # ---- HTML report ----
            html=f"""
            <h2>✅ Co-Word Analysis Completed (Top 20+ Selection)</h2>
            <p>Detected {len(vertices)} top terms across clusters.</p>
            <p><b>{aac_label}</b></p>
            <ul>
              <li><a href="/static/vertices.csv" target="_blank">Vertices CSV</a></li>
              <li><a href="/static/relations.csv" target="_blank">Relations CSV</a></li>
              <li><a href="/static/theme_assignment.csv" target="_blank">Theme–Article Assignment CSV</a></li>
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
              © 2025 Smile Chien · TAAA Semantic–Co-Word Analyzer v18.2
            </footer>
            """
            return HTMLResponse(html)

        else:
            return HTMLResponse("<h3>Abstract mode detected — feature available separately.</h3>")

    except Exception as e:
        return HTMLResponse(f"<h3>Internal Error:<br>{e}</h3>")
