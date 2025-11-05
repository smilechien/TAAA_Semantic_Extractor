# ================================================================
# 🌐 TAAA Semantic + Cluster Visualizer (Top-20 Edges + AAC + Edge Tooltips)
# ================================================================
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from openai import OpenAI
import pandas as pd, networkx as nx, tempfile, os, re, itertools

app = FastAPI(
    title="TAAA Semantic + Cluster Visualizer (AAC)",
    description="Upload abstracts, extract multilingual keywords, build top-20 co-word clusters, compute AAC metric, and show interactive tooltips.",
    version="3.6.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- Home --------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <h2>TAAA Semantic + Cluster Visualizer (AAC Metric)</h2>
    <form action="/visualize" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <button type="submit">Analyze & Visualize</button>
    </form>
    """)

# -------------------- Visualization --------------------
@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "abstract" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "Missing 'abstract' column."})

    df["keywords"] = df["abstract"].apply(lambda x: extract_keywords(str(x)))
    edges, nodes = build_cooccurrence(df)

    # keep top-20 edges
    edges = edges.sort_values("Weight", ascending=False).head(20)
    sel = set(edges["Source"]) | set(edges["Target"])
    nodes = nodes[nodes["keyword"].isin(sel)]

    # clusters
    G = nx.from_pandas_edgelist(edges, "Source", "Target", "Weight")
    clusters = {n: i for i, comp in enumerate(nx.connected_components(G), 1) for n in comp}
    nodes["cluster"] = nodes["keyword"].map(clusters).fillna(0).astype(int)

    # AAC
    top3 = nodes["frequency"].nlargest(3).tolist() + [1, 1, 1]
    r1, r2, r3 = top3[:3]
    aac = ((r1/r2)/(1+r1/r2))*((r2/r3)/(1+r2/r3))
    aac = round(aac, 2)

    # save CSVs
    kw_file = tempfile.NamedTemporaryFile(delete=False, suffix="_keywords.csv")
    cl_file = tempfile.NamedTemporaryFile(delete=False, suffix="_clusters.csv")
    df.to_csv(kw_file.name, index=False, encoding="utf-8-sig")
    nodes.to_csv(cl_file.name, index=False, encoding="utf-8-sig")

    # positions
    pos = nx.spring_layout(G, k=0.4, seed=42)
    data_edges = []
    for _, w in edges.iterrows():
        s, t = w["Source"], w["Target"]
        wgt = float(w["Weight"])
        c1, c2 = int(nodes.loc[nodes["keyword"]==s,"cluster"].values[0]), int(nodes.loc[nodes["keyword"]==t,"cluster"].values[0])
        data_edges.append({"source": s, "target": t, "weight": wgt, "tooltip": f"{s} ↔ {t} (Weight = {wgt}, Clusters {c1}–{c2})"})

    data_nodes = [
        {"id": n, "x": float(pos[n][0]), "y": float(pos[n][1]),
         "cluster": int(clusters.get(n,0)),
         "freq": int(nodes.loc[nodes["keyword"]==n,"frequency"].values[0])}
        for n in G.nodes()
    ]

    result = {
        "aac": aac,
        "meanX": float(nodes["frequency"].mean()),
        "meanY": float(edges["Weight"].mean()),
        "nodes": data_nodes,
        "edges": data_edges,
        "download": {"keywords": kw_file.name, "clusters": cl_file.name}
    }
    return JSONResponse(content=result)

# -------------------- GPT keyword extraction --------------------
def extract_keywords(txt):
    if not txt or pd.isna(txt): return ""
    prompt = (
      "請根據以下摘要內容，萃取 10 個具語義代表性的學術關鍵詞或片語，"
      "可為繁體中文或英文（依原文語言自動判斷）。"
      "使用逗號（,）分隔結果：\n\n"+txt)
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2)
        return normalize_commas(res.choices[0].message.content.strip())
    except Exception as e:
        return f"Error: {e}"

def normalize_commas(t):
    return re.sub(r"[、;；\|／/，\s]+", ", ", t).strip(" ,")

# -------------------- Co-occurrence builder --------------------
def build_cooccurrence(df):
    edges,nodes=[],[]
    for _,r in df.iterrows():
        kws=[k.strip() for k in str(r["keywords"]).split(",") if k.strip()]
        nodes+=kws
        for a,b in itertools.combinations(sorted(set(kws)),2):
            edges.append((a,b))
    e=pd.DataFrame(edges,columns=["Source","Target"])
    e["Weight"]=1
    e=e.groupby(["Source","Target"],as_index=False)["Weight"].sum()
    n=pd.DataFrame(pd.Series(nodes).value_counts()).reset_index()
    n.columns=["keyword","frequency"]
    return e,n

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT",10000)))
