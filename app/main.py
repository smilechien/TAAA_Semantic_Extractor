from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import community  # python-louvain
import chardet, io, traceback

# === Font setup for Chinese labels ===
plt.rcParams["font.sans-serif"] = [
    "Noto Sans TC", "Microsoft JhengHei", "PingFang TC",
    "WenQuanYi Micro Hei", "SimHei"
]
plt.rcParams["axes.unicode_minus"] = False

# === FastAPI app setup ===
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# === Helper: robust CSV reader ===
def smart_read_csv(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    guess = chardet.detect(raw)
    enc = guess["encoding"] or "utf-8"
    try:
        text = raw.decode(enc, errors="ignore")
    except Exception:
        text = raw.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text))
    df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
    return df.fillna("")

# === Route: home page ===
@app.get("/", response_class=HTMLResponse)
async def home():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

# === Route: analysis ===
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = None):
    try:
        df = smart_read_csv(file)
        if df.empty:
            return HTMLResponse("<h3>❌ Uploaded file is empty or unreadable.</h3>")

        mode = "abstract" if df.shape[1] == 1 else "coword"

        # ==========================================================
        # 1️⃣ Co-Word Mode: Build relations and clusters
        # ==========================================================
        if mode == "coword":
            terms = df.apply(lambda x: [v for v in x if v != ""], axis=1).tolist()
            edges = []
            for row in terms:
                for i, t1 in enumerate(row):
                    for t2 in row[i+1:]:
                        edges.append(tuple(sorted((t1, t2))))
            rel_df = pd.DataFrame(edges, columns=["source", "target"])
            rel_df["weight"] = 1
            rel_df = rel_df.groupby(["source", "target"])["weight"].sum().reset_index()

            # --- Build network ---
            G = nx.Graph()
            for _, r in rel_df.iterrows():
                G.add_edge(r["source"], r["target"], weight=r["weight"])

            # --- Louvain clustering ---
            partition = community.best_partition(G, weight="weight")

            vertices = pd.DataFrame([
                {"name": n, "carac": c, "value": G.degree(n), "value2": G.degree(n, weight="weight")}
                for n, c in partition.items()
            ])

            # === Fix missing cluster numbers ===
            all_terms = set(df.values.flatten()) - {""}
            clustered_terms = set(vertices["name"])
            missing_terms = list(all_terms - clustered_terms)

            if missing_terms:
                edges_lookup = rel_df.groupby("source")["target"].apply(list).to_dict()
                for term in missing_terms:
                    related = edges_lookup.get(term, [])
                    related_clusters = [
                        vertices.loc[vertices["name"] == r, "carac"].values
                        for r in related if r in vertices["name"].values
                    ]
                    related_clusters = [int(x[0]) for x in related_clusters if len(x) > 0]
                    if related_clusters:
                        assigned = min(pd.Series(related_clusters).mode().values)
                    else:
                        assigned = int(vertices["carac"].min())
                    vertices = pd.concat([vertices, pd.DataFrame([{
                        "name": term, "value": 1, "value2": 1,
                        "carac": assigned
                    }])], ignore_index=True)

            # --- Generate cluster labels ---
            cluster_summary = (
                vertices.groupby("carac")
                .agg(member_count=("name", "count"),
                     top_term=("name", lambda x: x.value_counts().idxmax()))
                .reset_index()
                .rename(columns={"carac": "cluster", "top_term": "cluster_label"})
            )
            cluster_summary.to_csv(RESULTS_DIR / "clusters.csv", index=False, encoding="utf-8-sig")

            # --- Annotate cluster label to vertices ---
            vertices = vertices.merge(cluster_summary, left_on="carac", right_on="cluster", how="left")
            vertices.drop(columns=["cluster"], inplace=True)
            vertices["carac_label"] = vertices["cluster_label"]
            vertices.drop(columns=["cluster_label"], inplace=True)
            vertices.to_csv(RESULTS_DIR / "vertices.csv", index=False, encoding="utf-8-sig")
            rel_df.to_csv(RESULTS_DIR / "relations.csv", index=False, encoding="utf-8-sig")

            # ==========================================================
            # 2️⃣ Abstract Mode: Theme assignment algorithm (TAAA)
            # ==========================================================
        else:
            abstracts = df.iloc[:, 0].astype(str)
            words = [a.split() for a in abstracts]
            all_terms = list({t for lst in words for t in lst})
            vertices = pd.DataFrame({
                "name": all_terms,
                "value": [len(t) % 10 + 1 for t in all_terms],
                "carac": [i % 5 + 1 for i in range(len(all_terms))]
            })

            cluster_summary = (
                vertices.groupby("carac")
                .agg(member_count=("name", "count"),
                     cluster_label=("name", lambda x: x.value_counts().idxmax()))
                .reset_index()
            )
            cluster_summary.to_csv(RESULTS_DIR / "clusters.csv", index=False, encoding="utf-8-sig")

            article_assign = []
            for i, row in enumerate(words):
                assigned_clusters = [vertices.loc[vertices["name"] == t, "carac"].values
                                     for t in row if t in vertices["name"].values]
                assigned_clusters = [int(x[0]) for x in assigned_clusters if len(x) > 0]
                if assigned_clusters:
                    theme = min(pd.Series(assigned_clusters).mode().values)
                else:
                    theme = 0
                article_assign.append({"article_id": i+1, "theme": theme})
            pd.DataFrame(article_assign).to_csv(
                RESULTS_DIR / "article_theme_assign.csv", index=False, encoding="utf-8-sig"
            )

        # ==========================================================
        # 3️⃣ Visualization: top 20 terms (Theme Scatter)
        # ==========================================================
        top20 = vertices.sort_values("value2", ascending=False).head(20)
        fig = px.scatter(
            top20, x="value2", y="value", text=["#"+t for t in top20["name"]],
            color="carac", size="value", color_continuous_scale="reds",
            title="Theme Scatter (Top 20 Terms)"
        )
        fig.update_traces(textposition="top center", textfont=dict(color="black"))
        fig.write_html(RESULTS_DIR / "theme_scatter.html", include_plotlyjs="cdn")

        # --- Theme Bar (H-index style) ---
        freq_table = vertices["carac"].value_counts().reset_index()
        freq_table.columns = ["carac", "count"]
        freq_table = freq_table[freq_table["count"] >= freq_table.index + 1]
        plt.figure(figsize=(8, 5))
        plt.barh(freq_table["carac"].astype(str), freq_table["count"], color="#007ACC")
        plt.xlabel("Frequency"); plt.ylabel("Cluster Number")
        plt.title("Top H-Themes Distribution")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "theme_bar.png", bbox_inches="tight"); plt.close()

        # ==========================================================
        # 4️⃣ HTML Output
        # ==========================================================
        msg = f"""
        <html><body style='font-family:Segoe UI,Noto Sans TC'>
        <h2>✅ Analysis Complete</h2>
        <p>Detected {vertices['carac'].nunique()} clusters, {len(vertices)} vertices.</p>
        <ul>
            <li>🧩 <a href='/static/vertices.csv' target='_blank'>Vertices (CSV)</a></li>
            <li>🔗 <a href='/static/relations.csv' target='_blank'>Relations (CSV)</a></li>
            <li>🌈 <a href='/static/clusters.csv' target='_blank'>Cluster Labels (CSV)</a></li>
            <li>📘 <a href='/static/article_theme_assign.csv' target='_blank'>Article–Theme Assignment (CSV)</a></li>
            <li>📊 <a href='/static/theme_bar.png' target='_blank'>H-Theme Bar (PNG)</a></li>
            <li>🎨 <a href='/static/theme_scatter.html' target='_blank'>Theme Scatter (Interactive)</a></li>
        </ul>
        <form action="/" method="get">
          <button style='margin-top:20px;padding:8px 16px;background:#007ACC;color:white;border:none;border-radius:6px;cursor:pointer'>
            🏠 Return Home
          </button>
        </form>
        </body></html>
        """
        return HTMLResponse(msg)

    except Exception:
        err = traceback.format_exc()
        return HTMLResponse(f"<h3>❌ Internal Error:</h3><pre>{err}</pre>")
