import os, io, re, time, base64, requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import plotly.express as px

# ============================================================
# ✅ Config
# ============================================================
app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ============================================================
# ✅ Safe CSV Reader
# ============================================================
def safe_read_csv(uploaded):
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            uploaded.file.seek(0)
            return pd.read_csv(uploaded.file, encoding=enc)
        except Exception:
            continue
    raise ValueError("Cannot decode file (UTF-8/Big5 failed).")

# ============================================================
# ✅ DOI → Abstract → Keywords
# ============================================================
def get_abstracts_from_dois(df):
    dois = df.iloc[:, 0].dropna().astype(str)
    dois = [d for d in dois if re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", d)]
    if not dois:
        return df

    print(f"📖 Fetching abstracts for {len(dois)} DOIs")
    abstracts = []
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    use_gpt = bool(api_key)

    for doi in dois:
        encoded = requests.utils.quote(doi, safe="")
        url = f"https://api.openalex.org/works/https://doi.org/{encoded}"
        abstract_text = ""
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "abstract_inverted_index" in data:
                    inv = data["abstract_inverted_index"]
                    pairs = sorted([(i, w) for w, pos in inv.items() for i in pos])
                    abstract_text = " ".join([w for _, w in pairs])
                elif "title" in data:
                    abstract_text = data["title"]
        except Exception as e:
            abstract_text = f"Error fetching DOI: {e}"

        if not abstract_text:
            abstracts.append("No abstract found")
            continue

        # GPT-based or TF-IDF-based term extraction
        if use_gpt:
            try:
                client = OpenAI(api_key=api_key)
                msg = [{"role": "user", "content":
                        f"Extract 10 concise keywords from this research abstract:\n\n{abstract_text}\n\nReturn comma-separated terms only."}]
                resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg)
                sem_terms = resp.choices[0].message.content.strip()
            except Exception:
                sem_terms = abstract_text
        else:
            vect = TfidfVectorizer(stop_words="english", max_features=10)
            vect.fit([abstract_text])
            sem_terms = ", ".join(vect.get_feature_names_out())

        abstracts.append(sem_terms)
        time.sleep(1)

    return pd.DataFrame({"abstract": abstracts})

# ============================================================
# ✅ Analysis Endpoint
# ============================================================
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = safe_read_csv(file)
        df = df.dropna(how="all")
        if df.empty:
            return HTMLResponse("<h3>❌ Empty CSV file.</h3>")

        mode = "abstract" if df.shape[1] <= 1 else "coword"
        print(f"🔍 Detected mode: {mode}")

        # DOI extraction
        if mode == "abstract" and df.iloc[:, 0].astype(str).str.contains("10\\.").any():
            df = get_abstracts_from_dois(df)

        # ----------------------------------------------------
        # Build vertices
        # ----------------------------------------------------
        if mode == "coword":
            terms = pd.unique(df.values.ravel())
            terms = [t for t in terms if isinstance(t, str) and t.strip()]
            vertices = pd.DataFrame({"name": terms})
            vertices["value"] = vertices["name"].map(lambda x: (df == x).sum().sum())
        else:
            all_terms = []
            for t in df.iloc[:, 0].dropna():
                all_terms.extend([w.strip() for w in str(t).split(",") if w.strip()])
            vertices = pd.DataFrame(pd.Series(all_terms).value_counts().reset_index())
            vertices.columns = ["name", "value"]

        vertices["value2"] = vertices["value"].rank(ascending=False, method="min")
        vertices["carac"] = vertices["value"].rank(ascending=False).astype(int)

        # ----------------------------------------------------
        # Build relations (edges)
        # ----------------------------------------------------
        edges = []
        for _, row in df.iterrows():
            words = [str(x).strip() for x in row if isinstance(x, str) and x.strip()]
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    edges.append((words[i], words[j]))
        relations = pd.DataFrame(edges, columns=["source", "target"]).value_counts().reset_index(name="weight")

        valid_names = set(vertices["name"])
        relations = relations[
            relations["source"].isin(valid_names) & relations["target"].isin(valid_names)
        ]

        # ----------------------------------------------------
        # Build graph & clusters
        # ----------------------------------------------------
        G = nx.Graph()
        for _, r in relations.iterrows():
            G.add_edge(r["source"], r["target"], weight=r["weight"])

        if len(G.nodes) == 0:
            return HTMLResponse("<h3>❌ No valid term pairs found for clustering.</h3>")

        communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
        cluster_map = {}
        for i, comm in enumerate(communities, start=1):
            for node in comm:
                cluster_map[node] = i
        vertices["carac"] = vertices["name"].map(cluster_map).fillna(0).astype(int)

        # ----------------------------------------------------
        # Build theme table (safe)
        # ----------------------------------------------------
        theme_records = []
        for cid, comm in enumerate(communities, start=1):
            sub = vertices[vertices["name"].isin(comm)]
            if sub.empty:
                continue
            leader = sub.sort_values("value", ascending=False).iloc[0]["name"]
            members = ", ".join(sub["name"].tolist())
            theme_records.append({
                "carac": cid,
                "cluster_label": leader,
                "member_count": len(comm),
                "semantic_label": leader,
                "members": members
            })
        theme = pd.DataFrame(theme_records)
        if theme.empty:
            return HTMLResponse("<h3>⚠️ No clusters found — please upload richer data.</h3>")

        # ----------------------------------------------------
        # Save outputs
        # ----------------------------------------------------
        vpath = os.path.join(STATIC_DIR, "vertices.csv")
        rpath = os.path.join(STATIC_DIR, "relations.csv")
        tpath = os.path.join(STATIC_DIR, "theme.csv")
        vertices.to_csv(vpath, index=False, encoding="utf-8-sig")
        relations.to_csv(rpath, index=False, encoding="utf-8-sig")
        theme.to_csv(tpath, index=False, encoding="utf-8-sig")

        # ----------------------------------------------------
        # H-theme filtering
        # ----------------------------------------------------
        freq = theme["member_count"].sort_values(ascending=False).reset_index(drop=True)
        h = sum(freq >= freq.index + 1)
        h_themes = theme.nlargest(h, "member_count")

        plt.figure(figsize=(8, 5))
        plt.barh(h_themes["semantic_label"], h_themes["member_count"], color="skyblue")
        plt.gca().invert_yaxis()
        plt.title(f"H-Theme Distribution (h={h})")
        plt.tight_layout()
        hbar_path = os.path.join(STATIC_DIR, "h_theme_bar.png")
        plt.savefig(hbar_path, dpi=150)
        plt.close()

        # ----------------------------------------------------
        # Theme scatter (categorical, higher layout)
        # ----------------------------------------------------
        top20 = vertices.nlargest(20, "value")
        fig = px.scatter(
            top20,
            x="value", y="value2",
            color=top20["carac"].astype(str),
            text="name",
            title="Theme Scatter (Top 20 Terms, by Category)",
            height=700
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey"))
        )
        fig.update_layout(
            font=dict(size=14),
            margin=dict(l=80, r=80, t=100, b=80),
            legend_title_text="Cluster",
            plot_bgcolor="rgba(255,255,255,0.95)"
        )
        scatter_path = os.path.join(STATIC_DIR, "theme_scatter.html")
        fig.write_html(scatter_path)

        # ----------------------------------------------------
        # TAAA: Article–Theme assignment
        # ----------------------------------------------------
        art_assign = []
        for idx, row in enumerate(df.itertuples(index=False), 1):
            terms = [t for t in row if isinstance(t, str) and t.strip()]
            assigned_clusters = [cluster_map.get(t, 0) for t in terms]
            if not assigned_clusters or all(c == 0 for c in assigned_clusters):
                theme_label = "Unassigned"
            else:
                mode_cluster = max(set(assigned_clusters), key=assigned_clusters.count)
                label_match = theme.loc[theme["carac"] == mode_cluster, "semantic_label"]
                theme_label = label_match.iloc[0] if not label_match.empty else "Unassigned"
            art_assign.append({"article_id": idx, "theme": theme_label})
        assign_df = pd.DataFrame(art_assign)
        apath = os.path.join(STATIC_DIR, "article_theme_assign.csv")
        assign_df.to_csv(apath, index=False, encoding="utf-8-sig")

        # ----------------------------------------------------
        # Output HTML
        # ----------------------------------------------------
        html = f"""
        <h2>✅ Analysis Completed (Mode: {mode})</h2>
        <p>Detected {len(vertices)} vertices and {len(relations)} unique relations.</p>
        <ul>
          <li><a href="/static/vertices.csv" target="_blank">Vertices CSV</a></li>
          <li><a href="/static/relations.csv" target="_blank">Relations CSV</a></li>
          <li><a href="/static/theme.csv" target="_blank">Theme CSV</a></li>
          <li><a href="/static/article_theme_assign.csv" target="_blank">Article–Theme Assignment CSV</a></li>
          <li><a href="/static/h_theme_bar.png" target="_blank">H-Theme Bar (PNG)</a></li>
          <li><a href="/static/theme_scatter.html" target="_blank">Theme Scatter (Interactive)</a></li>
        </ul>
        """
        return HTMLResponse(html)

    except Exception as e:
        return HTMLResponse(f"<h3>Internal Error:<br>{e}</h3>")

# ============================================================
# ✅ Home
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
