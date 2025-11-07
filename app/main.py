# ==========================================================
# main.py — TAAA Co-Word / Abstract Analyzer
# Version: v13.0  (GPT Store Ready)
# Author: Smile (Tsair-Wei Chien inspired)
# ==========================================================

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations
import chardet, io, traceback, unicodedata

# ---------- Initialize FastAPI ----------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- Robust CSV Reader ----------
def smart_read_csv(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    guess = chardet.detect(raw)
    enc = guess["encoding"] or "utf-8"
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin1", on_bad_lines="skip")
    return df.fillna("")


# ---------- Text Cleaner ----------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    try:
        s = unicodedata.normalize("NFKC", s)
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        pass
    return s.strip()


# ---------- Home ----------
@app.get("/", response_class=HTMLResponse)
async def home():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---------- Analyzer ----------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = None):
    try:
        df = smart_read_csv(file)
        if df.empty:
            return HTMLResponse("<h3>❌ Uploaded file is empty or unreadable.</h3>")

        mode = "abstract" if df.shape[1] == 1 else "coword"

        # =====================================================
        # 🧩 STEP 1. Generate terms & themes
        # =====================================================
        terms = []
        if mode == "abstract":
            for i, text in enumerate(df.iloc[:, 0].astype(str)):
                for w in text.replace("、", ",").replace("，", ",").split(","):
                    if w.strip():
                        terms.append({"term": clean_text(w.strip()), "freq": 1})
        else:
            for col in df.columns:
                for val in df[col]:
                    if isinstance(val, str) and val.strip():
                        terms.append({"term": clean_text(val), "freq": 1})

        themes = pd.DataFrame(terms)
        theme_counts = (
            themes.groupby("term")["freq"].sum().reset_index().sort_values("freq", ascending=False)
        )
        theme_counts.to_csv(RESULTS_DIR / "themes.csv", index=False, encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 2. Build Vertices / Relations
        # =====================================================
        all_terms = pd.unique(df.values.ravel())
        all_terms = [clean_text(t) for t in all_terms if isinstance(t, str) and t.strip()]
        vertices = pd.DataFrame(sorted(set(all_terms)), columns=["name"])

        # frequency per term
        term_counts = themes.groupby("term")["freq"].sum().reset_index()
        term_counts.columns = ["name", "value"]

        # build co-occurrence edges (for coword mode)
        pairs = []
        if mode == "coword":
            for _, row in df.iterrows():
                words = [clean_text(w) for w in row if isinstance(w, str) and w.strip()]
                for a, b in combinations(sorted(set(words)), 2):
                    pairs.append(tuple(sorted((a, b))))
        if pairs:
            rel = pd.Series(pairs).value_counts().reset_index()
            rel.columns = ["pair", "weight"]
            rel[["source", "target"]] = pd.DataFrame(rel["pair"].tolist(), index=rel.index)
            relations = rel[["source", "target", "weight"]]
        else:
            relations = pd.DataFrame(columns=["source", "target", "weight"])
        relations.to_csv(RESULTS_DIR / "relations.csv", index=False, encoding="utf-8-sig")

        # compute total edges per vertex
        if not relations.empty:
            edge_counts = pd.concat([
                relations["source"].value_counts(),
                relations["target"].value_counts()
            ], axis=1).fillna(0).sum(axis=1).astype(int).reset_index()
            edge_counts.columns = ["name", "value2"]
        else:
            edge_counts = pd.DataFrame(columns=["name", "value2"])

        # merge attributes
        vertices = vertices.merge(term_counts, on="name", how="left")
        vertices = vertices.merge(edge_counts, on="name", how="left").fillna(0)

        # temporary pseudo cluster IDs
        vertices["carac"] = (vertices.index % 9) + 1
        vertices["name"] = vertices["name"].apply(clean_text)

        # =====================================================
        # 🧩 STEP 3. Cluster Representatives & Legend
        # =====================================================
        cluster_representatives = (
            vertices.sort_values("value", ascending=False)
            .groupby("carac").first()["name"].to_dict()
        )
        vertices["carac_label"] = vertices["carac"].map(cluster_representatives)

        cluster_summary = (
            vertices.sort_values(["carac", "value"], ascending=[True, False])
            .groupby("carac")
            .agg(
                cluster_label=("name", "first"),
                member_terms=("name", lambda x: ", ".join(x))
            ).reset_index()
        )
        cluster_summary.to_csv(RESULTS_DIR / "clusters.csv", index=False, encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 4. H-Theme Bar Chart
        # =====================================================
        theme_counts["rank"] = range(1, len(theme_counts) + 1)
        h_theme = theme_counts[theme_counts["freq"] >= theme_counts["rank"]]
        h_value = len(h_theme) if len(h_theme) > 0 else 1

        plt.figure(figsize=(8, 5))
        plt.barh(h_theme["term"][::-1], h_theme["freq"][::-1], color="#007ACC")
        plt.xlabel("Frequency", color="black", fontsize=11)
        plt.ylabel("Term", color="black", fontsize=11)
        plt.title(f"Top H-Theme Distribution (H = {h_value})", color="black", fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "theme_bar.png", bbox_inches="tight")
        plt.close()

        # =====================================================
        # 🧩 STEP 5. Scatter (Top 20)
        # =====================================================
        top20 = vertices.sort_values("value", ascending=False).head(20)
        fig = px.scatter(
            top20,
            x="value", y="value2",
            size="value", color="carac",
            hover_name="name",
            color_continuous_scale="RdYlBu",
            title=f"Theme Scatter (Top 20 Terms, H={h_value})"
        )
        fig.update_traces(
            text=top20["name"].apply(lambda x: "#" + x),
            textposition="top center",
            textfont=dict(color="red", size=12),
            marker=dict(line=dict(width=1, color="black"))
        )
        fig.update_layout(
            font=dict(color="black"),
            xaxis_title="Term Frequency (value)",
            yaxis_title="Total Edges (value2)",
            plot_bgcolor="rgba(240,240,240,0.8)"
        )
        fig.write_html(RESULTS_DIR / "theme_scatter.html", include_plotlyjs="cdn")

        # =====================================================
        # 🧩 STEP 6. TAAA for Abstract Mode
        # =====================================================
        if mode == "abstract":
            term_to_cluster = dict(zip(vertices["name"], vertices["carac"]))
            records = []
            for idx, text in enumerate(df.iloc[:, 0].astype(str), start=1):
                terms = [t.strip() for t in text.replace("、", ",").replace("，", ",").split(",") if t.strip()]
                clusters = [term_to_cluster.get(t, None) for t in terms if t in term_to_cluster]
                clusters = [c for c in clusters if c is not None]
                if not clusters:
                    assigned_cluster = None
                    assigned_label = "Unclassified"
                else:
                    s = pd.Series(clusters).value_counts()
                    topfreq = s.max()
                    tied = s[s == topfreq].index.tolist()
                    assigned_cluster = min(map(int, tied))
                    assigned_label = cluster_representatives.get(assigned_cluster, f"Theme_{assigned_cluster}")
                records.append({
                    "article_id": idx,
                    "abstract": text[:200] + ("..." if len(text) > 200 else ""),
                    "assigned_theme": assigned_cluster,
                    "theme_label": assigned_label
                })
            pd.DataFrame(records).to_csv(RESULTS_DIR / "article_theme_assign.csv", index=False, encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 7. Save Vertices
        # =====================================================
        vertices.to_csv(RESULTS_DIR / "vertices.csv", index=False, encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 8. HTML Output
        # =====================================================
        article_link = "<li>🧬 <a href='/static/article_theme_assign.csv' target='_blank'>Article–Theme Assignment (CSV)</a></li>" if mode == "abstract" else ""
        return HTMLResponse(f"""
        <html><body style='font-family:Segoe UI, Noto Sans TC'>
        <h2>✅ Analysis Complete</h2>
        <p>Mode detected: <b>{mode.upper()}</b> — Auto theme classification finished.</p>
        <ul>
            <li>🧩 <a href='/static/themes.csv' target='_blank'>Themes (CSV)</a></li>
            <li>🧭 <a href='/static/clusters.csv' target='_blank'>Clusters (CSV)</a></li>
            <li>🧠 <a href='/static/vertices.csv' target='_blank'>Vertices (CSV)</a></li>
            <li>🔗 <a href='/static/relations.csv' target='_blank'>Relations (CSV)</a></li>
            {article_link}
            <li>📊 <a href='/static/theme_bar.png' target='_blank'>Theme Bar (PNG)</a></li>
            <li>🌈 <a href='/static/theme_scatter.html' target='_blank'>Theme Scatter (Interactive)</a></li>
        </ul>
        <form action="/" method="get">
            <button style='margin-top:20px;padding:8px 16px;background:#007ACC;color:white;border:none;border-radius:6px;cursor:pointer'>
                🏠 Return Home
            </button>
        </form>
        </body></html>
        """)
    except Exception:
        err = traceback.format_exc()
        return HTMLResponse(f"<h3>❌ Internal Error:</h3><pre>{err}</pre>")
