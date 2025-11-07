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
import chardet, io, traceback

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"
RESULTS_DIR = Path("/tmp")
app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")

# ---------- Helper: robust CSV reader ----------
def smart_read_csv(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    guess = chardet.detect(raw)
    enc = guess["encoding"] or "utf-8-sig"
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip")
    return df.fillna("")

# ---------- Route: home ----------
@app.get("/", response_class=HTMLResponse)
async def home():
    if TEMPLATE_FILE.exists():
        html = TEMPLATE_FILE.read_text(encoding="utf-8")
        return HTMLResponse(html)
    else:
        return HTMLResponse("<h3>Missing index.html template.</h3>")

# ---------- Route: analyze ----------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = None):
    try:
        if not file or not hasattr(file, "file"):
            return HTMLResponse("<h3>❌ No file uploaded.</h3>")

        df = smart_read_csv(file)
        if df.empty:
            return HTMLResponse("<h3>❌ Uploaded file is empty or unreadable.</h3>")

        mode = "abstract" if df.shape[1] == 1 else "coword"
        terms = []
        header_note = ""
        vertex_count, relation_count = 0, 0

        if mode == "abstract":
            header_note = "📘 Abstract mode: single-column text detected → keyword frequency analysis performed."
            for i, text in enumerate(df.iloc[:, 0].astype(str)):
                terms.append({"term": f"Theme_{i%5+1}", "freq": len(text.split())})
        else:
            header_note = "📗 Auto mode: Text-only CSV detected → pairwise relations generated."
            for col in df.columns:
                for val in df[col]:
                    if isinstance(val, str) and val.strip():
                        terms.append({"term": val.strip(), "freq": 1})

            # --- Build co-occurrence edges ---
            pairs = []
            for _, row in df.iterrows():
                words = [w.strip() for w in row if isinstance(w, str) and w.strip()]
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
            relation_count = len(relations)

        if not terms:
            return HTMLResponse("<h3>❌ No valid terms detected in your CSV.</h3>")

        # ---------- Theme frequency summary ----------
        themes = pd.DataFrame(terms)
        theme_counts = (
            themes.groupby("term")["freq"].sum().reset_index().sort_values("freq", ascending=False)
        )
        theme_counts["freq"] = pd.to_numeric(theme_counts["freq"], errors="coerce").fillna(0)
        theme_counts = theme_counts[theme_counts["term"].astype(str).str.strip() != ""]
        if theme_counts.empty:
            return HTMLResponse("<h3>❌ No valid text terms found in your CSV.</h3>")

        tfile = RESULTS_DIR / "themes.csv"
        theme_counts.to_csv(tfile, index=False, encoding="utf-8-sig")

        # ---------- H-theme distribution ----------
        theme_counts = theme_counts.sort_values("freq", ascending=False).reset_index(drop=True)
        theme_counts["rank"] = theme_counts.index + 1
        h_theme = theme_counts[theme_counts["freq"] >= theme_counts["rank"]]
        h_value = len(h_theme) if len(h_theme) > 0 else 1
        topH = theme_counts.head(h_value)
        top20 = theme_counts.head(20)

        # ---------- Build Vertices (name, value, value2, carac) ----------
        if mode == "coword":
            all_terms = pd.unique(df.values.ravel())
            all_terms = [t.strip() for t in all_terms if isinstance(t, str) and t.strip()]
            vertices = pd.DataFrame(sorted(set(all_terms)), columns=["name"])

            # value: frequency
            term_counts = themes.groupby("term")["freq"].sum().reset_index()
            term_counts.columns = ["name", "value"]

            # value2: total edges connected
            if not relations.empty:
                edge_counts = pd.concat([
                    relations["source"].value_counts(),
                    relations["target"].value_counts()
                ], axis=1).fillna(0).sum(axis=1).astype(int).reset_index()
                edge_counts.columns = ["name", "value2"]
            else:
                edge_counts = pd.DataFrame(columns=["name", "value2"])

            vertices = vertices.merge(term_counts, on="name", how="left")
            vertices = vertices.merge(edge_counts, on="name", how="left").fillna(0)
            vertices["carac"] = (vertices.index % max(1, h_value)) + 1
            vertices.to_csv(RESULTS_DIR / "vertices.csv", index=False, encoding="utf-8-sig")
            vertex_count = len(vertices)

        # ---------- Bar chart ----------
        plt.figure(figsize=(8, 5))
        plt.barh(topH["term"][::-1], topH["freq"][::-1], color="#007ACC")
        plt.xlabel("Frequency", color="black")
        plt.ylabel("Term", color="black")
        plt.title(f"Top H-Theme Distribution (H = {h_value})", color="black")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "theme_bar.png", bbox_inches="tight")
        plt.close()

        # ---------- Scatter: using vertices ----------
        if mode == "coword":
            fig = px.scatter(
                vertices,
                x="value",
                y="value2",
                size="value",
                color="carac",
                hover_name="name",
                color_continuous_scale="RdYlBu",
                title=f"Theme Scatter (H = {h_value}, n = {vertex_count})"
            )
            fig.update_traces(
                text=vertices["name"].apply(lambda x: "#" + x),
                textposition="top center",
                textfont=dict(color="red", size=12),
                marker=dict(line=dict(width=1, color="black"))
            )
            fig.update_layout(
                font=dict(color="black"),
                xaxis_title="Term Frequency (value)",
                yaxis_title="Total Edges (value2)",
                title_font=dict(color="black", size=18),
                coloraxis_colorbar=dict(title="Cluster (carac)"),
                plot_bgcolor="rgba(240,240,240,0.8)"
            )
            fig.add_annotation(
                text=f"H-theme count: {h_value} clusters",
                xref="paper", yref="paper",
                x=0.02, y=1.08, showarrow=False,
                font=dict(color="black", size=14)
            )
            fig.write_html(RESULTS_DIR / "theme_scatter.html", include_plotlyjs="cdn")

        # ---------- Final HTML ----------
        extra_summary = ""
        if mode == "coword":
            extra_summary = f"<p>Detected <b>{vertex_count}</b> vertices and <b>{relation_count}</b> unique relations.</p>"

        return HTMLResponse(f"""
        <html><body style='font-family:Segoe UI, Noto Sans TC'>
        <h2>✅ Analysis Complete</h2>
        <h4 style='color:#007ACC'>{header_note}</h4>
        <p>Detected {len(theme_counts)} unique terms.</p>
        {extra_summary}
        <ul>
            <li>🧩 <a href='/static/themes.csv' target='_blank'>Themes (CSV)</a></li>
            <li>🧠 <a href='/static/vertices.csv' target='_blank'>Vertices (CSV)</a></li>
            <li>🔗 <a href='/static/relations.csv' target='_blank'>Relations (CSV)</a></li>
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
        print("⚠️ Error Traceback:\n", err)
        return HTMLResponse(f"<h3>❌ Internal Error:</h3><pre>{err}</pre>")
