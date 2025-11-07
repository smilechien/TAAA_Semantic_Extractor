from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import io, os, chardet, pandas as pd, matplotlib.pyplot as plt, networkx as nx, plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# 🚀 App initialization
# ------------------------------------------------------------
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ------------------------------------------------------------
# 🧩 Helper functions
# ------------------------------------------------------------
def smart_read_csv(file_bytes):
    """Auto-detect encoding and safely read CSV."""
    try:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        det = chardet.detect(file_bytes)
        enc = det.get("encoding", "utf-8")
        print(f"⚠️ Using fallback encoding: {enc}")
        return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, errors="replace")

def clean_text(x):
    if pd.isna(x): return ""
    return str(x).replace("\x00", " ").replace("\n", " ").strip()

# ------------------------------------------------------------
# 🏠 Home route
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = TEMPLATE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html not found under templates/.</h3>")

# ------------------------------------------------------------
# 📈 Analysis route
# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile):
    contents = await file.read()
    df = smart_read_csv(contents)

    if df.empty:
        return HTMLResponse("<h3>❌ CSV file is empty or unreadable.</h3>")

    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(how="all")

    # Auto-detect mode
    mode = "abstract" if len(df.columns) == 1 else "coword"

    # --------------------------------------------------------
    # 🧠 Abstract Mode
    # --------------------------------------------------------
    if mode == "abstract":
        text_col = df.columns[0]
        df[text_col] = df[text_col].map(clean_text)
        df = df[df[text_col] != ""]
        if df.empty:
            return HTMLResponse("<h3>❌ No valid abstract text found.</h3>")

        corpus = df[text_col].tolist()
        tfidf = TfidfVectorizer(max_features=200)
        X = tfidf.fit_transform(corpus)
        km = KMeans(n_clusters=min(5, len(corpus)), random_state=42, n_init="auto")
        df["theme"] = km.fit_predict(X)
        df["theme"] = df["theme"].apply(lambda x: f"#Theme{x+1}")

        # --- Theme Bar ---
        theme_counts = df["theme"].value_counts().reset_index()
        theme_counts.columns = ["theme", "count"]
        plt.figure(figsize=(8, 4))
        plt.barh(theme_counts["theme"], theme_counts["count"], color="#0078d4")
        plt.title("Core Themes (H-Theme Bar)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "theme_bar.png", bbox_inches="tight")
        plt.close()

        # --- Theme Scatter (Plotly Interactive) ---
        coords = X.toarray()[:, :2]
        df["x"], df["y"] = coords[:, 0], coords[:, 1]
        scatter_fig = px.scatter(
            df.head(20), x="x", y="y", text="theme", color="theme",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Top 20 Themes (Interactive Scatter)"
        )
        scatter_fig.update_traces(textfont=dict(color="red", size=14))
        scatter_fig.write_html(RESULTS_DIR / "theme_scatter.html")

        # --- Save outputs ---
        df.to_csv(RESULTS_DIR / "themes.csv", index=False, encoding="utf-8-sig")

        html = f"""
        <meta charset='utf-8'>
        <h2>🎉 Analysis Complete (Abstract Mode)</h2>
        <p>Detected {len(theme_counts)} major themes.</p>
        <a href="/results/themes.csv" download>🧩 Themes</a><br>
        <a href="/results/theme_bar.png" download>📊 Theme Bar</a><br>
        <a href="/results/theme_scatter.html" target="_blank">🌈 Theme Scatter (Interactive)</a>
        <hr>
        <div style='text-align:center;margin-top:25px;'>
          <a href="/" style="
            display:inline-block;background:#0078d4;color:white;
            padding:10px 22px;border-radius:8px;text-decoration:none;
            font-size:15px;font-weight:500;transition:background 0.3s ease;">
            ⬅️ Return to Home
          </a>
        </div>
        """
        return HTMLResponse(html)

    # --------------------------------------------------------
    # 🐾 Co-Word Mode
    # --------------------------------------------------------
    else:
        df = df.applymap(clean_text)
        terms = df.columns
        G = nx.Graph()

        for _, row in df.iterrows():
            row_terms = [t for t in terms if str(row[t]).strip() != ""]
            for i, a in enumerate(row_terms):
                for b in row_terms[i + 1:]:
                    G.add_edge(a, b)

        if len(G.nodes()) == 0:
            return HTMLResponse("<h3>❌ No valid relations found in co-word data.</h3>")

        degree_df = (
            pd.DataFrame(G.degree(), columns=["term", "degree"])
            .sort_values("degree", ascending=False)
            .head(20)
        )

        # --- Theme Bar ---
        plt.figure(figsize=(8, 5))
        plt.barh(degree_df["term"], degree_df["degree"], color="#1f77b4")
        plt.title("Top 20 Core Themes (H-Theme Bar)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "theme_bar.png", bbox_inches="tight")
        plt.close()

        # --- Scatter (Interactive) ---
        pos = nx.spring_layout(G, seed=42)
        scatter_df = pd.DataFrame(pos).reset_index()
        scatter_df.columns = ["term", "x", "y"]
        scatter_df = scatter_df.merge(degree_df, on="term", how="inner")
        scatter_fig = px.scatter(
            scatter_df, x="x", y="y",
            text=scatter_df["term"].apply(lambda t: f"#{t}"),
            color="degree", color_continuous_scale="Bluered",
            title="Top 20 Themes (Interactive Scatter)"
        )
        scatter_fig.update_traces(textfont=dict(color="red", size=14))
        scatter_fig.write_html(RESULTS_DIR / "theme_scatter.html")

        # --- Save CSV Outputs ---
        degree_df.to_csv(RESULTS_DIR / "themes.csv", index=False, encoding="utf-8-sig")
        nx.write_weighted_edgelist(G, RESULTS_DIR / "relations.csv")
        pd.DataFrame(G.nodes(), columns=["term"]).to_csv(
            RESULTS_DIR / "vertices.csv", index=False, encoding="utf-8-sig"
        )

        html = f"""
        <meta charset='utf-8'>
        <h2>🎉 Analysis Complete (Co-Word Mode)</h2>
        <p>Detected {len(G.nodes())} terms.</p>
        <a href="/results/vertices.csv" download>🔹 Vertices</a><br>
        <a href="/results/relations.csv" download>🔸 Relations</a><br>
        <a href="/results/themes.csv" download>🧩 Themes</a><br>
        <a href="/results/theme_bar.png" download>📊 Theme Bar</a><br>
        <a href="/results/theme_scatter.html" target="_blank">🌈 Theme Scatter (Interactive)</a>
        <hr>
        <div style='text-align:center;margin-top:25px;'>
          <a href="/" style="
            display:inline-block;background:#0078d4;color:white;
            padding:10px 22px;border-radius:8px;text-decoration:none;
            font-size:15px;font-weight:500;transition:background 0.3s ease;">
            ⬅️ Return to Home
          </a>
        </div>
        """
        return HTMLResponse(html)

# ------------------------------------------------------------
# 📦 File Downloader
# ------------------------------------------------------------
@app.get("/results/{filename}")
async def download_file(filename: str):
    file_path = RESULTS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return HTMLResponse("<h3>❌ File not found.</h3>")
