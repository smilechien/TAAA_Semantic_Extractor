# ============================================================
# 🌐 TAAA Semantic–Co-Word Analyzer v9.0
# Auto-detects abstract vs co-word CSVs, fixes encoding issues,
# performs jieba+TF-IDF or co-word correlation,
# and outputs bilingual CSVs + scatter/bar charts
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import io, chardet, matplotlib.pyplot as plt, matplotlib
import jieba, math
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
import networkx as nx

matplotlib.use("Agg")  # Headless backend for Render
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------------------------------------------
# 🏠 Home Route
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = TEMPLATE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>❌ index.html not found under app/templates/.</h3>")


# ------------------------------------------------------------
# 👀 Preview Route — returns first 5 rows decoded safely
# ------------------------------------------------------------
@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    content = await file.read()
    enc = chardet.detect(content).get("encoding") or "utf-8"
    try:
        decoded = content.decode(enc, errors="ignore")
        df = pd.read_csv(io.StringIO(decoded))
        df = df.dropna(how="all").head(5)
        return JSONResponse({
            "encoding": enc,
            "columns": list(df.columns),
            "rows": df.fillna("").values.tolist()
        })
    except Exception as e:
        return JSONResponse({"error": f"Preview failed: {str(e)}"})


# ------------------------------------------------------------
# 🚀 Analyze Route — full semantic/co-word analysis
# ------------------------------------------------------------
@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        enc = chardet.detect(content).get("encoding") or "utf-8"
        decoded = content.decode(enc, errors="ignore")
        data = pd.read_csv(io.StringIO(decoded)).dropna(how="all")
        if data.empty:
            return HTMLResponse("<h3>❌ Uploaded file is empty.</h3>")

        # Auto detect mode
        mode = "abstract" if data.shape[1] <= 1 else "coword"

        # Output paths
        out_vertices = STATIC_DIR / "output_vertices.csv"
        out_relations = STATIC_DIR / "output_relations.csv"
        out_themes = STATIC_DIR / "output_themes.csv"
        bar_path = STATIC_DIR / "theme_bar.png"
        scatter_path = STATIC_DIR / "theme_scatter.png"

        # --------------------------------------------------------
        # 🧠 ABSTRACT MODE
        # --------------------------------------------------------
        if mode == "abstract":
            text_col = data.columns[0]
            texts = data[text_col].astype(str).tolist()

            # Language detection (first non-empty)
            lang = "zh"
            for t in texts:
                try:
                    lang = detect(t)
                    break
                except:
                    continue

            # Jieba for Chinese, whitespace for others
            if lang.startswith("zh"):
                tokenized = [" ".join(jieba.cut(t)) for t in texts]
            else:
                tokenized = texts

            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(tokenized)
            terms = vectorizer.get_feature_names_out()
            weights = X.sum(axis=0).A1
            vertices = pd.DataFrame({"term": terms, "score": weights}).sort_values(
                "score", ascending=False
            )
            vertices.to_csv(out_vertices, index=False, encoding="utf-8-sig")

            # Mock relations via co-occurrence
            relations = []
            for i in range(min(10, len(terms))):
                for j in range(i + 1, min(10, len(terms))):
                    relations.append({"source": terms[i], "target": terms[j], "weight": 1})
            pd.DataFrame(relations).to_csv(out_relations, index=False, encoding="utf-8-sig")

            # Themes
            theme_df = pd.DataFrame({
                "theme": [f"Theme_{i%5+1}" for i in range(len(vertices))],
                "term": vertices["term"],
                "score": vertices["score"]
            })
            theme_df.to_csv(out_themes, index=False, encoding="utf-8-sig")

            # Visualization: bar + scatter
            top20 = vertices.head(20)
            plt.figure(figsize=(8, 4))
            plt.barh(top20["term"][::-1], top20["score"][::-1], color="#2ecc71")
            plt.title("Top 20 Semantic Terms (TF-IDF)")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.scatter(range(len(top20)), top20["score"], c="#27ae60")
            for i, t in enumerate(top20["term"]):
                plt.text(i, top20["score"].iloc[i], t, fontsize=8, rotation=45)
            plt.title("Theme Scatter (Abstract Mode)")
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150)
            plt.close()

        # --------------------------------------------------------
        # 🐾 CO-WORD MODE
        # --------------------------------------------------------
        else:
            df = data.copy()
            num_df = df.select_dtypes(include=["number"])
            if num_df.empty:
                # Try to encode categorical columns numerically
                num_df = df.apply(lambda col: pd.factorize(col)[0])

            corr = num_df.corr().fillna(0)
            terms = corr.columns.tolist()

            # Vertices
            vertices = pd.DataFrame({"term": terms, "degree": corr.abs().sum()})
            vertices = vertices.sort_values("degree", ascending=False).head(20)
            vertices.to_csv(out_vertices, index=False, encoding="utf-8-sig")

            # Relations
            edges = []
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    w = corr.iat[i, j]
                    if abs(w) > 0.3:
                        edges.append({"source": terms[i], "target": terms[j], "weight": round(w, 3)})
            pd.DataFrame(edges).to_csv(out_relations, index=False, encoding="utf-8-sig")

            # Themes via network clustering
            G = nx.Graph()
            for e in edges:
                G.add_edge(e["source"], e["target"], weight=e["weight"])
            clusters = list(nx.connected_components(G))
            theme_list = []
            for idx, cset in enumerate(clusters):
                for term in cset:
                    theme_list.append({"theme": f"Cluster_{idx+1}", "term": term})
            theme_df = pd.DataFrame(theme_list)
            theme_df.to_csv(out_themes, index=False, encoding="utf-8-sig")

            # Bar plot
            plt.figure(figsize=(8, 4))
            plt.barh(vertices["term"][::-1], vertices["degree"][::-1], color="#3498db")
            plt.title("Top 20 Co-Word Terms (Degree)")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()

            # Scatter plot
            plt.figure(figsize=(5, 5))
            plt.scatter(range(len(vertices)), vertices["degree"], c="#2980b9")
            for i, t in enumerate(vertices["term"]):
                plt.text(i, vertices["degree"].iloc[i], t, fontsize=8, rotation=45)
            plt.title("Theme Scatter (Co-Word Mode)")
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150)
            plt.close()

        # --------------------------------------------------------
        # ✅ Return bilingual results
        # --------------------------------------------------------
        mode_en = "🧠 Abstract Mode" if mode == "abstract" else "🐾 Co-Word Mode"
        mode_zh = "🧠 摘要模式" if mode == "abstract" else "🐾 共詞模式"

        html = f"""
        <html><head><meta charset='UTF-8'><title>TAAA Results</title>
        <style>
            body {{font-family:'Segoe UI','Noto Sans TC',sans-serif;text-align:center;background:#fafafa;}}
            .box {{background:#fff;padding:30px;margin:40px auto;border-radius:14px;
                   box-shadow:0 2px 8px rgba(0,0,0,0.1);width:600px;}}
            a {{text-decoration:none;color:#0077cc;}}
            .btn {{display:inline-block;margin:10px;padding:10px 18px;border-radius:8px;background:#f0f0f0;}}
            .btn:hover {{background:#e0e0e0;}}
        </style></head>
        <body><div class='box'>
        <h2>{mode_en} / {mode_zh}</h2>
        <p>✅ Analysis complete.<br>分析完成。</p>
        <h3>📤 Download Results / 下載結果</h3>
        <a class='btn' href='/static/output_vertices.csv' download>🔹 Vertices</a>
        <a class='btn' href='/static/output_relations.csv' download>🔸 Relations</a>
        <a class='btn' href='/static/output_themes.csv' download>🧩 Themes</a><br>
        <a class='btn' href='/static/theme_bar.png' download>📊 Theme Bar</a>
        <a class='btn' href='/static/theme_scatter.png' download>🌈 Theme Scatter</a><br><br>
        <a href='/' style='font-size:14px;'>🏠 Return Home / 返回首頁</a>
        </div></body></html>
        """
        return HTMLResponse(html)

    except Exception as e:
        return HTMLResponse(f"<h3>❌ Processing error: {str(e)}</h3>")
