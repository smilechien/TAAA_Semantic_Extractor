# ==========================================================
# main.py — TAAA Analyzer (Unified Abstract & Co-Word Modes)
# Version: v14.0
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

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Utility ----------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    try:
        s = unicodedata.normalize("NFKC", s)
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        pass
    return s.strip()

def smart_read_csv(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    enc = (chardet.detect(raw)["encoding"] or "utf-8")
    for e in [enc, "utf-8", "utf-8-sig", "big5", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=e)
            if not df.empty: return df.fillna("")
        except Exception:
            continue
    raise ValueError("❌ Cannot decode CSV file.")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.post("/analyze_csv", response_class=HTMLResponse)
async def analyze_csv(file: UploadFile = None):
    try:
        df = smart_read_csv(file)
        if df.empty:
            return HTMLResponse("<h3>❌ Uploaded file is empty or unreadable.</h3>")
        mode = "abstract" if df.shape[1] == 1 else "coword"

        # =====================================================
        # 🧩 STEP 1. Build term list
        # =====================================================
        terms = []
        if mode == "abstract":
            for text in df.iloc[:,0].astype(str):
                for w in text.replace("、",",").replace("，",",").split(","):
                    if w.strip(): terms.append({"term": clean_text(w), "freq":1})
        else:
            for col in df.columns:
                for val in df[col]:
                    if isinstance(val,str) and val.strip():
                        terms.append({"term": clean_text(val), "freq":1})
        themes = pd.DataFrame(terms)
        theme_counts = (themes.groupby("term")["freq"]
                        .sum().reset_index().sort_values("freq",ascending=False))
        theme_counts.to_csv(RESULTS_DIR/"themes_raw.csv",index=False,encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 2. Vertices & Relations
        # =====================================================
        all_terms = pd.unique(df.values.ravel())
        all_terms = [clean_text(t) for t in all_terms if isinstance(t,str) and t.strip()]
        vertices = pd.DataFrame(sorted(set(all_terms)), columns=["name"])
        term_counts = themes.groupby("term")["freq"].sum().reset_index()
        term_counts.columns=["name","value"]
        # edges
        pairs=[]
        if mode=="coword":
            for _,row in df.iterrows():
                words=[clean_text(w) for w in row if isinstance(w,str) and w.strip()]
                for a,b in combinations(sorted(set(words)),2): pairs.append(tuple(sorted((a,b))))
        if pairs:
            rel=pd.Series(pairs).value_counts().reset_index()
            rel.columns=["pair","weight"]
            rel[["source","target"]]=pd.DataFrame(rel["pair"].tolist(),index=rel.index)
            relations=rel[["source","target","weight"]]
        else:
            relations=pd.DataFrame(columns=["source","target","weight"])
        # edge counts
        if not relations.empty:
            edge_counts=pd.concat([relations["source"].value_counts(),
                                   relations["target"].value_counts()],axis=1).fillna(0).sum(axis=1)
            edge_counts=edge_counts.astype(int).reset_index()
            edge_counts.columns=["name","value2"]
        else:
            edge_counts=pd.DataFrame(columns=["name","value2"])
        vertices=vertices.merge(term_counts,on="name",how="left")
        vertices=vertices.merge(edge_counts,on="name",how="left").fillna(0)
        vertices["carac"]=(vertices.index%9)+1  # placeholder cluster id
        vertices["name"]=vertices["name"].apply(clean_text)

        # =====================================================
        # 🧩 STEP 3. Top-20 cluster rule (R-style)
        # =====================================================
        nodes=vertices.copy()
        nodes["freq"]=nodes.groupby("carac")["carac"].transform("count")
        selected=pd.DataFrame(columns=["name","value","carac","freq"])
        def pick(df,cond,per,rem):
            elig=df.query(cond)
            if elig.empty or rem<=0: return pd.DataFrame(columns=["name","value","carac","freq"])
            cl=(elig.groupby("carac").agg(freq=("freq","first"),sum_value=("value","sum"))
                .sort_values(["freq","sum_value"],ascending=[False,False]).reset_index())
            maxcl=max(1,rem//per)
            chosen=cl.head(maxcl)["carac"].tolist()
            res=(elig[elig["carac"].isin(chosen)]
                 .sort_values(["carac","value"],ascending=[True,False])
                 .groupby("carac").head(per))
            return res[["name","value","carac","freq"]]
        cap=20; rem=cap
        lvl1=pick(nodes,"freq >= 4",4,rem); selected=pd.concat([selected,lvl1]); rem=cap-len(selected)
        if rem>0:
            lvl2=pick(nodes,"freq == 3",3,22-len(selected))
            if not lvl2.empty: cap=22; selected=pd.concat([selected,lvl2])
        rem=cap-len(selected)
        if rem>0:
            cap3=max(cap,cap+1)
            lvl3=pick(nodes,"freq == 2",2,cap3-len(selected))
            if not lvl3.empty: cap=max(cap,cap+1); selected=pd.concat([selected,lvl3])
        rem=cap-len(selected)
        if rem>0:
            lvl4=pick(nodes,"freq == 1",1,rem); selected=pd.concat([selected,lvl4])
        selected=(selected.drop_duplicates(subset=["name","carac"])
                  .sort_values("value",ascending=False)
                  .head(cap).reset_index(drop=True))
        vertices=vertices[vertices["name"].isin(selected["name"])]
        relations=relations[relations["source"].isin(vertices["name"]) &
                            relations["target"].isin(vertices["name"])]

        # =====================================================
        # 🧩 STEP 4. Cluster labels & themes.csv
        # =====================================================
        cluster_reps=(vertices.sort_values("value",ascending=False)
                      .groupby("carac").first()["name"].to_dict())
        vertices["carac_label"]=vertices["carac"].map(cluster_reps)
        theme_summary=(vertices.groupby("carac")
                       .agg(cluster_label=("carac_label","first"),
                            member_count=("name","count"))
                       .reset_index())
        theme_summary["semantic_label"]=theme_summary["cluster_label"]
        theme_summary.to_csv(RESULTS_DIR/"themes.csv",index=False,encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 5. Bar & Scatter
        # =====================================================
        theme_counts["rank"]=range(1,len(theme_counts)+1)
        h_theme=theme_counts[theme_counts["freq"]>=theme_counts["rank"]]
        h_val=len(h_theme) if len(h_theme)>0 else 1
        plt.figure(figsize=(8,5))
        plt.barh(h_theme["term"][::-1],h_theme["freq"][::-1],color="#007ACC")
        plt.title(f"Top H-Theme Distribution (H = {h_val})")
        plt.tight_layout(); plt.savefig(RESULTS_DIR/"theme_bar.png"); plt.close()
        top20=vertices.sort_values("value",ascending=False).head(20)
        fig=px.scatter(top20,x="value",y="value2",size="value",color="carac",
                       hover_name="name",color_continuous_scale="RdYlBu",
                       title=f"Theme Scatter (Top 20 Terms, H={h_val})")
        fig.update_traces(text=top20["name"].apply(lambda x:"#"+x),
                          textposition="top center",
                          textfont=dict(color="red",size=12),
                          marker=dict(line=dict(width=1,color="black")))
        fig.write_html(RESULTS_DIR/"theme_scatter.html",include_plotlyjs="cdn")

        # =====================================================
        # 🧩 STEP 6. Theme-to-Article Assignment (TAAA)
        # =====================================================
        term_to_cluster=dict(zip(vertices["name"],vertices["carac"]))
        cluster_to_label=dict(zip(vertices["carac"],vertices["carac_label"]))
        records=[]
        if mode=="abstract":
            for i,text in enumerate(df.iloc[:,0].astype(str),start=1):
                terms=[clean_text(t) for t in text.replace("、",",").replace("，",",").split(",") if t.strip()]
                clusters=[term_to_cluster.get(t) for t in terms if t in term_to_cluster]
                clusters=[c for c in clusters if c is not None]
                if clusters:
                    s=pd.Series(clusters).value_counts()
                    topf=s.max(); tied=s[s==topf].index.tolist()
                    theme=min(map(int,tied))
                    label=cluster_to_label.get(theme,f"Theme_{theme}")
                else:
                    theme=None; label="Unclassified"
                records.append({"article_id":i,"text":text[:200],"assigned_theme":theme,"theme_label":label})
        else:
            for i, row in df.iterrows():
                features=[clean_text(v) for v in row if isinstance(v,str) and v.strip()]
                clusters=[term_to_cluster.get(f) for f in features if f in term_to_cluster]
                clusters=[c for c in clusters if c is not None]
                if clusters:
                    s=pd.Series(clusters).value_counts()
                    topf=s.max(); tied=s[s==topf].index.tolist()
                    theme=min(map(int,tied))
                    label=cluster_to_label.get(theme,f"Theme_{theme}")
                else:
                    theme=None; label="Unclassified"
                entity=row.iloc[0] if isinstance(row.iloc[0],str) else f"Row{i+1}"
                records.append({"entity_id":i+1,"entity":entity,"assigned_theme":theme,"theme_label":label})
        pd.DataFrame(records).to_csv(RESULTS_DIR/"article_theme_assign.csv",
                                     index=False,encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 7. Verification Log
        # =====================================================
        with open(RESULTS_DIR/"validation_log.txt","w",encoding="utf-8") as f:
            f.write("===== Cluster Verification Report =====\n")
            f.write(f"Vertices retained: {len(vertices)}\n")
            f.write(f"Relations retained: {len(relations)}\n")
            missing=set(vertices["name"])-set(relations["source"])-set(relations["target"])
            if missing:
                f.write(f"⚠️ Orphan vertices: {len(missing)}\n"+", ".join(list(missing))+"\n")
            else:
                f.write("✅ All vertices appear in relations.\n")

        vertices.to_csv(RESULTS_DIR/"vertices.csv",index=False,encoding="utf-8-sig")
        relations.to_csv(RESULTS_DIR/"relations.csv",index=False,encoding="utf-8-sig")

        # =====================================================
        # 🧩 STEP 8. Output Page
        # =====================================================
        return HTMLResponse(f"""
        <html><body style='font-family:Segoe UI, Noto Sans TC'>
        <h2>✅ Analysis Complete</h2>
        <p>Detected mode: <b>{mode.upper()}</b> — Themes successfully assigned to each article/entity.</p>
        <ul>
            <li>🧩 <a href='/static/themes.csv' target='_blank'>Themes (Cluster Leaders)</a></li>
            <li>🧠 <a href='/static/vertices.csv' target='_blank'>Vertices</a></li>
            <li>🔗 <a href='/static/relations.csv' target='_blank'>Relations</a></li>
            <li>🧭 <a href='/static/article_theme_assign.csv' target='_blank'>Theme–Article Assignment</a></li>
            <li>🧾 <a href='/static/validation_log.txt' target='_blank'>Validation Log</a></li>
            <li>📊 <a href='/static/theme_bar.png' target='_blank'>Theme Bar</a></li>
            <li>🌈 <a href='/static/theme_scatter.html' target='_blank'>Theme Scatter</a></li>
        </ul>
        <form action="/" method="get">
          <button style='margin-top:20px;padding:8px 16px;background:#007ACC;color:white;border:none;border-radius:6px;cursor:pointer'>
            🏠 Return Home
          </button>
        </form></body></html>""")
    except Exception:
        err=traceback.format_exc()
        return HTMLResponse(f"<h3>❌ Internal Error:</h3><pre>{err}</pre>")
