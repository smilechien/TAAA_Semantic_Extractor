# 🌐 TAAA Semantic–Co-Word Analyzer (v4.7)

![Build](https://img.shields.io/github/actions/workflow/status/smilechien/TAAA_Semantic_Analyzer/render-deploy.yml?label=Build&logo=render)
[![Live App](https://img.shields.io/badge/Live%20App-Open-blue?logo=render)](https://taaa-semantic-analyzer.onrender.com/)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.9999999.svg)](https://doi.org/10.5281/zenodo.9999999)

---

### 🧠 Overview

**TAAA Semantic–Co-Word Analyzer** is a multilingual FastAPI-based web tool that performs:
- GPT-powered or TF-IDF fallback **semantic keyword extraction**  
- **Co-word network clustering (Louvain algorithm)**  
- **TAAA (Theme Assignment Algorithm in Articles)** for per-row thematic labeling  
- **h-index bar chart & top-20 term network visualization**  
- **Automatic font & language switching** for publication-quality plots  

Built for bibliometric and semantic-trend analysis in academic datasets  
(e.g., abstracts from PubMed, WoS, Scopus, Airiti).

---

### ⚙️ Features

| Function | Description |
|-----------|-------------|
| **Multilingual Input** | Handles English, 中文, 日本語, 한국어, Español automatically |
| **Three Semantic Engines** | GPT-4o → TF-IDF → ChatGPT fallback modes |
| **TAAA Assignment** | Assigns theme & cluster to each document row |
| **Publication-Quality Figures** | Auto-switch fonts (`Noto Sans CJK TC` ↔ `Segoe UI`) |
| **Render-Ready Health Check** | `/health` endpoint for API + engine status |
| **Downloadable Outputs** | Vertices, relations, and full dataset (with theme) |

---
### 🧱 Folder structure reminder

Your repo should now look like:

TAAA_Semantic_Extractor/
├── app/
│   ├── main.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── abstract_sample.csv
│   │   ├── coword_sample.csv
│   │   ├── readme_dataset_info.csv
│   └── ...
└── requirements.txt

###✅ Summary of what this version does
Feature	Description
HTML Frontend	Serves app/templates/index.html cleanly
Multilingual engine	Auto-detects language (English/中文/etc.)
Engine fallback	Uses GPT if available, otherwise TF-IDF
Modes	abstract (semantic extraction) or coword (co-occurrence)
Outputs	2 downloadable CSVs (_vertices, _relations)
Plots	Scatter plot with red reference lines
Health Check	/health endpoint returns status JSON
Render ready	Works with Python 3.12–3.13 on Render
### 🚀 Quick Start

 
