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

### 🚀 Quick Start

 
