# 🌐 TAAA Semantic Extractor (Global Edition)  
**多語言語義關鍵詞擷取系統 | Multilingual Semantic Keyword Extractor for Academic Abstracts**

> ✨ Built with [FastAPI](https://fastapi.tiangolo.com/) + [OpenAI GPT-4o-mini](https://platform.openai.com/docs/models/gpt-4o)  
> Developed by **Smile Chien (RaschOnline)**  

---

## 🚀 Live Demo | 線上體驗
🔗 **https://taaa-semantic-extractor.onrender.com**

Upload a CSV file containing research abstracts and get **10 semantically representative keywords** — automatically adapting to the abstract’s detected language.

上傳包含研究摘要的 CSV 檔案，即可自動萃取「十個具語義代表性」的學術關鍵詞。  
系統可依據摘要語言自動判斷、或強制生成中英雙語結果。

---

## 🧠 Introduction | 專案簡介

**TAAA Semantic Extractor** (Theme Assignment Algorithm in Articles, TAAA)  
is a GPT-powered academic tool that performs **semantic keyword extraction** from abstracts.  

It is designed for multilingual bibliometric and scientometric workflows, allowing seamless use in **Chinese**, **English**, **Japanese**, **Korean**, **Spanish**, **French**, and other languages.

本系統以「論文主題指派演算法」（TAAA, Theme Assignment Algorithm in Articles）為核心，  
結合 GPT-4o-mini 的自然語義理解能力，提供多語言研究摘要的語義關鍵詞萃取。  

---

## 🌍 Key Features | 系統特色

| Feature | 說明 |
|----------|------|
| 💬 **Auto-Detect Mode** | Automatically detects the language of each abstract and generates keywords in that language. |
| 🌏 **Bilingual Mode** | Always generates *both Traditional Chinese and English* keywords. |
| 🧩 **Multi-language Support** | Works for Chinese, English, Japanese, Korean, French, Spanish, and more. |
| ⚡ **FastAPI-based** | Lightweight, scalable, and deployable on Render or GPT Store. |
| 🔐 **Private API Key Support** | Securely uses your OpenAI API key via environment variables. |
| 📄 **CSV I/O Workflow** | Accepts and returns UTF-8 CSV files with an `abstract` column. |

---

## 🧪 Example | 範例輸入與輸出

### 🗂️ Input CSV
```csv
id,abstract
1,本研究探討人工智慧在醫療影像分析中的應用與挑戰。
2,This study evaluates deep learning approaches for disease detection in radiology.



id,abstract,language,keywords
1,本研究探討人工智慧在醫療影像分析中的應用與挑戰。,chinese,人工智慧、醫療影像、深度學習、分類模型、診斷準確率
2,This study evaluates deep learning approaches for disease detection in radiology.,english,deep learning,medical imaging,AI,diagnostic accuracy,radiology

User → FastAPI (main.py)
       ↓
OpenAI GPT-4o-mini
       ↓
Keyword Extraction → CSV Download

export OPENAI_API_KEY="sk-your-key-here"

TAAA_Semantic_Extractor/
 ├─ app/
 │   ├─ main.py
 │   ├─ index.html
 │   ├─ sample_abstracts.csv
 │   ├─ sample_abstracts_bilingual.csv
 │   └─ requirements.txt
 ├─ deployment.yaml
 └─ README.md


