

# 🧠 TAAA Semantic Extractor

**Theme Assignment Algorithm in Articles (TAAA)** — an AI-powered assistant for bibliometric research.  
It automatically extracts **semantic keywords** from journal abstracts in Chinese or English using OpenAI’s `gpt-4o-mini` model.  
Designed for large-scale theme assignment, metadata analysis, and journal trend studies.

---

## ✨ Features

- 🔍 Upload CSVs containing abstracts → auto-extract 10 semantic keywords per row  
- ⚙️ Smart encoding detection (`UTF-8`, `BIG5`, `CP950`) for Traditional Chinese datasets  
- 💾 Output downloadable as `taaa_keywords.csv` (UTF-8-BOM for Excel compatibility)  
- 🖥️ Simple HTML front-end with dark-mode toggle and ChatGPT-style progress animation  
- 🤖 Directly connectable to **ChatGPT Custom GPT Store** through OpenAPI schema  

---

## 📁 Folder Structure
TAAA_Semantic_Extractor/
├── app/
│ ├── main.py # FastAPI application
│ ├── utils.py # GPT keyword extraction helper
│ ├── index.html # Frontend upload interface
│ ├── style.css # UI styling (light/dark)
│ ├── openapi.json # OpenAPI schema for GPT Store
│ ├── requirements.txt # Python dependencies
├── deployment.yaml # Render deployment configuration
└── README.md # This file


---

## ⚙️ Local Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YourUser/TAAA_Semantic_Extractor.git
cd TAAA_Semantic_Extractor/app



pip install -r requirements.txt



