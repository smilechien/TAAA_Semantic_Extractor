# utils.py
import re, requests, os
from langdetect import detect
from openai import OpenAI

def get_gpt_client(filename: str):
    if "smilechien" in filename.lower():
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY missing.")
        return OpenAI(api_key=key)
    return None

def detect_language(text):
    try:
        code = detect(text)
    except Exception:
        code = "unknown"
    lang_map = {
        "zh-cn":"Chinese","zh-tw":"Chinese","en":"English",
        "ja":"Japanese","ko":"Korean","fr":"French","es":"Spanish"
    }
    return lang_map.get(code, code)

def fetch_abstract_from_doi(doi):
    for url in [
        f"https://api.crossref.org/works/{doi}",
        f"https://api.openalex.org/works/doi:{doi}"
    ]:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                js = r.json()
                abs_ = js.get("message", {}).get("abstract") or js.get("abstract", {}).get("value")
                if abs_:
                    return re.sub(r"<[^>]+>", "", abs_)
        except Exception:
            pass
    return ""
