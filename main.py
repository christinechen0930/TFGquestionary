import os
import re
import requests
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import fitz  # PyMuPDF
from tavily import TavilyClient

# ====== 設定 API Key ======
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ====== 頁面設定 ======
st.set_page_config(page_title="🌿 綠園事務詢問欄", page_icon="🌱", layout="centered")
os.makedirs("downloads", exist_ok=True)

# ====== 模型加載 ======
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

model = load_model()

# ====== 搜尋並下載最新 PDF 或擷取子頁內容 ======
def search_and_process_content(keyword):
    query = f"site:fg.tp.edu.tw {keyword}"
    try:
        response = tavily_client.search(
            query,
            search_depth="advanced",
            max_results=5,
            sort_by="date"
        )
    except Exception as e:
        return f"❌ 搜尋服務錯誤：{e}"

    results = response.get("results", [])
    subpages = [r for r in results if not urlparse(r['url']).netloc.endswith("fg.tp.edu.tw") or "/news/" in r["url"]]

    if not subpages:
        suggest_words = ["招生", "校內公告", "學生活動", "校規", "交換學生"]
        suggestion = suggest_words[torch.randint(0, len(suggest_words), (1,)).item()]
        return f"❌ 找不到符合「{keyword}」的子頁面，請嘗試其他關鍵字，例如：**{suggestion}**"

    top_url = subpages[0]['url']
    try:
        page_resp = requests.get(top_url, timeout=10)
        soup = BeautifulSoup(page_resp.text, 'html.parser')
        page_text = soup.get_text()
        page_text = re.sub(r"\s+", " ", page_text).strip()

        pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
        pdf_info = None
        if pdf_links:
            pdf_url = pdf_links[0] if pdf_links[0].startswith("http") else top_url.rsplit('/', 1)[0] + '/' + pdf_links[0]
            pdf_response = requests.get(pdf_url)
            safe_keyword = re.sub(r'[\\/*?:"<>|]', "_", keyword)
            pdf_filename = os.path.join("downloads", f"{safe_keyword}_attached.pdf")
            with open(pdf_filename, "wb") as f:
                f.write(pdf_response.content)
            pdf_info = {"path": pdf_filename, "url": pdf_url}

        return {"url": top_url, "text": page_text, "pdf": pdf_info}

    except Exception as e:
        return f"❌ 無法擷取子頁面內容：{e}"

# ====== 清理文字 ======
def clean_and_split_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"第\s*\d+\s*頁", "", text)
    paragraphs = re.split(r'(?<=[。！？])', text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 10]

# ====== 讀取 PDF ======
def read_pdf(file_path):
    try:
        doc = fitz.Document(file_path)
        all_paragraphs = []
        for page in doc:
            raw_text = page.get_text()
            paragraphs = clean_and_split_text(raw_text)
            all_paragraphs.extend(paragraphs)
        return all_paragraphs
    except Exception as e:
        return [f"讀取 PDF 錯誤：{str(e)}"]

# ====== 找到相關段落 ======
def retrieve_relevant_content(task, paragraphs):
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    query_embedding = model.encode(task, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_k = min(10, len(paragraphs))
    top_results = torch.topk(scores, k=top_k)
    return "\n".join([paragraphs[idx] for idx in top_results.indices])

# ====== 整合回答 ======
def generate_response_combined(task, keyword):
    if not keyword.strip():
        return "❌ 請輸入關鍵字"

    result = search_and_process_content(keyword)
    if isinstance(result, str):
        return result

    paragraphs = clean_and_split_text(result["text"])

    if result.get("pdf"):
        pdf_paragraphs = read_pdf(result["pdf"]["path"])
        if pdf_paragraphs and "錯誤" not in pdf_paragraphs[0]:
            paragraphs.extend(pdf_paragraphs)

    if not paragraphs:
        return "❌ 無法擷取任何文字內容，請嘗試其他關鍵字。"

    relevant_content = retrieve_relevant_content(task, paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    source_links = f"- [來源網頁]({result['url']})"
    if result.get("pdf"):
        source_links += f"\n- [附加PDF]({result['pdf']['url']})"

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的內容協助回答問題，
請使用繁體中文，以條列式或摘要方式簡潔表達。

問題：{task}

相關內容：
{relevant_content}

來源清單：
{source_links}
"""

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(f"{api_url}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                model_reply = response_json["candidates"][0]["content"]["parts"][0]["text"]
                return model_reply + "\n\n---\n### 📄 資料來源\n" + source_links
            else:
                return "❌ 無法取得模型回答"
        else:
            return f"❌ 錯誤：{response.status_code}, {response.text}"
    except Exception as e:
        return f"❌ 請求失敗：{e}"

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：畢業典禮流程？")
keyword = st.text_input("輸入關鍵字（自動搜尋北一女網站）", "例如：畢業典禮")

if st.button("生成回答"):
    with st.spinner('正在處理...'):
        response = generate_response_combined(task, keyword)
        st.success('處理完成！')
        st.markdown(response)

st.markdown("---")
if st.button("瞭解北一女校史"):
    js = "window.open('https://christinechen0930.github.io/TFGquestionary/TFGhistory.html')"
    st.components.v1.html(f"<script>{js}</script>", height=0, width=0)
