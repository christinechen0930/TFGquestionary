import os
import re
import requests
import torch
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
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

# ====== 搜尋最新網頁 + 擷取網頁與 PDF ======
def search_latest_fgu_webpage_and_pdf(keyword):
    query = f"site:fg.tp.edu.tw {keyword} -inurl:fg.tp.edu.tw$"
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
    filtered = [r for r in results if "fg.tp.edu.tw" in r["url"] and not re.fullmatch(r"https?://(www\\.)?fg\\.tp\\.edu\\.tw/?", r["url"])]

    if not filtered:
        return f"❌ 找不到符合「{keyword}」的子頁面，請嘗試其他關鍵字。"

    latest_page = filtered[0]
    page_url = latest_page["url"]

    try:
        page_html = requests.get(page_url, timeout=10).text
        soup = BeautifulSoup(page_html, "html.parser")
        content_tags = soup.select("article, main, .content, .entry-content")
        text_content = " ".join(tag.get_text(separator=" ", strip=True) for tag in content_tags if tag)

        pdf_links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")]
        pdf_path = None
        pdf_url = None
        if pdf_links:
            pdf_url = pdf_links[0]
            if not pdf_url.startswith("http"):
                pdf_url = requests.compat.urljoin(page_url, pdf_url)
            response = requests.get(pdf_url, timeout=10)
            safe_keyword = re.sub(r'[\\/*?:"<>|]', "_", keyword)
            pdf_path = os.path.join("downloads", f"{safe_keyword}_webpage.pdf")
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        return {
            "page_url": page_url,
            "text_content": text_content.strip(),
            "pdf_path": pdf_path,
            "pdf_url": pdf_url
        }

    except Exception as e:
        return f"❌ 讀取網頁失敗：{e}"

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

    result = search_latest_fgu_webpage_and_pdf(keyword)
    if isinstance(result, str):
        return result

    text_paragraphs = clean_and_split_text(result["text_content"])
    pdf_paragraphs = read_pdf(result["pdf_path"]) if result["pdf_path"] else []

    all_paragraphs = text_paragraphs + pdf_paragraphs
    if not all_paragraphs or "錯誤" in all_paragraphs[0]:
        return all_paragraphs[0] if all_paragraphs else "❌ 找不到有效內容"

    relevant_content = retrieve_relevant_content(task, all_paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    source_links = f"- [來源網頁]({result['page_url']})"
    if result["pdf_url"]:
        source_links += f"\n- [PDF 附件]({result['pdf_url']})"

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的文件內容協助回答問題，
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
                return model_reply + "\n\n---\n### 📄 來源連結\n" + source_links
            else:
                return "❌ 無法取得模型回答"
        else:
            return f"❌ 錯誤：{response.status_code}, {response.text}"
    except Exception as e:
        return f"❌ 請求失敗：{e}"

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：如何申請交換學生？")
keyword = st.text_input("輸入關鍵字（自動搜尋北一女相關頁面）", "例如：交換學生")

if st.button("生成回答"):
    with st.spinner('正在處理...'):
        response = generate_response_combined(task, keyword)
        st.success('處理完成！')
        st.markdown(response)

st.markdown("---")
if st.button("瞭解北一女校史"):
    js = "window.open('https://christinechen0930.github.io/TFGquestionary/TFGhistory.html')"
    st.components.v1.html(f"<script>{js}</script>", height=0, width=0)
