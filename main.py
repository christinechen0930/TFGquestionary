import os
import re
import requests
import torch
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

# ====== 設定 API Key ======
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ====== 頁面設定 ======
st.set_page_config(page_title="🌿 綠園事務詢問欄", page_icon="🌱", layout="centered")
os.makedirs("downloads", exist_ok=True)

# ====== 模型加載 ======
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

model = load_model()

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

# ====== 擷取北一女最新消息中的關鍵字子頁面 ======
def fetch_relevant_news_page(keyword):
    base_url = "https://www.fg.tp.edu.tw"
    news_url = f"{base_url}/category/news/news1/"
    try:
        res = requests.get(news_url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        return f"❌ 無法連接到最新消息頁面：{e}"

    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    matched_pages = []
    for link in links:
        title = link.get_text(strip=True)
        href = link["href"]
        if keyword in title and "/news/" in href:
            full_url = urljoin(base_url, href)
            matched_pages.append(full_url)

    if not matched_pages:
        return f"❌ 找不到符合「{keyword}」的子頁面，請嘗試其他關鍵字。"

    return matched_pages[0]  # 最新一筆子頁面網址

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

    page_url = fetch_relevant_news_page(keyword)
    if isinstance(page_url, str) and page_url.startswith("❌"):
        return page_url

    try:
        res = requests.get(page_url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        return f"❌ 無法讀取子頁面內容：{e}"

    soup = BeautifulSoup(res.text, "html.parser")
    content_text = soup.get_text()
    cleaned_paragraphs = clean_and_split_text(content_text)

    # 擷取 PDF 連結
    pdf_links = [urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")]

    # 下載並讀取 PDF
    for i, pdf_url in enumerate(pdf_links):
        try:
            r = requests.get(pdf_url, timeout=10)
            local_path = os.path.join("downloads", f"attached_{i}.pdf")
            with open(local_path, "wb") as f:
                f.write(r.content)
            cleaned_paragraphs.extend(read_pdf(local_path))
        except Exception as e:
            cleaned_paragraphs.append(f"❌ 無法下載附件：{pdf_url}，錯誤：{e}")

    relevant_content = retrieve_relevant_content(task, cleaned_paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的資料協助回答問題，
請使用繁體中文，以條列式或摘要方式簡潔表達。

問題：{task}

相關內容：
{relevant_content}

來源：
{page_url}
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

                # 產生 PDF 附件連結（避免重複，並顯示檔名）
                if pdf_links:
                    seen = set()
                    attachments = []
                    for url in pdf_links:
                        cleaned_url = url.replace(" ", "%20")
                        if cleaned_url not in seen:
                            seen.add(cleaned_url)
                            filename = os.path.basename(cleaned_url)
                            attachments.append(f"- [{filename}]({cleaned_url})")
                    if attachments:
                        attachment_text = "\n📎 附件下載：\n" + "\n".join(attachments)
                        model_reply += attachment_text

                model_reply += f"\n\n---\n🔗 [來源子頁面]({page_url})"
                return model_reply
            else:
                return "❌ 無法取得模型回答"
        else:
            return f"❌ 錯誤：{response.status_code}, {response.text}"
    except Exception as e:
        return f"❌ 請求失敗：{e}"

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：今年的畢業典禮是哪一天？")
keyword = st.text_input("輸入關鍵字（從北一女校網最新消息中搜尋）", "例如：畢業典禮")

if st.button("生成回答"):
    with st.spinner('正在搜尋與生成回覆...'):
        response = generate_response_combined(task, keyword)
        st.success('處理完成！')
        st.markdown(response)

st.markdown("---")
if st.button("瞭解北一女校史"):
    js = "window.open('https://christinechen0930.github.io/TFGquestionary/TFGhistory.html')"
    st.components.v1.html(f"<script>{js}</script>", height=0, width=0)
