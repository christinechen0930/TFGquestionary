import os
import re
import requests
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ====== 頁面設定 ======
st.set_page_config(page_title="🌿 綠園事務詢問欄", page_icon="🌱", layout="centered")
os.makedirs("downloads", exist_ok=True)

# ====== 模型加載 ======
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

model = load_model()

# ====== 擷取最新消息中的子頁面 ======
def fetch_latest_news_links(keyword):
    base_url = "https://www.fg.tp.edu.tw"
    news_url = f"{base_url}/category/news/news1/"
    try:
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"❌ 無法連接到最新消息頁面：{e}"

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", href=True)

    matched_links = []
    for a in articles:
        title = a.get_text(strip=True)
        href = a["href"]
        if keyword in title and href.startswith("/news/"):
            full_url = urljoin(base_url, href)
            matched_links.append(full_url)

    if not matched_links:
        return f"❌ 找不到符合「{keyword}」的子頁面，請嘗試其他關鍵字。"

    return matched_links

# ====== 擷取網頁文字內容 ======
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"❌ 無法連接到網頁：{e}"

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="entry-content")
    if not content_div:
        return "❌ 無法找到網頁內容。"

    paragraphs = content_div.stripped_strings
    return "\n".join(paragraphs)

# ====== 下載並讀取 PDF 檔案 ======
def download_and_read_pdfs(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return [], f"❌ 無法連接到網頁：{e}"

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = soup.find_all("a", href=re.compile(r".*\.pdf$"))

    texts = []
    for link in pdf_links:
        pdf_url = urljoin(url, link["href"])
        try:
            pdf_response = requests.get(pdf_url, timeout=10)
            pdf_response.raise_for_status()
            pdf_filename = os.path.join("downloads", os.path.basename(pdf_url))
            with open(pdf_filename, "wb") as f:
                f.write(pdf_response.content)

            doc = fitz.open(pdf_filename)
            for page in doc:
                texts.append(page.get_text())
            doc.close()
        except Exception as e:
            texts.append(f"❌ 無法下載或讀取 PDF：{e}")

    return texts, None

# ====== 清理文字 ======
def clean_and_split_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"第\s*\d+\s*頁", "", text)
    paragraphs = re.split(r'(?<=[。！？])', text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 10]

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

    links = fetch_latest_news_links(keyword)
    if isinstance(links, str):
        return links

    all_texts = []
    for link in links:
        page_text = extract_text_from_url(link)
        pdf_texts, error = download_and_read_pdfs(link)
        if error:
            return error
        all_texts.extend([page_text] + pdf_texts)

    paragraphs = []
    for text in all_texts:
        paragraphs.extend(clean_and_split_text(text))

    if not paragraphs:
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    relevant_content = retrieve_relevant_content(task, paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    source_links = "\n".join([f"- [來源頁面]({link})" for link in links])

    response = f"""
### 🔍 問題：{task}

{relevant_content}

---

### 📄 來源頁面
{source_links}
"""
    return response

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：如何申請交換學生？")
keyword = st.text_input("輸入關鍵字（自動搜尋北一女最新消息）", "例如：畢業典禮")

if st.button("生成回答"):
    with st.spinner('正在處理...'):
        response = generate_response_combined(task, keyword)
        st.success('處理完成！')
        st.markdown(response)

st.markdown("---")
if st.button("瞭解北一女校史"):
    js = "window.open('https://christinechen0930.github.io/TFGquestionary/TFGhistory.html')"
    st.components.v1.html(f"<script>{js}</script>", height=0, width=0)
