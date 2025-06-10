import os
import re
import requests
import torch
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote, urlparse
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
    except Exception:
        return None  # ❗ 改為回傳 None，不終止流程

    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    for link in links:
        title = link.get_text(strip=True)
        href = link["href"]
        if keyword in title and "/news/" in href:
            return urljoin(base_url, href)

    return None  # ❗ 沒找到就回 None

# ====== 找到相關段落 ======
def retrieve_relevant_content(task, paragraphs):
    if not paragraphs:
        return ""
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    query_embedding = model.encode(task, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_k = min(10, len(paragraphs))
    top_results = torch.topk(scores, k=top_k)
    return "\n".join([paragraphs[idx] for idx in top_results.indices])

# ====== 從 URL 解析出原始檔名 ======
def get_filename_from_url(url):
    path = urlparse(url).path
    return unquote(os.path.basename(path)).replace(" ", "_")

# ====== 整合回答邏輯 ======
def generate_response_combined(task, keyword):
    cleaned_paragraphs = []
    pdf_links_collected = []
    page_url = None

    if keyword.strip():
        page_url = fetch_relevant_news_page(keyword)
        if page_url:
            try:
                res = requests.get(page_url, timeout=10)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                content_text = soup.get_text()
                cleaned_paragraphs.extend(clean_and_split_text(content_text))

                # 擷取 PDF
                pdf_links = {
                    urljoin(page_url, a["href"].replace(" ", "%20"))
                    for a in soup.find_all("a", href=True)
                    if a["href"].endswith(".pdf")
                }

                for pdf_url in pdf_links:
                    try:
                        file_name = get_filename_from_url(pdf_url)
                        local_path = os.path.join("downloads", file_name)
                        r = requests.get(pdf_url, timeout=10)
                        with open(local_path, "wb") as f:
                            f.write(r.content)
                        cleaned_paragraphs.extend(read_pdf(local_path))
                        pdf_links_collected.append((file_name, pdf_url))
                    except Exception as e:
                        cleaned_paragraphs.append(f"❌ 無法下載附件：{pdf_url}，錯誤：{e}")
            except Exception as e:
                cleaned_paragraphs.append(f"❌ 無法讀取子頁面內容：{e}")

    # 🔍 不管有沒有找到網頁，都要繼續處理
    relevant_content = retrieve_relevant_content(task, cleaned_paragraphs)

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的資料協助回答問題，
請使用繁體中文，以條列式或摘要方式簡潔表達。

問題：{task}

相關內容：
{relevant_content if relevant_content else "（未找到其他相關內容）"}

關於北一女中的大小事：
1. 北一女中制服上衣是綠色的，運動服上衣是白色的。
2. 萬聖節可以穿任何服裝，包括友校制服、南瓜裝、北一制服。
3. 北一女中學生不會出校吃午餐，會吃大小熱的食物，或者訂外送、自己帶便當。
4. 小熱的薯不辣是由薯條和甜不辣搭配而成。
5. 大熱近兩年沒有賣過鹽酥雞，有賣過糖葫蘆、仙草蜜、芒果冰。
6. 北一女中光復樓是古蹟。
7. 北一女中學珠樓是以江學珠校長的名字命名。
8. 仁愛樓不在北一女中。
"""

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(f"{api_url}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                model_reply = response_json["candidates"][0]["content"]["parts"][0]["text"]
                attachments_text = ""
                if pdf_links_collected:
                    attachments_text += "\n📎 附件下載：\n"
                    for name, link in pdf_links_collected:
                        attachments_text += f"- [{name}]({link})\n"
                source_note = f"\n\n---\n🔗 [來源子頁面]({page_url})" if page_url else "\n\n---\n⚠️ 未從校網找到子頁面。"
                return model_reply + source_note + attachments_text
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

st.markdown("""
<style>
.button-container {
    display: flex;
    justify-content: flex-start;
    gap: 0.5rem;
    margin-top: 1rem;
}

a.fake-button {
    background-color: #f0f2f6;
    color: black;
    padding: 0.4rem 1rem;
    text-decoration: none;
    border-radius: 0.5rem;
    border: 1px solid #d3d3d3;
    font-weight: 500;
    font-size: 1rem;
    display: inline-block;
    transition: background-color 0.2s ease;
}

a.fake-button:hover {
    background-color: #e0e0e0;
}
</style>

<div class="button-container">
    <a href="https://christinechen0930.github.io/TFGquestionary/TFGQA.html" target="_blank" class="fake-button">🔍 前往北一女中問答集</a>
    <a href="https://christinechen0930.github.io/TFGquestionary/TFGhistory.html" target="_blank" class="fake-button">📜 瞭解北一女校史</a>
</div>
""", unsafe_allow_html=True)
