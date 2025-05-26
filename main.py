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

# ====== 搜尋最新網頁 ======
def search_latest_webpage(keyword):
    query = f"site:fg.tp.edu.tw {keyword}"
    try:
        response = tavily_client.search(
            query,
            search_depth="advanced",
            max_results=5,
            sort_by="date"
        )
        results = response.get("results", [])
        if not results:
            return None, "❌ 沒找到相關網頁"
        return results[0]["url"], None
    except Exception as e:
        return None, f"❌ 搜尋服務錯誤：{e}"

# ====== 擷取網頁與 PDF ======
def extract_webpage_and_pdf(url):
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join([line for line in lines if line])

        pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]

        pdf_paths = []
        for link in pdf_links:
            if not link.startswith("http"):
                link = requests.compat.urljoin(url, link)
            try:
                pdf_response = requests.get(link, timeout=10)
                filename = os.path.join("downloads", os.path.basename(link))
                with open(filename, "wb") as f:
                    f.write(pdf_response.content)
                pdf_paths.append({"path": filename, "url": link})
            except:
                continue

        return clean_text, pdf_paths, None
    except Exception as e:
        return None, [], f"❌ 擷取網頁失敗：{e}"

# ====== 產生回應 ======
def generate_answer_from_web_and_pdf(task, keyword):
    url, error = search_latest_webpage(keyword)
    if error:
        return error

    web_text, pdf_infos, error = extract_webpage_and_pdf(url)
    if error:
        return error

    all_paragraphs = clean_and_split_text(web_text)
    for info in pdf_infos:
        all_paragraphs.extend(read_pdf(info["path"]))

    if not all_paragraphs:
        return "❌ 找不到可用的內容"

    relevant_content = retrieve_relevant_content(task, all_paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的內容"

    pdf_links_md = "\n".join([f"- [PDF 附件]({info['url']})" for info in pdf_infos]) if pdf_infos else "無 PDF 附件"

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的內容協助回答問題，
請使用繁體中文，以條列式或摘要方式簡潔表達。

問題：{task}

相關內容：
{relevant_content}

來源：
- [來源網頁]({url})
{pdf_links_md}
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
            model_reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return model_reply + f"\n\n---\n### 📄 來源\n- [網頁連結]({url})\n{pdf_links_md}"
        else:
            return f"❌ 錯誤：{response.status_code}"
    except Exception as e:
        return f"❌ 請求失敗：{e}"

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：畢業典禮時間？")
keyword = st.text_input("輸入關鍵字", "例如：畢業典禮")

if st.button("生成回答"):
    with st.spinner("正在搜尋並分析最新網頁..."):
        response = generate_answer_from_web_and_pdf(task, keyword)
        st.success("完成！")
        st.markdown(response)

st.markdown("---")
if st.button("瞭解北一女校史"):
    js = "window.open('https://christinechen0930.github.io/TFGquestionary/TFGhistory.html')"
    st.components.v1.html(f"<script>{js}</script>", height=0, width=0)
