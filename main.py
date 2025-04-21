import os
import torch
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from tavily import TavilyClient
import re

# ====== 設定 API Key ======
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ====== 頁面設定 ======
st.set_page_config(page_title="🌿 綠園事務詢問欄", page_icon="🌱", layout="centered")

# ====== 加載模型 ======
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

# ====== 讀取 PDF 檔內容 ======
def read_pdf(file_path):
    try:
        doc = fitz.Document(file_path)
        all_paragraphs = []
        for page in doc:
            text = page.get_text()
            all_paragraphs.extend(clean_and_split_text(text))
        return all_paragraphs
    except Exception as e:
        return [f"讀取 PDF 錯誤：{str(e)}"]

# ====== 搜尋與下載 PDF（保留 URL） ======
def search_and_download_pdfs(keyword):
    query = f"site:fg.tp.edu.tw {keyword} filetype:pdf"
    try:
        response = tavily_client.search(query)
    except Exception as e:
        return f"❌ 服務錯誤：{e}"

    results = [r for r in response.get("results", []) if r["url"].endswith(".pdf")]
    if not results:
        return "❌ 沒有找到相關的 PDF 檔案！"

    pdf_data = []
    for i, r in enumerate(results):
        try:
            url = r["url"]
            filename = f"downloads/{keyword}_{i+1}.pdf"
            pdf_bytes = requests.get(url, timeout=10).content
            with open(filename, "wb") as f:
                f.write(pdf_bytes)
            pdf_data.append((filename, url))
        except Exception as e:
            return f"❌ 下載失敗：{url}，錯誤：{e}"

    return pdf_data

# ====== 擷取相關內容 ======
def retrieve_relevant_content(task, paragraphs):
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    query_embedding = model.encode(task, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_k = min(10, len(paragraphs))
    top_results = torch.topk(scores, k=top_k)
    return "\n".join([paragraphs[idx] for idx in top_results.indices])

# ====== 整合回答 ======
def generate_response_combined(task, keyword, file=None):
    if not keyword.strip() and not file:
        return "❌ 請輸入關鍵字或上傳 PDF"

    if file:
        paragraphs = read_pdf(file)
        sources = [file.name]
        urls = []
    else:
        result = search_and_download_pdfs(keyword)
        if isinstance(result, str):
            return result
        paragraphs = []
        sources = []
        urls = []
        for local_path, url in result:
            paragraphs.extend(read_pdf(local_path))
            sources.append(os.path.basename(local_path))
            urls.append(url)

    if not paragraphs or "錯誤" in paragraphs[0]:
        return paragraphs[0]

    relevant = retrieve_relevant_content(task, paragraphs)
    if not relevant.strip():
        return "❌ 找不到與問題相關的內容，請嘗試其他關鍵字。"

    source_links = "\n".join(
        [f"- [{name}]({url})" for name, url in zip(sources, urls)]
    ) if urls else f"- 使用者上傳：{sources[0]}"

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的文件內容協助回答問題。
回答請使用繁體中文，並以條列式或摘要方式簡潔表達。

問題：{task}

相關內容：
{relevant}

來源清單：
{source_links}
    """

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        res = requests.post(f"{api_url}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        if res.status_code == 200:
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text + "\n\n---\n### 📄 來源 PDF 文件\n" + source_links
        else:
            return f"❌ 錯誤：{res.status_code} - {res.text}"
    except Exception as e:
        return f"❌ 請求失敗：{e}"

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")
task = st.text_input("輸入詢問事項", "例如：如何申請交換學生？")
keyword = st.text_input("輸入關鍵字（自動搜尋北一女 PDF）", "例如：招生簡章")
file = st.file_uploader("或上傳 PDF", type=["pdf"])

if st.button("生成回答"):
    with st.spinner("處理中，請稍候..."):
        result = generate_response_combined(task, keyword, file)
    st.success("完成！")
    st.markdown(result)
