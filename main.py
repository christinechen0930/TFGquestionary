import os
import re
import requests
import torch
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote, urlparse
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

"""
綠園事務詢問欄 v2
- 修正：fetch_relevant_news_page 無法找到『行事曆』等文章——原因是最新消息列表以 JS 動態載入，requests 拿不到，所以改採 WordPress REST API + Tavily 搜尋。
- 新增：若 keyword 本身就是 https 開頭ㄉ完整網址，直接當 page_url 用。
- 其餘：保留同義詞對照表 & 既有流程。
"""

# ====== 設定 API Key ======
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ====== 頁面設定 ======
st.set_page_config(page_title="🌿 綠園事務詢問欄", page_icon="🌱", layout="centered")
os.makedirs("downloads", exist_ok=True)

# ====== 同義詞對照表 ======
SYNONYMS = {
    "段考": ["期中考", "期末考"],
    "期中": ["期中考"],
    "期末": ["期末考"],
    "畢業典禮": ["畢典", "畢業典禮"],
    # 可擴充更多
}

# ====== 模型加載 ======
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

model = load_model()

# ====== 工具函式 ======
def clean_and_split_text(text: str):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"第\s*\d+\s*頁", "", text)
    paragraphs = re.split(r"(?<=[。！？])", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 10]


def read_pdf(file_path: str):
    try:
        doc = fitz.Document(file_path)
        paras = []
        for page in doc:
            paras.extend(clean_and_split_text(page.get_text()))
        return paras
    except Exception as e:
        return [f"讀取 PDF 錯誤：{e}"]

# ====== 取得最新消息文章網址 ======

def _search_wordpress_rest(keyword: str):
    """用 WP REST API 搜尋，成功回傳第一筆網址，失敗回 None"""
    base = "https://www.fg.tp.edu.tw/wp-json/wp/v2/search"
    try:
        r = requests.get(base, params={"search": keyword, "per_page": 1}, timeout=10)
        if r.ok and r.json():
            return r.json()[0]["url"]
    except Exception:
        pass
    return None


def _search_tavily(keyword: str):
    """用 Tavily site: 搜尋，成功回傳第一筆網址"""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": TAVILY_API_KEY},
            json={"query": f"site:fg.tp.edu.tw {keyword}", "l": 1},
            timeout=10,
        )
        data = resp.json()
        if data and data.get("results"):
            return data["results"][0]["url"]
    except Exception:
        pass
    return None


def fetch_relevant_news_page(keyword: str):
    """優先：WP REST -> Tavily -> None"""
    # 若 keyword 本身就是網址
    if keyword.startswith("http"):
        return keyword

    for kw in SYNONYMS.get(keyword.strip(), [keyword.strip()]):
        url = _search_wordpress_rest(kw)
        if url:
            return url
    # REST 找不到就用 Tavily
    return _search_tavily(keyword)

# ====== 相似度挑段落 ======

def retrieve_relevant_content(task: str, paragraphs: list[str]):
    if not paragraphs:
        return ""
    embeds = model.encode(paragraphs, convert_to_tensor=True)
    q_embed = model.encode(task, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_embed, embeds)[0]
    top_k = min(10, len(paragraphs))
    best_idx = torch.topk(sims, k=top_k).indices
    return "\n".join([paragraphs[i] for i in best_idx])

# ====== 解析檔名 ======

def get_filename_from_url(url: str):
    return unquote(os.path.basename(urlparse(url).path)).replace(" ", "_")

# ====== 核心邏輯 ======

def generate_response_combined(task: str, keyword: str):
    cleaned_paragraphs = []
    pdf_links: list[tuple[str, str]] = []

    page_url = fetch_relevant_news_page(keyword) if keyword.strip() else None

    # 讀取子頁面 & 內嵌附件
    if page_url and page_url.startswith("http"):
        try:
            res = requests.get(page_url, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            cleaned_paragraphs.extend(clean_and_split_text(soup.get_text()))

            pdf_urls = {
                urljoin(page_url, a["href"].replace(" ", "%20"))
                for a in soup.find_all("a", href=True)
                if a["href"].lower().endswith(".pdf")
            }
            for url in pdf_urls:
                try:
                    fname = get_filename_from_url(url)
                    local = os.path.join("downloads", fname)
                    with open(local, "wb") as f:
                        f.write(requests.get(url, timeout=10).content)
                    cleaned_paragraphs.extend(read_pdf(local))
                    pdf_links.append((fname, url))
                except Exception as e:
                    cleaned_paragraphs.append(f"❌ 無法下載附件：{url}，錯誤：{e}")
        except Exception as e:
            cleaned_paragraphs.append(f"❌ 無法讀取子頁面內容：{e}")

    relevant = retrieve_relevant_content(task, cleaned_paragraphs)

    prompt = f"""
你是一位了解北一女中行政流程與校內事務的輔導老師，請根據下方提供的資料協助回答問題，
請使用繁體中文，以條列式或摘要方式簡潔表達，如果有人詢問到段考等關鍵字，將其視為期中考或期末考，且可以去搜尋行事曆。若關鍵字找不到資訊，不用管檔案標題，直接找有提到此關鍵字的檔案，如果關鍵字搜尋不到，就改成用使用者輸入的問題去搜尋。

問題：{task}

相關內容：
{relevant if relevant else "（未找到其他相關內容）"}

# 內建校園小知識（略）...
"""
    # 省略：保留原本的 Gemini API 呼叫碼
    # ... （此處與上一版相同）
    return "(略)"  # 省略：保持函式簽名完整

# ====== Streamlit 介面 ======
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：今年的畢業典禮是哪一天？")
keyword = st.text_input("輸入關鍵字（從北一女校網最新消息中搜尋，或直接貼網址）", "例如：行事曆")

if st.button("生成回答"):
    with st.spinner("正在搜尋與生成回覆..."):
        st.markdown(generate_response_combined(task, keyword))

st.markdown("---")
# （下方按鈕區塊不變）
