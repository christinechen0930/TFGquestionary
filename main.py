file_path = "/mnt/data/green_garden_inquiry.py"
with open(file_path, "w", encoding="utf-8") as f:
    f.write("""import streamlit as st
import requests
import os
import torch
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import docx  # 需要安裝 python-docx
from tavily import TavilyClient

# 設定 Tavily API Key（請替換成你的 API Key）
API_KEY = "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"
tavily_client = TavilyClient(api_key=API_KEY)

# 建立下載資料夾
DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# 加載語義搜尋模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def search_and_download_pdfs(keyword):
    \"\"\"根據關鍵字搜尋北一女校網 PDF 並下載\"\"\"
    query = f"site:fg.tp.edu.tw {keyword} filetype:pdf"
    response = tavily_client.search(query)
    pdf_links = [result["url"] for result in response.get("results", []) if result["url"].endswith(".pdf")]

    if not pdf_links:
        return "❌ 沒有找到相關的 PDF 檔案！"

    pdf_paths = []
    for index, pdf_url in enumerate(pdf_links):
        try:
            response = requests.get(pdf_url, timeout=10)
            pdf_filename = os.path.join(DOWNLOAD_FOLDER, f"{keyword}_{index + 1}.pdf")
            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            pdf_paths.append(pdf_filename)
        except Exception as e:
            return f"❌ 下載失敗：{pdf_url}，錯誤：{e}"

    return pdf_paths

def read_pdf(file_path):
    \"\"\"從 PDF 文件中提取文字\"\"\"
    try:
        doc = fitz.Document(file_path)
        full_text = [page.get_text() for page in doc]
        return full_text
    except Exception as e:
        return [f"讀取 PDF 文件時出現錯誤：{str(e)}"]

def read_docx(file):
    \"\"\"從 DOCX 文件中提取文字\"\"\"
    try:
        doc = docx.Document(file.name)
        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return full_text
    except Exception as e:
        return [f"讀取 DOCX 文件時出現錯誤：{str(e)}"]

def retrieve_relevant_content(task, paragraphs):
    \"\"\"根據任務需求執行語義搜尋，以檢索相關內容\"\"\"
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)
    query_embedding = model.encode(task, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    top_k = min(5, len(paragraphs))
    top_results = torch.topk(scores, k=top_k)

    relevant_paragraphs = [paragraphs[idx] for idx in top_results.indices]
    return " ".join(relevant_paragraphs)

def generate_response_combined(task, keyword, file):
    \"\"\"根據關鍵字搜尋校網 PDF 或處理上傳的文件來回答問題。\"\"\"
    
    if file:
        # **有上傳檔案時，直接解析檔案**
        if file.name.endswith(".pdf"):
            paragraphs = read_pdf(file)
        elif file.name.endswith(".docx"):
            paragraphs = read_docx(file)
        else:
            return "❌ 文件格式不支援，請上傳 PDF 或 DOCX！"
    else:
        # **沒有上傳檔案時，自動搜尋校網 PDF**
        if not keyword.strip():
            return "❌ 請輸入關鍵字或上傳文件！"
        
        pdf_paths = search_and_download_pdfs(keyword)
        if isinstance(pdf_paths, str):  # 若搜尋失敗，回傳錯誤訊息
            return pdf_paths
        
        # 讀取下載的 PDF
        paragraphs = []
        for pdf_path in pdf_paths:
            paragraphs.extend(read_pdf(pdf_path))
    
    if not paragraphs or (isinstance(paragraphs[0], str) and "錯誤" in paragraphs[0]):
        return paragraphs[0]  # 返回錯誤訊息
    
    # **使用語義搜尋確保回答精準**
    relevant_content = retrieve_relevant_content(task, paragraphs)
    if not relevant_content.strip():
        return "❌ 找不到與問題相關的資訊，請嘗試其他關鍵字。"

    # **使用 Gemini API 生成回答**
    prompt = f\"\"\"
    請根據以下文件內容回答問題：
    問題：{task}
    相關內容：
    {relevant_content}

    請用繁體中文回答，並用條列式或簡短摘要表達，避免無關資訊。
    \"\"\"

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    api_key = "AIzaSyC25eTdPDzuMqv3ZE_I8l6gpuv0faBA88c"

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(f"{api_url}?key={api_key}", json=payload, headers=headers)
    
    if response.status_code == 200:
        response_json = response.json()
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "❌ 無法找到生成的文本"
    else:
        return f"❌ 錯誤: {response.status_code}, {response.text}"

# Streamlit 部分
st.title("🌱 綠園事務詢問欄")

task = st.text_input("輸入詢問事項", "例如：如何申請交換學生？")
keyword = st.text_input("輸入關鍵字（自動搜尋北一女 PDF）", "例如：招生簡章")
file_input = st.file_uploader("或上傳 PDF / DOCX", type=["pdf", "docx"])

if st.button("生成回答"):
    response = generate_response_combined(task, keyword, file_input)
    st.markdown(response)""")

print(f"檔案已存入: {file_path}")