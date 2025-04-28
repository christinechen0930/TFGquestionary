import gradio as gr
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta

# === 設定 ===
BASE_URL = "https://www.fg.tp.edu.tw"
NEWS_URL = f"{BASE_URL}/news"
DOWNLOAD_FOLDER = "downloads"
DAYS_LIMIT = 90  # 只抓最近幾天的公告

# 建立下載資料夾
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# 小工具：偽裝瀏覽器，避免被擋
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# 小工具：爬一頁公告列表
def fetch_announcements(url):
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    announcements = []
    today = datetime.today()
    cutoff_date = today - timedelta(days=DAYS_LIMIT)

    # 這邊要針對北一女公告的結構調整
    news_items = soup.select(".news-item")  # 假設是這個class，記得確認
    if not news_items:  # 防呆
        print("⚠️ 抓不到公告，可能class名稱要調整")

    for item in news_items:
        title_tag = item.select_one(".title")
        date_tag = item.select_one(".date")
        link_tag = item.select_one("a")

        if not title_tag or not date_tag or not link_tag:
            continue

        title = title_tag.text.strip()
        date_str = date_tag.text.strip()
        link = link_tag["href"]

        # 日期轉換（北一女是西元格式？）
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            print(f"❌ 日期解析失敗: {date_str}")
            continue

        if date >= cutoff_date:
            announcements.append({
                "title": title,
                "date": date,
                "link": BASE_URL + link if not link.startswith("http") else link
            })

    return announcements

# 小工具：下載公告裡的附件
def download_attachments_from_announcement(announcement):
    resp = requests.get(announcement["link"], headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    file_links = []

    for a in soup.select("a"):
        href = a.get("href", "")
        if any(href.lower().endswith(ext) for ext in [".pdf", ".docx", ".doc", ".pptx", ".ppt"]):
            file_url = href if href.startswith("http") else BASE_URL + href
            file_links.append(file_url)

    downloaded_files = []
    for file_url in file_links:
        file_name = os.path.join(DOWNLOAD_FOLDER, os.path.basename(file_url))

        try:
            file_resp = requests.get(file_url, headers=HEADERS)
            with open(file_name, "wb") as f:
                f.write(file_resp.content)
            downloaded_files.append(f"✅ 成功下載: {file_name}")
        except Exception as e:
            downloaded_files.append(f"❌ 下載失敗: {file_url}, 錯誤: {e}")
    
    return downloaded_files

# 使用Gradio介面
def gradio_interface():
    announcements = fetch_announcements(NEWS_URL)
    results = []
    
    for ann in announcements:
        results.append(f"公告標題: {ann['title']} ({ann['date'].strftime('%Y-%m-%d')})")
        download_results = download_attachments_from_announcement(ann)
        results.extend(download_results)
    
    return "\n".join(results)

# 設置 Gradio 介面
iface = gr.Interface(fn=gradio_interface, inputs=None, outputs="text", live=True, title="北一女公告下載器", description="抓取並下載最近90天內的公告及附件")

# 啟動 Gradio 應用
iface.launch()
