import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
import time

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

    for file_url in file_links:
        file_name = os.path.join(DOWNLOAD_FOLDER, os.path.basename(file_url))

        try:
            file_resp = requests.get(file_url, headers=HEADERS)
            with open(file_name, "wb") as f:
                f.write(file_resp.content)
            print(f"✅ 成功下載: {file_name}")
        except Exception as e:
            print(f"❌ 下載失敗: {file_url}, 錯誤: {e}")

# === 主流程 ===
def main():
    print(f"🚀 開始從北一女中校網抓最近 {DAYS_LIMIT} 天內有附件ㄉ公告...")

    announcements = fetch_announcements(NEWS_URL)

    print(f"🎯 總共找到 {len(announcements)} 筆公告")

    for ann in announcements:
        print(f"🔍 處理公告：{ann['title']} ({ann['date'].strftime('%Y-%m-%d')})")
        download_attachments_from_announcement(ann)
        time.sleep(1)  # 禮貌一點，慢慢來避免被封XD

    print("🏁 全部下載完成ㄌ！")

if __name__ == "__main__":
    main()
