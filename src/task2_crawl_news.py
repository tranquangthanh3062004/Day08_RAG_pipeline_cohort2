"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Đã điền 5 URL bài báo thực tế về vụ án nghệ sĩ liên quan đến ma túy
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-chi-dan-bi-dieu-tra-dung-ma-tuy-4730419.html",
    "https://tuoitre.vn/khoi-to-ca-si-chi-dan-nguoi-mau-an-tay-va-tiktoker-co-dong-anh-vi-ma-tuy-20241114144342416.htm",
    "https://thanhnien.vn/ca-si-chi-dan-nguoi-mau-an-tay-bi-bat-vi-to-chuc-su-dung-ma-tuy-18524111415443213.htm",
    "https://dantri.com.vn/phap-luat/ca-si-chi-dan-nguoi-mau-an-tay-bi-bat-vi-to-chuc-su-dung-ma-tuy-20241114150533552.htm",
    "https://vietnamnet.vn/cong-an-tphcm-bat-giam-ca-si-chi-dan-nguoi-mau-an-tay-2341991.html",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    # Khởi tạo và chạy crawler
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # Crawl4AI trả về markdown trong result.markdown
        return {
            "url": url,
            "title": "Bài báo (tiêu đề được lưu trong file markdown)",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
