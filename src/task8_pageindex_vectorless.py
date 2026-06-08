"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex import PageIndexClient
    from markdown_pdf import MarkdownPdf, Section
    
    pdf_path = STANDARDIZED_DIR / "all_documents.pdf"
    pdf = MarkdownPdf(toc_level=2)
    
    import re
    print("Đang gộp các file Markdown thành 1 file PDF duy nhất...")
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = f"# {md_file.name}\n\n" + md_file.read_text(encoding="utf-8")
        # Xóa các link markdown bị hỏng (ví dụ [1](#_ftn1)) để tránh lỗi của markdown-pdf
        content = re.sub(r'\[(.*?)\]\([^)]+\)', r'\1', content)
        pdf.add_section(Section(content))
        
    pdf.save(str(pdf_path))
    print(f"✓ Đã tạo file {pdf_path.name}")
    
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    print("Đang upload file PDF lên PageIndex...")
    response = client.submit_document(str(pdf_path))
    
    doc_id = response.get("doc_id")
    print(f"✓ Upload thành công! Doc ID: {doc_id}")
    
    # Lưu doc_id ra file để dùng cho hàm search
    doc_id_file = STANDARDIZED_DIR / "pageindex_doc_id.txt"
    doc_id_file.write_text(doc_id)
    return doc_id


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient
    import time
    
    doc_id_file = STANDARDIZED_DIR / "pageindex_doc_id.txt"
    if not doc_id_file.exists():
        print("⚠ Lỗi: Chưa tìm thấy doc_id. Hãy chạy hàm upload_documents() trước.")
        return []
        
    doc_id = doc_id_file.read_text().strip()
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    print(f"Gửi truy vấn '{query}' tới PageIndex...")
    response = client.submit_query(doc_id=doc_id, query=query)
    retrieval_id = response.get("retrieval_id")
    
    # Polling chờ kết quả
    for _ in range(15): # Đợi tối đa 30s
        time.sleep(2)
        status_res = client.get_retrieval(retrieval_id)
        if status_res.get("status") == "completed":
            break
            
    results = status_res.get("results", [])
    
    formatted = []
    for r in results[:top_k]:
        formatted.append({
            "content": r.get("content", r.get("text", "")),
            "score": r.get("score", 1.0),
            "metadata": {"source": "pageindex_api"},
            "source": "pageindex"
        })
    return formatted


if __name__ == "__main__":
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        print("⚠ Hãy set PAGEINDEX_API_KEY hợp lệ trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
