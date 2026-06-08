"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

def load_corpus() -> list[dict]:
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    try:
        chroma_client = chromadb.PersistentClient(path=str(db_path))
        collection = chroma_client.get_collection(name="DrugLawDocs")
        all_docs = collection.get()
        corpus = []
        if all_docs['documents']:
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                corpus.append({'content': doc, 'metadata': meta})
        return corpus
    except ValueError:
        print("⚠ Lỗi: Collection 'DrugLawDocs' chưa tồn tại. Hãy chắc chắn bạn đã chạy xong Task 4.")
        return []

CORPUS: list[dict] = load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus: return None
    # Tokenize đơn giản bằng cách viết thường và tách theo khoảng trắng
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Khởi tạo Index một lần khi load module
BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not BM25_INDEX or not CORPUS:
        return []

    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)
    
    # Lấy top_k kết quả cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    import sys
    import io
    # Fix lỗi UnicodeEncodeError khi print tiếng Việt ra console trên Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
