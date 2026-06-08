"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    # Khởi tạo model embedding (cùng model ở Task 4)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    # Khởi tạo client và lấy collection từ ChromaDB
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        collection = chroma_client.get_collection(name="DrugLawDocs")
    except ValueError:
        print("⚠ Lỗi: Collection 'DrugLawDocs' chưa tồn tại. Hãy chắc chắn bạn đã chạy xong Task 4.")
        return []

    # Truy vấn (ChromaDB tự động tính distance tuỳ thuộc config HNSW lúc tạo collection)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted_results = []
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, dist in zip(docs, metadatas, distances):
            # Ở Task 4 ta config hnsw:space là cosine, distance của ChromaDB là (1 - cosine_similarity)
            # Do đó similarity score = 1 - distance
            score = 1.0 - dist
            formatted_results.append({
                "content": doc,
                "score": score,
                "metadata": meta
            })
            
    # Đảm bảo sắp xếp theo score giảm dần
    formatted_results.sort(key=lambda x: x['score'], reverse=True)
    return formatted_results


if __name__ == "__main__":
    import sys
    import io
    # Fix lỗi UnicodeEncodeError khi print tiếng Việt ra console trên Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
