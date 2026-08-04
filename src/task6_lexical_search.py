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

DATA_DIR = Path(__file__).parent.parent / "data" / "standardized"


def load_corpus() -> list[dict]:
    """
    Nạp corpus cho BM25 — dùng ĐÚNG bộ chunk mà Task 4 đã index vào ChromaDB.

    Vì sao phải chunk thay vì đọc nguyên file? Hai lý do:
      1. BM25 trên 8 file nguyên vẹn gần như vô dụng: chỉ có 8 "document" để xếp hạng,
         lại bị length normalization dìm đúng file dài nhưng liên quan nhất.
      2. Task 7 gộp thứ hạng bằng RRF. Nếu Semantic trả chunk 800 ký tự còn BM25 trả
         nguyên file thì hai danh sách không bao giờ trùng một phần tử nào → RRF chỉ
         xen kẽ 2 danh sách chứ không có "đồng thuận giữa 2 ranker".
    """
    try:
        try:
            from .task4_chunking_indexing import chunk_documents, load_documents
        except ImportError:  # khi chạy trực tiếp: python src/task6_lexical_search.py
            from task4_chunking_indexing import chunk_documents, load_documents

        chunks = chunk_documents(load_documents())
        if chunks:
            return [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    except Exception as e:
        print(f"  [Info] Không dùng được chunker của Task 4 ({e}), tạm đọc nguyên file .md")

    # Fallback: đọc nguyên file (kém hơn, chỉ để module vẫn chạy được khi thiếu Task 4)
    fallback = []
    if DATA_DIR.exists():
        for md_file in DATA_DIR.rglob("*.md"):
            doc_type = "legal" if "legal" in str(md_file) else "news"
            fallback.append({
                "content": md_file.read_text(encoding="utf-8", errors="ignore"),
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return fallback


CORPUS: list[dict] = load_corpus()  # List of {'content': str, 'metadata': dict}

bm25_index = None


def tokenize(text: str) -> list[str]:
    """
    Tách token cho BM25. Dùng regex \\w+ thay vì split() để bỏ dấu câu và ký tự
    markdown — split() để lại token rác kiểu "**bánh", "#", "-" nên từ khoá thật
    không khớp được. Corpus và query bắt buộc dùng chung hàm này.

    Hạn chế đã biết: đây là tách theo âm tiết, chưa phải tách từ tiếng Việt, nên
    "Hà Giang" thành 2 token rời "hà" + "giang" và dễ khớp nhầm với văn bản khác.
    Muốn chính xác hơn thì dùng underthesea.word_tokenize (+5 bonus theo đề bài).
    """
    import re

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


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
    global bm25_index
    if bm25_index is None:
        bm25_index = build_bm25_index(CORPUS)
    
    if bm25_index is None or not CORPUS:
        return []

    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)

    # Get top_k indices
    import numpy as np
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
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
