"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAMES = ["holiday_support_docs", "ecommerce_support_docs", "travel_support_docs"]

_MODEL_CACHE = None


def get_embedding_model():
    """Tải và cache embedding model nhẹ SentenceTransformer."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Mặc định phải trùng model của Task 4 (BAAI/bge-m3, 1024 chiều), nếu không
            # vector query sẽ lệch số chiều với vector đã index trong chroma_db/.
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            _MODEL_CACHE = SentenceTransformer(model_name)
        except Exception as e:
            print(f"  [Info] Local SentenceTransformer unavailable ({e}), using fallback embedder...")
            _MODEL_CACHE = False
    return _MODEL_CACHE


def embed_query_text(query: str, dim: int = 384) -> list[float]:
    """Tạo vector embedding cho query string."""
    # Ưu tiên dùng đúng hàm embed của Task 4: query và document được embed bằng cùng
    # model / cùng provider (EMBEDDING_PROVIDER trong .env) nên luôn cùng không gian
    # vector — đổi provider ở Task 4 là Task 5 tự đổi theo, không phải sửa 2 nơi.
    try:
        try:
            from .task4_chunking_indexing import embed_texts
        except ImportError:  # khi chạy trực tiếp: python src/task5_semantic_search.py
            from task4_chunking_indexing import embed_texts
        return embed_texts([query])[0]
    except Exception:
        pass

    # Try sentence-transformers first
    model = get_embedding_model()
    if model:
        try:
            return model.encode(query).tolist()
        except Exception:
            pass

    # Try OpenAI / OpenRouter if API key present
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
        except Exception:
            pass

    # Deterministic fallback vector if local PyTorch DLL is blocked
    return hash_vector(query, dim)


def hash_vector(text: str, dim: int = 384) -> list[float]:
    """Fallback deterministic vector based on text tokens."""
    import hashlib
    import math
    
    vec = [0.0] * dim
    words = text.lower().split()
    if not words:
        return vec

    for word in words:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        val = 1.0 if (h % 2 == 0) else -1.0
        vec[idx] += val

    # Normalize L2
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def generate_hypothetical_document(query: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings):
    Sinh câu trả lời/đoạn văn giả định từ LLM để dùng làm vector query thay cho câu hỏi thô.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return query

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        prompt = f"Hãy viết một đoạn văn ngắn (2-3 câu) trả lời hoặc giải thích chi tiết cho câu hỏi du lịch sau: {query}"
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return query


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Có sử dụng HyDE (Hypothetical Document Embeddings) hay không

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # 1. Nếu chưa có thư mục chroma_db hoặc không tồn tại, trả về []
    if not CHROMA_DIR.exists():
        return []

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Tìm collection phù hợp
        existing_collections = [c.name for c in client.list_collections()]
        if not existing_collections:
            return []

        collection = None
        for name in COLLECTION_NAMES:
            if name in existing_collections:
                collection = client.get_collection(name)
                break
        
        if collection is None:
            collection = client.get_collection(existing_collections[0])

        if collection is None or collection.count() == 0:
            return []

        # 2. Xử lý HyDE nếu được kích hoạt
        search_text = query
        if use_hyde:
            search_text = generate_hypothetical_document(query)

        # 3. Embed query
        # Lấy sample embedding từ collection để biết dimension thực tế
        sample_item = collection.get(limit=1, include=["embeddings"])
        dim = 384
        # Dùng len() chứ không dùng truthiness: chromadb 1.x trả embeddings là numpy
        # array, `if array` sẽ ném ValueError "truth value of an array is ambiguous".
        sample_embeddings = sample_item.get("embeddings") if sample_item else None
        if sample_embeddings is not None and len(sample_embeddings) > 0:
            dim = len(sample_embeddings[0])

        query_vector = embed_query_text(search_text, dim=dim)

        # 4. Query vector store
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                # Cosine distance -> similarity score
                score = max(0.0, 1.0 - float(dist))
                output.append({
                    "content": doc,
                    "score": round(score, 4),
                    "metadata": meta or {}
                })

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:top_k]

    except Exception as e:
        print(f"  [Info] semantic_search fallback: {e}")
        return []


if __name__ == "__main__":
    test_query = "Gợi ý lịch trình du lịch Hà Giang 3 ngày 2 đêm"
    print(f"Querying: {test_query}")
    results = semantic_search(test_query, top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
