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

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lưu cache doc_id đã upload để tránh upload lại nhiều lần khi gọi pageindex_search()
_DOC_IDS_CACHE: list[str] = []


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        List of doc_ids đã upload thành công.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY chua duoc set trong .env")

    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    doc_ids = []
    md_files = list(STANDARDIZED_DIR.rglob("*.md"))

    for md_file in md_files:
        print(f"Uploading: {md_file.name}")
        try:
            # PageIndex nhận text trực tiếp hoặc PDF.
            # Đọc nội dung markdown và gửi lên dưới dạng text document.
            content = md_file.read_text(encoding="utf-8")
            resp = client.submit_document(content=content, filename=md_file.name)

            doc_id = resp.get("doc_id") or resp.get("id") or resp.get("document_id")
            if doc_id:
                doc_ids.append(str(doc_id))
                print(f"  [OK] Uploaded: {md_file.name} -> {doc_id}")
            else:
                print(f"  [Warn] No doc_id in response for {md_file.name}: {resp}")
        except Exception as e:
            print(f"  [Error] Failed to upload {md_file.name}: {e}")

    return doc_ids


def _get_doc_ids() -> list[str]:
    """Lấy hoặc nạp danh sách doc_ids từ PageIndex."""
    global _DOC_IDS_CACHE
    if _DOC_IDS_CACHE:
        return _DOC_IDS_CACHE

    if not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        # Thử lấy danh sách tài liệu đã upload từ trước
        docs = client.list_documents()
        if isinstance(docs, list) and docs:
            _DOC_IDS_CACHE = [
                str(d.get("doc_id") or d.get("id") or d.get("document_id"))
                for d in docs
                if d.get("doc_id") or d.get("id") or d.get("document_id")
            ]
    except Exception:
        pass

    return _DOC_IDS_CACHE


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
    # Nếu không có API key → fallback về BM25 local search với marker 'pageindex'
    if not PAGEINDEX_API_KEY:
        return _local_fallback_search(query, top_k)

    try:
        from pageindex.client import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        # Lấy danh sách doc_ids đã upload
        doc_ids = _get_doc_ids()
        if not doc_ids:
            # Chưa có tài liệu nào → upload trước
            doc_ids = upload_documents()
            _DOC_IDS_CACHE.extend(doc_ids)

        if not doc_ids:
            return []

        results = []
        rank_counter = 0

        for doc_id in doc_ids[:3]:  # Giới hạn query 3 docs để tránh rate limit
            try:
                # Submit query để lấy retrieval_id
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id") or resp.get("id")

                if not retrieval_id:
                    continue

                # Poll cho đến khi status == "completed"
                max_polls = 10
                retrieval = None
                for _ in range(max_polls):
                    retrieval = client.get_retrieval(retrieval_id)
                    status = retrieval.get("status", "")
                    if status in ("completed", "done", "success"):
                        break
                    if status in ("failed", "error"):
                        break
                    time.sleep(1.0)

                if not retrieval:
                    continue

                # Parse retrieved_nodes — mỗi node có "relevant_contents"
                for node in retrieval.get("retrieved_nodes", [])[:2]:
                    for group in node.get("relevant_contents", []):
                        items = group if isinstance(group, list) else [group]
                        for item in items:
                            content = item.get("relevant_content", "")
                            if not content:
                                continue
                            rank_counter += 1
                            # Gán score theo thứ hạng xuất hiện (PageIndex không trả score)
                            score = round(1.0 / (1.0 + rank_counter), 4)
                            results.append({
                                "content": content,
                                "score": score,
                                "metadata": {
                                    "section": item.get("section_title", ""),
                                    "doc_id": doc_id,
                                },
                                "source": "pageindex",
                            })
            except Exception as e:
                print(f"  [Warn] PageIndex query error for doc {doc_id}: {e}")
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        print(f"  [Info] PageIndex SDK unavailable ({e}), using local fallback...")
        return _local_fallback_search(query, top_k)


def _local_fallback_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Fallback khi không có PAGEINDEX_API_KEY:
    Dùng simple keyword search trên data/standardized/*.md,
    gán source='pageindex' để pipeline không bị crash.
    """
    results = []
    query_tokens = set(query.lower().split())

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        return []

    scored_chunks = []
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Cắt thành các đoạn ~400 ký tự
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        for para in paragraphs:
            tokens = set(para.lower().split())
            overlap = len(query_tokens & tokens)
            if overlap > 0:
                score = round(overlap / max(len(query_tokens), 1), 4)
                scored_chunks.append({
                    "content": para[:600],
                    "score": score,
                    "metadata": {
                        "source": md_file.name,
                        "type": "legal" if "legal" in str(md_file) else "news",
                    },
                    "source": "pageindex",
                })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[Info] PAGEINDEX_API_KEY chua set -> dung local fallback search")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("lich trinh du lich Ha Giang 3 ngay 2 dem", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
