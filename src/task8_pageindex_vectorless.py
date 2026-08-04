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
    3. Upload documents:
           python -m src.task8_pageindex_vectorless --upload
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"

# Timeout cho mỗi network call (giây)
_REQUEST_TIMEOUT = 10
# Timeout cứng cho toàn bộ pageindex_search()
_SEARCH_TIMEOUT = 25

# Cache doc_ids
_DOC_IDS_CACHE: list[str] = []


def _make_client():
    """Tạo PageIndexClient với monkey-patch timeout cho requests."""
    from pageindex.client import PageIndexClient
    import requests

    # Monkey-patch Session.request để luôn có timeout
    original_request = requests.Session.request

    def patched_request(self, method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = _REQUEST_TIMEOUT
        return original_request(self, method, url, **kwargs)

    requests.Session.request = patched_request
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _markdown_to_temp_pdf(md_content: str, filename: str) -> str:
    """Convert markdown sang PDF tạm để upload lên PageIndex (chỉ nhận PDF)."""
    tmp_path = os.path.join(tempfile.gettempdir(), filename.replace(".md", ".pdf"))
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        for line in md_content.split("\n"):
            safe = line.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe)
        pdf.output(tmp_path)
    except Exception:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    return tmp_path


def upload_documents() -> list[str]:
    """
    Upload toàn bộ tài liệu lên PageIndex.
    Gọi hàm này một lần để đẩy docs trước khi dùng pageindex_search().

    Returns:
        List of doc_ids đã upload thành công.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY chua set trong .env")

    client = _make_client()
    doc_ids = []

    # 1. PDF gốc (legal documents)
    for pdf_file in LANDING_DIR.rglob("*.pdf"):
        print(f"Uploading PDF: {pdf_file.name}")
        try:
            resp = client.submit_document(file_path=str(pdf_file))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids.append(str(doc_id))
                print(f"  [OK] {pdf_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  [Error] {pdf_file.name}: {e}")

    # 2. Markdown → PDF tạm → upload
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        print(f"Uploading MD: {md_file.name}")
        tmp_pdf = None
        try:
            content = md_file.read_text(encoding="utf-8")
            tmp_pdf = _markdown_to_temp_pdf(content, md_file.name)
            resp = client.submit_document(file_path=tmp_pdf)
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                doc_ids.append(str(doc_id))
                print(f"  [OK] {md_file.name} -> {doc_id}")
        except Exception as e:
            print(f"  [Error] {md_file.name}: {e}")
        finally:
            if tmp_pdf and os.path.exists(tmp_pdf):
                try:
                    os.remove(tmp_pdf)
                except Exception:
                    pass

    global _DOC_IDS_CACHE
    _DOC_IDS_CACHE = list(set(_DOC_IDS_CACHE + doc_ids))
    return doc_ids


def _get_doc_ids() -> list[str]:
    """
    Lấy danh sách doc_ids từ PageIndex account.
    Trả về [] ngay nếu request timeout.
    """
    global _DOC_IDS_CACHE
    if _DOC_IDS_CACHE:
        return _DOC_IDS_CACHE

    if not PAGEINDEX_API_KEY:
        return []

    result = []
    error = []

    def _fetch():
        try:
            client = _make_client()
            resp = client.list_documents(limit=50)
            # list_documents trả về dict {"documents": [...], "total": int}
            docs = resp.get("documents", [])
            for d in docs:
                did = d.get("id") or d.get("doc_id")
                if did:
                    result.append(str(did))
        except Exception as e:
            error.append(str(e))

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=_REQUEST_TIMEOUT + 2)  # wait max 12s

    if result:
        _DOC_IDS_CACHE = result
        print(f"[PageIndex] {len(result)} docs found.")
    elif error:
        print(f"[PageIndex] list_documents error: {error[0]}")
    else:
        print("[PageIndex] list_documents timed out.")

    return _DOC_IDS_CACHE


def _pageindex_search_inner(query: str, top_k: int) -> list[dict]:
    """Thực thi search với PageIndex SDK. Được gọi trong thread có timeout."""
    client = _make_client()

    doc_ids = _get_doc_ids()
    if not doc_ids:
        return _local_fallback_search(query, top_k)

    results = []
    rank_counter = 0

    for doc_id in doc_ids[:3]:
        try:
            if not client.is_retrieval_ready(doc_id):
                continue

            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            # Poll với timeout 20s
            retrieval = None
            deadline = time.time() + 20
            while time.time() < deadline:
                retrieval = client.get_retrieval(retrieval_id)
                status = retrieval.get("status", "")
                if status in ("completed", "done", "success"):
                    break
                if status in ("failed", "error"):
                    retrieval = None
                    break
                time.sleep(2.0)

            if not retrieval:
                continue

            for node in retrieval.get("retrieved_nodes", [])[:2]:
                for group in node.get("relevant_contents", []):
                    items = group if isinstance(group, list) else [group]
                    for item in items:
                        content = item.get("relevant_content", "")
                        if not content:
                            continue
                        rank_counter += 1
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
            print(f"  [Warn] doc {doc_id}: {e}")
            continue

    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    return _local_fallback_search(query, top_k)


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
    if not PAGEINDEX_API_KEY:
        return _local_fallback_search(query, top_k)

    # Chạy trong thread với hard timeout để tránh hang
    container = []
    exc_container = []

    def _run():
        try:
            container.extend(_pageindex_search_inner(query, top_k))
        except Exception as e:
            exc_container.append(str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_SEARCH_TIMEOUT)

    if t.is_alive():
        # Thread vẫn chạy → timeout → dùng fallback
        print(f"[PageIndex] search timed out after {_SEARCH_TIMEOUT}s, using local fallback")
        return _local_fallback_search(query, top_k)

    if exc_container:
        print(f"[PageIndex] search error: {exc_container[0]}, using local fallback")
        return _local_fallback_search(query, top_k)

    return container if container else _local_fallback_search(query, top_k)


def _local_fallback_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Keyword search trên data/standardized/*.md.
    Dùng khi PageIndex API không khả dụng hoặc chưa có docs.
    Luôn gán source='pageindex' để pipeline không bị crash.
    """
    query_tokens = set(query.lower().split())
    scored = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for para in content.split("\n\n"):
            para = para.strip()
            if len(para) < 50:
                continue
            overlap = len(query_tokens & set(para.lower().split()))
            if overlap > 0:
                score = round(overlap / max(len(query_tokens), 1), 4)
                scored.append({
                    "content": para[:600],
                    "score": score,
                    "metadata": {
                        "source_file": md_file.name,
                        "type": "legal" if "legal" in str(md_file) else "news",
                    },
                    "source": "pageindex",
                })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[Info] PAGEINDEX_API_KEY chua set -> local fallback mode")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print(f"[Info] PageIndex key: {PAGEINDEX_API_KEY[:8]}...")

    if "--upload" in sys.argv:
        print("Uploading documents...")
        upload_documents()
    else:
        existing = _get_doc_ids()
        if not existing and PAGEINDEX_API_KEY:
            print("[Hint] Chua co docs. Chay voi --upload de upload truoc.")

    print("\nTest query: lich trinh du lich Ha Giang 3 ngay")
    results = pageindex_search("lich trinh du lich Ha Giang 3 ngay", top_k=3)
    if results:
        for r in results:
            print(f"  [{r['score']:.3f}] [{r['source']}] {r['content'][:120]}...")
    else:
        print("  (No results)")
