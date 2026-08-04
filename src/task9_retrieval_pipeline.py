"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from typing import Optional

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import _fusion_key, rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # khi chạy trực tiếp: python src/task9_retrieval_pipeline.py
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import _fusion_key, rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Threshold đã calibrate THẬT trên corpus này (BAAI/bge-m3, 155 chunk du lịch),
# đo bằng calibrate_threshold() ở cuối file — 6 câu liên quan vs 6 câu lạc đề:
#
#     Câu LIÊN QUAN  (Hà Giang, Đà Nẵng, Hải Phòng...) : cosine 0.6577 → 0.7479
#     Câu LẠC ĐỀ     (Kubernetes, cổ phiếu, ký tự rác) : cosine 0.3316 → 0.3937
#                                                        ↑ khe hở rộng 0.26 ↑
# 0.48 nằm giữa khe hở nên tách sạch hai nhóm. Lưu ý giá trị mẫu 0.3 của starter
# KHÔNG dùng được: câu lạc đề vẫn đạt tới 0.3937 > 0.3 nên fallback không bao giờ
# kích hoạt. Đổi corpus hoặc đổi embedding model thì phải đo lại.
SCORE_THRESHOLD = 0.48  # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,         # điểm của reranker (RRF ~0.016–0.032), KHÔNG phải cosine
            'cosine_score': float,  # cosine gốc của đoạn này, dùng để so ngưỡng
            'metadata': dict,
            'source': str           # 'hybrid' hoặc 'pageindex'
        }
        Trả về [] khi cosine gốc dưới ngưỡng VÀ PageIndex cũng không có gì —
        tức là thà không trả lời còn hơn đưa đoạn rác cho LLM.
    """
    # --- Step 1: chạy 2 ranker, lấy dư (top_k * 2) để RRF còn chỗ gộp ------------
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # --- Step 2: CHỐT điểm cosine gốc TRƯỚC khi RRF ghi đè key 'score' ----------
    # Đây là chỗ dễ sai nhất của cả bài: sau rerank_rrf() thì item["score"] không
    # còn là cosine nữa mà là điểm RRF (~0.016–0.032), so với 0.48 thì câu nào cũng
    # "dưới ngưỡng" → pipeline fallback 100% số câu, kể cả câu trả lời tốt.
    best_cosine = dense_results[0]["score"] if dense_results else 0.0

    # --- Step 3: dưới ngưỡng → PageIndex Vectorless ------------------------------
    if best_cosine < score_threshold:
        print(f"  [Fallback] Cosine gốc tốt nhất {best_cosine:.4f} < ngưỡng {score_threshold} "
              f"→ ChromaDB không có đoạn nào thật sự liên quan, chuyển sang PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for item in fallback:
                item["source"] = "pageindex"
                item.setdefault("cosine_score", best_cosine)
            return fallback[:top_k]

        # PageIndex cũng trắng tay → trả RỖNG, không trả kết quả hybrid.
        # Cố tình không "vớt" mấy đoạn cosine ~0.33 vì đó đúng là đoạn rác: đưa cho
        # LLM thì nó sẽ chém gió dựa trên nội dung không liên quan. Trả rỗng để
        # Task 10 trả lời thẳng "không tìm thấy thông tin trong tài liệu".
        print("  [Fallback] PageIndex cũng không có kết quả → trả rỗng "
              "(thà không trả lời còn hơn trả lời sai)")
        return []

    # --- Step 4: gộp 2 danh sách bằng RRF ----------------------------------------
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)

    # Gắn kèm cosine gốc của từng đoạn để Task 10 / eval còn dùng được ngưỡng,
    # vì 'score' từ đây trở đi là điểm của reranker chứ không phải cosine.
    cosine_by_key = {_key(r): r["score"] for r in dense_results}
    for item in merged:
        item["source"] = "hybrid"
        item["cosine_score"] = cosine_by_key.get(_key(item))

    # --- Step 5: rerank ----------------------------------------------------------
    if use_reranking and merged and RERANK_METHOD != "rrf":
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        # RERANK_METHOD == "rrf": Step 4 đã chính là RRF rồi. Chạy rerank_rrf lần nữa
        # trên MỘT danh sách chỉ giữ nguyên thứ tự cũ mà lại ghi đè 'score' thành
        # 1/(60+hạng) — mất thông tin, nên bỏ qua.
        final_results = merged[:top_k]

    return final_results[:top_k]


def _key(item: dict) -> str:
    """Khoá định danh đoạn văn, dùng lại đúng hàm gộp của Task 7 cho nhất quán."""
    return _fusion_key(item)


QUERIES_LIEN_QUAN = [
    "kinh nghiệm du lịch Hà Giang tự túc",
    "lịch trình du lịch Đà Nẵng 3 ngày 2 đêm",
    "ăn gì ở Hải Phòng",
    "đặc sản Bắc Ninh mua làm quà",
    "du lịch Cà Mau có gì",
    "khách sạn homestay ở Huế",
]

QUERIES_LAC_DE = [
    "xyzabc123nonsense",
    "cách cài đặt Kubernetes trên Ubuntu",
    "giá cổ phiếu Tesla hôm nay",
    "công thức tính đạo hàm bậc hai",
    "quy trình xin visa du học Canada",
    "asdkjhasd qwe zxc",
]


def calibrate_threshold(
    lien_quan: Optional[list[str]] = None, lac_de: Optional[list[str]] = None
) -> dict:
    """
    Đo khoảng điểm cosine của câu hỏi liên quan vs câu hỏi lạc đề để chọn ngưỡng.

    Ngưỡng tốt là giá trị nằm trong "khe hở" giữa 2 nhóm: cao hơn mọi câu lạc đề
    và thấp hơn mọi câu liên quan. Chạy lại hàm này mỗi khi đổi corpus hoặc đổi
    embedding model — điểm cosine không so sánh được giữa các model khác nhau.

    Returns:
        {'lien_quan': (min, max), 'lac_de': (min, max), 'de_xuat': float}
    """
    lien_quan = lien_quan or QUERIES_LIEN_QUAN
    lac_de = lac_de or QUERIES_LAC_DE

    def do_nhom(ten: str, queries: list[str]) -> tuple[float, float]:
        diem = []
        print(f"  --- {ten} ---")
        for q in queries:
            r = semantic_search(q, top_k=1)
            s = r[0]["score"] if r else 0.0
            diem.append(s)
            print(f"    {s:.4f}  {q}")
        return (min(diem), max(diem)) if diem else (0.0, 0.0)

    lq = do_nhom("Câu LIÊN QUAN", lien_quan)
    ld = do_nhom("Câu LẠC ĐỀ", lac_de)

    khe_ho = lq[0] - ld[1]
    de_xuat = round((lq[0] + ld[1]) / 2, 2)
    print(f"\n  Liên quan: {lq[0]:.4f} → {lq[1]:.4f}")
    print(f"  Lạc đề   : {ld[0]:.4f} → {ld[1]:.4f}")
    if khe_ho > 0:
        print(f"  Khe hở rộng {khe_ho:.4f} → ngưỡng đề xuất ≈ {de_xuat} "
              f"(đang dùng {SCORE_THRESHOLD})")
    else:
        print("  [Cảnh báo] Hai nhóm chồng lấn — không có ngưỡng nào tách sạch được, "
              "cần thêm dữ liệu hoặc đổi embedding model.")

    return {"lien_quan": lq, "lac_de": ld, "de_xuat": de_xuat}


if __name__ == "__main__":
    print("=" * 70)
    print("Task 9: Retrieval Pipeline — Semantic + BM25 → RRF → PageIndex fallback")
    print(f"  Ngưỡng fallback: cosine gốc < {SCORE_THRESHOLD}")
    print("=" * 70)

    print("\n--- [1] Calibrate ngưỡng trên corpus hiện tại ---")
    calibrate_threshold()

    print("\n--- [2] Chạy pipeline: 2 câu liên quan + 2 câu lạc đề ---")
    for q in QUERIES_LIEN_QUAN[:2] + QUERIES_LAC_DE[:2]:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            cos = r.get("cosine_score")
            cos_txt = f"cosine={cos:.3f}" if isinstance(cos, float) else "cosine=n/a"
            print(f"  {i}. [{r['score']:.4f} | {cos_txt}] [{r['source']}] "
                  f"{r['content'][:70].replace(chr(10), ' ')}...")
