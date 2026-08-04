"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from typing import Union, cast


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


RRF_K = 60  # hằng số làm mượt, theo paper Cormack et al. 2009


def _fusion_key(item: dict) -> str:
    """
    Khoá định danh 1 đoạn văn khi gộp giữa các ranker.

    Ưu tiên (source, chunk_index) trong metadata: hai ranker có thể trả về cùng một
    chunk nhưng chuỗi content lệch nhau vài ký tự khoảng trắng, dùng metadata thì
    vẫn nhận ra là một. Không có metadata thì mới rơi về content đã chuẩn hoá.
    """
    meta = item.get("metadata") or {}
    source = meta.get("source")
    chunk_index = meta.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"{source}#{chunk_index}"
    if source is not None:
        return f"{source}#full"
    return " ".join(item.get("content", "").split())[:300]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = RRF_K
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Vì sao phải dùng RRF? Điểm cosine của Semantic Search nằm trong [0, 1], còn điểm
    BM25 là điểm thô không chặn trên (0 → 20+). Cộng thẳng hai loại điểm thì BM25 sẽ
    át hoàn toàn Semantic. RRF bỏ qua giá trị điểm, chỉ lấy THỨ HẠNG trong từng danh
    sách, nên hai ranker đóng góp cân bằng bất kể thang điểm của chúng.

    Vì sao k=60? k càng lớn thì chênh lệch giữa các thứ hạng liền nhau càng nhỏ:
        k=1  → rank 1 = 0.5000, rank 2 = 0.3333  (hạng 1 áp đảo)
        k=60 → rank 1 = 0.0164, rank 2 = 0.0161  (gần bằng nhau)
    Nhờ vậy một đoạn được CẢ HAI ranker xếp hạng trung bình sẽ thắng đoạn chỉ được
    MỘT ranker xếp hạng 1 — đúng tinh thần "đồng thuận giữa 2 ranker" của RRF.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending. Mỗi item giữ nguyên
        content/metadata, có thêm:
            'score'          — điểm RRF sau khi gộp
            'original_score' — điểm gốc của ranker xếp nó cao nhất
            'ranks'          — thứ hạng ở từng ranker, vd {0: 3, 1: 1} (ranker 0 hạng 3)
    """
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}
    ranks_map: dict[str, dict[int, int]] = {}

    for ranker_idx, ranked_list in enumerate(ranked_lists):
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, start=1):
            key = _fusion_key(item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks_map.setdefault(key, {})[ranker_idx] = rank

            # Giữ bản ghi của ranker xếp đoạn này cao nhất (rank nhỏ nhất)
            previous = item_map.get(key)
            if previous is None or rank < previous["_best_rank"]:
                item_map[key] = {**item, "_best_rank": rank}

    ordered = sorted(
        rrf_scores.items(),
        # tie-break: điểm bằng nhau thì ưu tiên đoạn được nhiều ranker cùng chọn
        key=lambda kv: (kv[1], len(ranks_map[kv[0]])),
        reverse=True,
    )

    results = []
    for key, score in ordered[:top_k]:
        item = dict(item_map[key])
        item.pop("_best_rank", None)
        item["original_score"] = item.get("score")
        item["score"] = score
        item["ranks"] = ranks_map[key]
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: Union[list[dict], list[list[dict]]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    # RRF chỉ dùng thứ hạng nên không cần `query`. Nhận cả 2 dạng input:
    #   - list[list[dict]] : nhiều ranked list (Semantic + BM25) → gộp thật sự
    #   - list[dict]       : 1 danh sách duy nhất → coi như 1 ranker (giữ thứ tự)
    if isinstance(candidates[0], list):
        ranked_lists = cast(list[list[dict]], candidates)
        flat = [item for sub in ranked_lists for item in sub]
    else:
        flat = cast(list[dict], candidates)
        ranked_lists = [flat]

    if method == "cross_encoder":
        return rerank_cross_encoder(query, flat, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        return rerank_rrf(ranked_lists, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


# =============================================================================
# Kiểm tra công thức RRF (k=60) — Checkpoint 3
# =============================================================================

def verify_rrf_balance(k: int = RRF_K) -> bool:
    """
    Kiểm chứng RRF với k=60 thật sự cân bằng giữa 2 ranker (Semantic vs BM25).

    Chạy 4 phép thử trên dữ liệu giả có thứ hạng biết trước, in kết quả và trả về
    True nếu cả 4 đều đạt. Dùng để trả lời câu hỏi của coach ở Checkpoint 3.
    """
    def fake(name: str, n: int) -> list[dict]:
        return [
            {"content": f"{name}-{i}", "score": 0.0, "metadata": {"source": name, "chunk_index": i}}
            for i in range(1, n + 1)
        ]

    semantic = fake("SEM", 5)
    bm25 = fake("BM25", 5)
    all_passed = True

    # --- Phép thử 1: đối xứng — đổi thứ tự 2 ranker không đổi kết quả -----------
    a = [r["content"] for r in rerank_rrf([semantic, bm25], top_k=10, k=k)]
    b = [r["content"] for r in rerank_rrf([bm25, semantic], top_k=10, k=k)]
    passed = sorted(a) == sorted(b)
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Đối xứng: đổi thứ tự Semantic/BM25 ra cùng tập kết quả")

    # --- Phép thử 2: không thiên vị — 2 list rời nhau thì chia đôi top-k --------
    fused = rerank_rrf([semantic, bm25], top_k=4, k=k)
    n_sem = sum(1 for r in fused if r["content"].startswith("SEM"))
    n_bm25 = sum(1 for r in fused if r["content"].startswith("BM25"))
    passed = n_sem == n_bm25 == 2
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Không thiên vị: top-4 gồm {n_sem} Semantic + {n_bm25} BM25 "
          f"(2 danh sách rời nhau → xen kẽ 50/50)")

    # --- Phép thử 3: đồng thuận thắng đơn lẻ -----------------------------------
    # "chung" đứng hạng 3 ở CẢ HAI ranker; "sem_top" đứng hạng 1 nhưng chỉ 1 ranker.
    chung = {"content": "CHUNG", "score": 0.0, "metadata": {"source": "x", "chunk_index": 99}}
    list_a = [dict(semantic[0]), dict(semantic[1]), chung]
    list_b = [dict(bm25[0]), dict(bm25[1]), chung]
    fused = rerank_rrf([list_a, list_b], top_k=5, k=k)
    diem_chung = 2 / (k + 3)
    diem_don = 1 / (k + 1)
    passed = fused[0]["content"] == "CHUNG" and diem_chung > diem_don
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Đồng thuận thắng: hạng 3 ở cả 2 ranker = {diem_chung:.5f} "
          f"> hạng 1 ở 1 ranker = {diem_don:.5f} → top-1 là '{fused[0]['content']}'")

    # --- Phép thử 4: vì sao k=60 chứ không phải k nhỏ ---------------------------
    gap_60 = 1 / (60 + 1) - 1 / (60 + 2)
    gap_1 = 1 / (1 + 1) - 1 / (1 + 2)
    passed = gap_60 < gap_1 / 10
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] k=60 làm mượt: chênh lệch hạng 1↔2 chỉ {gap_60:.5f} "
          f"(k=1 thì tới {gap_1:.5f}) → không ranker nào áp đảo chỉ vì giành được hạng 1")

    return bool(all_passed)


def demo_real_data(query: str, top_k: int = 5) -> None:
    """Gộp kết quả THẬT từ Task 5 (Semantic) và Task 6 (BM25) bằng RRF."""
    try:
        from .task5_semantic_search import semantic_search
        from .task6_lexical_search import lexical_search
    except ImportError:  # khi chạy trực tiếp: python src/task7_reranking.py
        from task5_semantic_search import semantic_search
        from task6_lexical_search import lexical_search

    semantic = semantic_search(query, top_k=10)
    bm25 = lexical_search(query, top_k=10)
    print(f"\n  Semantic trả {len(semantic)} kết quả (score cosine [0,1]): "
          f"{[round(r['score'], 3) for r in semantic[:3]]} ...")
    print(f"  BM25     trả {len(bm25)} kết quả (score thô không chặn trên): "
          f"{[round(r['score'], 2) for r in bm25[:3]]} ...")

    keys_sem = {_fusion_key(r) for r in semantic}
    keys_bm25 = {_fusion_key(r) for r in bm25}
    overlap = keys_sem & keys_bm25
    print(f"  Số đoạn được CẢ HAI ranker cùng trả về: {len(overlap)}")
    if not overlap and semantic and bm25:
        print("  [CẢNH BÁO] Không có đoạn nào trùng → 2 ranker đang làm việc trên đơn vị")
        print("             khác nhau (Task 6 index nguyên file, Task 5 trả chunk 800 ký tự).")
        print("             RRF vẫn chạy nhưng chỉ xen kẽ 2 danh sách, không có 'đồng thuận'.")

    fused = rerank_rrf([semantic, bm25], top_k=top_k)
    print(f"\n  Top-{top_k} sau RRF (k={RRF_K}):")
    for i, r in enumerate(fused, 1):
        nguon = {0: "Semantic", 1: "BM25"}
        tu = " + ".join(f"{nguon[idx]}#{rank}" for idx, rank in sorted(r["ranks"].items()))
        print(f"    {i}. [rrf={r['score']:.5f}] ({tu}) {r['metadata'].get('source', '?')} — "
              f"{r['content'][:70].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    print("=" * 70)
    print(f"Task 7: Reranking — Reciprocal Rank Fusion (k={RRF_K})")
    print("  RRF(d) = Σ 1 / (k + rank_r(d))")
    print("=" * 70)

    print("\n--- Kiểm tra công thức RRF cân bằng Semantic vs BM25 ---")
    ok = verify_rrf_balance()
    print(f"\n  → {'TẤT CẢ 4 PHÉP THỬ ĐẠT' if ok else 'CÓ PHÉP THỬ KHÔNG ĐẠT'}")

    print("\n--- Gộp kết quả thật từ Task 5 + Task 6 ---")
    try:
        demo_real_data("kinh nghiệm du lịch Hà Giang tự túc")
    except Exception as e:
        print(f"  [Info] Bỏ qua demo dữ liệu thật: {e}")
