"""
Task 7 — Reranking Module.

Đã implement cả 3 phương pháp, chọn qua tham số method của rerank():

    - "rrf"           — Reciprocal Rank Fusion, MẶC ĐỊNH của bài lab. Gộp thứ hạng từ
                        nhiều ranker (Semantic + BM25). Không cần model, không API key.
    - "mmr"           — Maximal Marginal Relevance. Chọn đoạn vừa liên quan vừa ĐA DẠNG,
                        tránh 5 kết quả nói cùng một ý. Cần embedding (tự lấy từ Task 4).
    - "cross_encoder" — Chấm lại điểm từng cặp (query, đoạn văn) bằng Jina Reranker v2
                        API. Chính xác nhất nhưng chậm nhất và cần JINA_API_KEY.
                        Khác bi-encoder (Task 4/5): bi-encoder mã hoá query và đoạn văn
                        RIÊNG rồi mới so cosine, cross-encoder đưa CẶP vào cùng một lượt
                        nên "đọc" được quan hệ hai bên → chỉ dùng rerank top-N, không
                        dùng quét cả kho.

Dùng cái nào khi nào:
    - Có 2 nguồn kết quả (dense + sparse) và cần gộp công bằng   → rrf
    - Kết quả bị trùng lặp nội dung, cần trải rộng góc nhìn      → mmr
    - Cần độ chính xác cao nhất cho top-N nhỏ, chấp nhận chậm    → cross_encoder

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import math
import os
from typing import Union, cast

from dotenv import load_dotenv

load_dotenv()

# Cross-encoder qua Jina Reranker API — bản v2 multilingual, đọc được tiếng Việt.
# Chạy trên server của Jina nên không phải tải model ~1GB về máy, chỉ cần JINA_API_KEY.
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


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
    if not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu JINA_API_KEY trong .env — cross-encoder dùng Jina Reranker API. "
            "Không có key thì dùng method='rrf' hoặc method='mmr' (đều chạy offline)."
        )

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["results"]
    return [
        {
            **candidates[r["index"]],
            "original_score": candidates[r["index"]].get("score"),
            "score": float(r["relevance_score"]),
            "reranker": "jina",
        }
        for r in results[:top_k]
    ]




def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector: dot(a,b) / (|a| * |b|). Trả 0.0 nếu vector rỗng."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    usable = [c for c in candidates if c.get("embedding")]
    if not usable:
        raise ValueError(
            "rerank_mmr cần mỗi candidate có key 'embedding'. "
            "Dùng rerank(query, candidates, method='mmr') để tự embed giúp."
        )

    selected: list[int] = []
    remaining = list(range(len(usable)))

    for _ in range(min(top_k, len(usable))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # Độ liên quan với câu hỏi
            relevance = cosine_sim(query_embedding, usable[idx]["embedding"])

            # Độ giống nhất với những đoạn ĐÃ chọn — càng giống thì càng bị trừ điểm,
            # đó là cách MMR loại bỏ các đoạn trùng lặp nội dung.
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(usable[idx]["embedding"], usable[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

        item = dict(usable[best_idx])
        item["original_score"] = item.get("score")
        item["score"] = best_score
        item["reranker"] = "mmr"
        usable[best_idx] = item

    return [usable[i] for i in selected]


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
        # MMR cần vector: embed query và những candidate chưa có sẵn 'embedding',
        # dùng đúng embed_texts() của Task 4 để cùng không gian vector.
        try:
            from .task4_chunking_indexing import embed_texts
        except ImportError:  # khi chạy trực tiếp: python src/task7_reranking.py
            from task4_chunking_indexing import embed_texts

        missing = [c for c in flat if not c.get("embedding")]
        if missing:
            vectors = embed_texts([c["content"] for c in missing])
            for c, vec in zip(missing, vectors):
                c["embedding"] = vec

        query_embedding = embed_texts([query])[0]
        return rerank_mmr(query_embedding, flat, top_k=top_k)
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


def demo_mmr(query: str, top_k: int = 5) -> None:
    """So sánh top-k của Semantic thuần và của MMR để thấy tác dụng đa dạng hoá."""
    try:
        from .task5_semantic_search import semantic_search
    except ImportError:
        from task5_semantic_search import semantic_search

    candidates = semantic_search(query, top_k=15)
    if not candidates:
        print("  [Info] Không có kết quả semantic để chạy MMR.")
        return

    def mo_ta(items):
        return [f"{c['metadata'].get('source', '?')}#{c['metadata'].get('chunk_index', '?')}"
                for c in items]

    print(f"  Semantic top-{top_k} (chỉ xét liên quan): {mo_ta(candidates[:top_k])}")

    # Embed 1 lần rồi tái dùng cho mọi λ
    query_vec = _query_vector(query)
    with_vectors = rerank(query, [dict(c) for c in candidates], top_k=len(candidates), method="mmr")
    by_key = {(c["metadata"].get("source"), c["metadata"].get("chunk_index")): c
              for c in with_vectors}
    pool = [dict(by_key[(c["metadata"].get("source"), c["metadata"].get("chunk_index"))])
            for c in candidates]

    for lam in (0.7, 0.3):
        picked = rerank_mmr(query_vec, [dict(c) for c in pool], top_k=top_k, lambda_param=lam)
        print(f"  MMR λ={lam} ({'ưu tiên liên quan' if lam > 0.5 else 'ưu tiên đa dạng'}): "
              f"{mo_ta(picked)}")


def _query_vector(query: str) -> list[float]:
    """Embed query bằng đúng hàm của Task 4."""
    try:
        from .task4_chunking_indexing import embed_texts
    except ImportError:
        from task4_chunking_indexing import embed_texts
    return embed_texts([query])[0]


def demo_cross_encoder(query: str, top_k: int = 3) -> None:
    """Chấm lại top ứng viên bằng cross-encoder (Jina API hoặc model local)."""
    try:
        from .task5_semantic_search import semantic_search
    except ImportError:
        from task5_semantic_search import semantic_search

    candidates = semantic_search(query, top_k=8)
    if not candidates:
        print("  [Info] Không có kết quả semantic để rerank.")
        return

    print(f"  Model: {JINA_RERANK_MODEL} (Jina Reranker API)")
    reranked = rerank_cross_encoder(query, candidates, top_k=top_k)
    for i, r in enumerate(reranked, 1):
        m = r["metadata"]
        print(f"    {i}. [ce={r['score']:.4f} | cosine gốc={r['original_score']:.3f}] "
              f"{m.get('source', '?')}#{m.get('chunk_index', '?')} — "
              f"{r['content'][:60].replace(chr(10), ' ')}...")


DEMO_QUERY = "kinh nghiệm du lịch Hà Giang tự túc"


if __name__ == "__main__":
    print("=" * 70)
    print(f"Task 7: Reranking — RRF (k={RRF_K}) | MMR | Cross-encoder")
    print("  RRF(d) = Σ 1 / (k + rank_r(d))")
    print("=" * 70)

    print("\n--- [1] Kiểm tra công thức RRF cân bằng Semantic vs BM25 ---")
    ok = verify_rrf_balance()
    print(f"\n  → {'TẤT CẢ 4 PHÉP THỬ ĐẠT' if ok else 'CÓ PHÉP THỬ KHÔNG ĐẠT'}")

    print("\n--- [2] RRF trên dữ liệu thật (Task 5 + Task 6) ---")
    try:
        demo_real_data(DEMO_QUERY)
    except Exception as e:
        print(f"  [Info] Bỏ qua demo dữ liệu thật: {e}")

    print("\n--- [3] MMR: liên quan vs đa dạng ---")
    try:
        demo_mmr(DEMO_QUERY)
    except Exception as e:
        print(f"  [Info] Bỏ qua demo MMR: {e}")

    print("\n--- [4] Cross-encoder: chấm lại từng cặp (query, đoạn văn) ---")
    try:
        demo_cross_encoder(DEMO_QUERY)
    except Exception as e:
        print(f"  [Info] Bỏ qua demo cross-encoder: {e}")
