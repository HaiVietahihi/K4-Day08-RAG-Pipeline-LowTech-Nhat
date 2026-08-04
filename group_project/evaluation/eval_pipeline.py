"""
RAG Evaluation Pipeline — Heuristic Edition.

Đánh giá RAG pipeline với 4 metrics không cần LLM API (tránh rate limit):
  - Faithfulness:        keyword overlap giữa answer và retrieved context
  - Answer Relevance:   keyword overlap giữa answer và question
  - Context Recall:     overlap giữa retrieved context và expected context keywords
  - Context Precision:  % chunks có đóng góp thực sự vào answer

So sánh A/B:
  Config A: Hybrid Search (Semantic + BM25) + RRF Reranking
  Config B: Dense-only    (Semantic Search only, không reranking)

Chạy:
    python -m group_project.evaluation.eval_pipeline
"""

import json
import sys
import time
from pathlib import Path

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# =============================================================================
# HELPERS: Tokenization & Overlap
# =============================================================================

def _tokenize(text: str) -> set[str]:
    """Simple tokenizer: lowercase, split on whitespace/punct, remove stopwords."""
    import re
    STOP_VI = {"là", "và", "có", "của", "trong", "để", "với", "không", "được",
               "những", "các", "tôi", "bạn", "này", "đó", "một", "hay", "nào",
               "như", "khi", "sẽ", "thì", "từ", "ra", "đi", "lên", "về",
               "cho", "theo", "tại", "ở", "vào", "qua", "nên"}
    tokens = set(re.findall(r"[a-zàáảãạăắặẳẵâầấậẩẫèéẹẻẽêềếệểễìíịỉĩ"
                            r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+",
                            text.lower()))
    return tokens - STOP_VI


def _overlap_ratio(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a)


# =============================================================================
# METRICS (Heuristic, no LLM calls)
# =============================================================================

def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: seberapa banyak answer có thể disproved bởi context.
    Heuristic: ratio token trong answer tìm thấy trong bất kỳ context nào.
    """
    if not answer or not contexts:
        return 0.0
    answer_tokens = _tokenize(answer)
    context_tokens = set()
    for ctx in contexts:
        context_tokens |= _tokenize(ctx)
    return _overlap_ratio(answer_tokens, context_tokens)


def compute_answer_relevance(answer: str, question: str) -> float:
    """
    Answer Relevance: answer có trả lời đúng câu hỏi không.
    Heuristic: keyword overlap giữa answer và question.
    """
    if not answer or not question:
        return 0.0
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    return _overlap_ratio(q_tokens, a_tokens)


def compute_context_recall(contexts: list[str], expected_context: str) -> float:
    """
    Context Recall: retriever có lấy về đúng context cần thiết không.
    Heuristic: keyword overlap giữa expected_context và retrieved contexts.
    """
    if not contexts or not expected_context:
        return 0.0
    expected_tokens = _tokenize(expected_context)
    retrieved_tokens = set()
    for ctx in contexts:
        retrieved_tokens |= _tokenize(ctx)
    return _overlap_ratio(expected_tokens, retrieved_tokens)


def compute_context_precision(answer: str, contexts: list[str]) -> float:
    """
    Context Precision: trong các chunks lấy về, bao nhiêu % thực sự hữu ích.
    Heuristic: tỉ lệ chunks có overlap với answer.
    """
    if not contexts or not answer:
        return 0.0
    answer_tokens = _tokenize(answer)
    useful = sum(1 for ctx in contexts if _overlap_ratio(_tokenize(ctx), answer_tokens) > 0.05)
    return useful / len(contexts)


# =============================================================================
# RAG PIPELINE WRAPPERS
# =============================================================================

def run_hybrid_pipeline(question: str, top_k: int = 5) -> dict:
    """Config A: Hybrid (Semantic + BM25) + RRF Reranking."""
    try:
        from src.task10_generation import generate_with_citation
        result = generate_with_citation(question, top_k=top_k)
        return result
    except Exception as e:
        return {"answer": f"[Error] {e}", "sources": [], "retrieval_source": "error"}


def run_dense_only_pipeline(question: str, top_k: int = 5) -> dict:
    """Config B: Dense-only (Semantic Search), no reranking."""
    try:
        from src.task5_semantic_search import semantic_search
        chunks = semantic_search(question, top_k=top_k)
        if not chunks:
            return {"answer": "Không tìm thấy thông tin.", "sources": [], "retrieval_source": "dense"}
        # Format context
        ctx_parts = []
        for i, c in enumerate(chunks, 1):
            src = c.get("metadata", {}).get("source", f"Doc {i}")
            ctx_parts.append(f"[{i}] {src}:\n{c['content']}")
        context = "\n\n".join(ctx_parts)

        # Simple LLM call (same as hybrid but dense-only context)
        from src.task10_generation import SYSTEM_PROMPT, TEMPERATURE, TOP_P, LLM_MODEL, LLM_MODEL_FALLBACK
        from openai import OpenAI
        import os
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=TEMPERATURE, top_p=TOP_P,
            )
            answer = resp.choices[0].message.content
        except Exception:
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL_FALLBACK,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}],
                    temperature=TEMPERATURE, top_p=TOP_P,
                )
                answer = resp.choices[0].message.content
            except Exception as e2:
                answer = f"[LLM Error] {e2}"

        return {"answer": answer, "sources": chunks, "retrieval_source": "dense"}
    except Exception as e:
        return {"answer": f"[Error] {e}", "sources": [], "retrieval_source": "error"}


# =============================================================================
# EVALUATE ONE CONFIG
# =============================================================================

def evaluate_config(config_name: str, pipeline_fn, golden_dataset: list[dict],
                    delay: float = 1.5) -> dict:
    """
    Chạy evaluation cho một config.

    Args:
        config_name:    Tên config (A/B)
        pipeline_fn:    Hàm nhận question, trả về dict {answer, sources}
        golden_dataset: List Q&A pairs
        delay:          Giây chờ giữa các câu (tránh rate limit)

    Returns:
        dict với per-question scores và tổng hợp
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")

    per_question = []

    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_context = item.get("expected_context", "")

        print(f"  [{i:02d}/{len(golden_dataset)}] {question[:55]}...")

        # Chạy pipeline
        try:
            result = pipeline_fn(question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            contexts = [s.get("content", "") for s in sources]
        except Exception as e:
            print(f"    [Pipeline Error] {e}")
            answer, contexts = "", []

        # Tính metrics
        faithfulness      = compute_faithfulness(answer, contexts)
        answer_relevance  = compute_answer_relevance(answer, question)
        context_recall    = compute_context_recall(contexts, expected_context)
        context_precision = compute_context_precision(answer, contexts)

        row = {
            "question": question,
            "answer_snippet": answer[:120] if answer else "(no answer)",
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "context_recall": round(context_recall, 4),
            "context_precision": round(context_precision, 4),
            "n_sources": len(sources),
        }
        per_question.append(row)

        print(f"    faithful={faithfulness:.3f} | relevance={answer_relevance:.3f} | "
              f"recall={context_recall:.3f} | precision={context_precision:.3f}")

        if delay > 0 and i < len(golden_dataset):
            time.sleep(delay)

    # Tổng hợp
    def _avg(key): return round(sum(r[key] for r in per_question) / len(per_question), 4)

    summary = {
        "config": config_name,
        "n_questions": len(per_question),
        "faithfulness":       _avg("faithfulness"),
        "answer_relevance":   _avg("answer_relevance"),
        "context_recall":     _avg("context_recall"),
        "context_precision":  _avg("context_precision"),
        "per_question": per_question,
    }

    print(f"\n  SUMMARY — {config_name}:")
    print(f"    Faithfulness:      {summary['faithfulness']:.4f}")
    print(f"    Answer Relevance:  {summary['answer_relevance']:.4f}")
    print(f"    Context Recall:    {summary['context_recall']:.4f}")
    print(f"    Context Precision: {summary['context_precision']:.4f}")

    return summary


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(config_a: dict, config_b: dict):
    """Export kết quả A/B comparison ra results.md."""

    def star(score: float) -> str:
        if score >= 0.75: return "🟢"
        if score >= 0.50: return "🟡"
        return "🔴"

    # Worst performers
    pq_a = config_a["per_question"]
    worst = sorted(pq_a, key=lambda x: (x["faithfulness"] + x["answer_relevance"]) / 2)[:3]

    lines = [
        "# RAG Evaluation Results — Trợ Lý Du Lịch Việt Nam\n",
        f"> Đánh giá ngày: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"> Corpus: du lịch Việt Nam ({config_a['n_questions']} câu hỏi)\n\n",
        "## 1. Tổng quan Metrics (A/B Comparison)\n",
        "| Metric | Config A: Hybrid+RRF | Config B: Dense-only | Winner |\n",
        "|--------|---------------------|---------------------|--------|\n",
    ]

    metrics = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevance",   "answer_relevance"),
        ("Context Recall",     "context_recall"),
        ("Context Precision",  "context_precision"),
    ]
    for label, key in metrics:
        a_val = config_a[key]
        b_val = config_b[key]
        winner = "**A** ✅" if a_val >= b_val else "**B** ✅"
        lines.append(f"| {label} | {star(a_val)} `{a_val:.4f}` | {star(b_val)} `{b_val:.4f}` | {winner} |\n")

    # Overall score
    a_overall = round(sum(config_a[k] for _, k in metrics) / 4, 4)
    b_overall = round(sum(config_b[k] for _, k in metrics) / 4, 4)
    overall_winner = "**A** ✅" if a_overall >= b_overall else "**B** ✅"
    lines += [
        f"| **Overall** | `{a_overall:.4f}` | `{b_overall:.4f}` | {overall_winner} |\n\n",
    ]

    # Per-question table (Config A)
    lines += [
        "## 2. Kết quả Per-Question (Config A: Hybrid+RRF)\n\n",
        "| # | Câu hỏi | Faith | Relev | Recall | Precis |\n",
        "|---|---------|-------|-------|--------|--------|\n",
    ]
    for i, r in enumerate(pq_a, 1):
        q_short = r["question"][:40].replace("|", "\\|")
        lines.append(
            f"| {i} | {q_short}... | {r['faithfulness']:.3f} | "
            f"{r['answer_relevance']:.3f} | {r['context_recall']:.3f} | "
            f"{r['context_precision']:.3f} |\n"
        )

    # Worst performers
    lines += [
        "\n## 3. Worst Performers (câu hỏi trả lời kém nhất)\n\n",
    ]
    for w in worst:
        lines.append(f"**Q**: {w['question']}\n\n")
        lines.append(f"- Faithfulness: `{w['faithfulness']:.3f}` | Relevance: `{w['answer_relevance']:.3f}`\n")
        lines.append(f"- Answer snippet: _{w['answer_snippet']}_\n\n")

    # Analysis
    lines += [
        "## 4. Phân tích & Đề xuất Cải tiến\n\n",
        "### Kết luận\n",
        f"- Config A (Hybrid+RRF) đạt overall score **{a_overall:.4f}**\n",
        f"- Config B (Dense-only) đạt overall score **{b_overall:.4f}**\n",
        f"- RRF Reranking {'cải thiện' if a_overall > b_overall else 'không cải thiện'} "
        f"kết quả so với Dense-only\n\n",
        "### Điểm mạnh\n",
        "- Retrieval pipeline kết hợp Semantic + BM25 cho độ phủ tốt\n",
        "- PageIndex fallback đảm bảo không bỏ lỡ câu hỏi ngoài domain\n",
        "- GSAP UI giúp trải nghiệm người dùng mượt mà\n\n",
        "### Đề xuất cải tiến\n",
        "1. **Tăng chunk overlap** trong Task 4 để cải thiện Context Recall\n",
        "2. **Cross-encoder reranking** thay RRF để cải thiện Context Precision\n",
        "3. **Mở rộng corpus** thêm tỉnh thành mới (Đà Nẵng, Nha Trang, Phú Yên)\n",
        "4. **Fine-tune threshold** 0.48 theo từng query type\n",
        "5. **Conversation memory** để handle follow-up questions tốt hơn\n\n",
        "---\n",
        "*Evaluation bằng heuristic (keyword overlap) — không dùng LLM để tránh rate limit.*\n",
    ]

    content = "".join(lines)
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n[Export] Results saved to: {RESULTS_PATH}")
    return content


# =============================================================================
# MAIN
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Evaluation Pipeline — Trợ Lý Du Lịch Việt Nam")
    print("=" * 60)

    dataset = load_golden_dataset()
    print(f"\nLoaded {len(dataset)} test cases from golden_dataset.json")

    # Chọn subset để tránh rate limit nếu cần
    USE_SUBSET = "--subset" in sys.argv
    if USE_SUBSET:
        dataset = dataset[:5]
        print(f"[Subset mode] Running on {len(dataset)} questions only")

    # --- Config A: Hybrid + RRF ---
    print("\n[Config A] Hybrid Search + RRF Reranking")
    results_a = evaluate_config(
        "Config A: Hybrid+RRF",
        run_hybrid_pipeline,
        dataset,
        delay=1.5
    )

    # --- Config B: Dense-only ---
    print("\n[Config B] Dense-only (Semantic Search)")
    results_b = evaluate_config(
        "Config B: Dense-only",
        run_dense_only_pipeline,
        dataset,
        delay=1.5
    )

    # --- Export ---
    export_results(results_a, results_b)
    print("\n[Done] Evaluation complete!")
