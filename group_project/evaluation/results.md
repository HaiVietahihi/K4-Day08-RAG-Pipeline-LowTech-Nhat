# RAG Evaluation Results

## Framework sử dụng

> Heuristic RAG Evaluation Framework (Faithfulness, Relevance, Recall, Precision via Keyword Overlap)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | `0.8037` | `0.7250` | `+0.0787` |
| Answer Relevance | `0.8982` | `0.8120` | `+0.0862` |
| Context Recall | `0.5810` | `0.4850` | `+0.0960` |
| Context Precision | `1.0000` | `0.8500` | `+0.1500` |
| **Average** | **`0.8207`** | **`0.7180`** | **`+0.1027`** |

---

## A/B Comparison Analysis

**Config A:**
> Hybrid Search (Semantic Search + BM25 Lexical) kết hợp thuật toán RRF (Reciprocal Rank Fusion, k=60) và PageIndex Vectorless Fallback (khi Cosine < 0.48).

**Config B:**
> Dense-only Retrieval (Chỉ sử dụng Semantic Search dựa trên Cosine Similarity với BAAI/bge-m3), không áp dụng RRF reranking và Lexical search.

**Kết luận:**
> Config A (Hybrid + RRF) đạt điểm trung bình **`0.8207`**, vượt trội hơn Config B (Dense-only) đạt **`0.7180`** (chênh lệch **`+0.1027`** / +10.27%). Sự kết hợp giữa Semantic và BM25 qua RRF giúp gia tăng Context Recall (+9.6%) và Context Precision (+15.0%) rõ rệt trên bộ dữ liệu du lịch.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Phương tiện di chuyển tốt nhất để khám phá Hà Giang là gì? | `0.766` | `0.727` | `0.667` | Retrieval | Chunking size quá rộng hoặc thiếu từ khóa đặc thù |
| 2 | Lịch trình du lịch Hà Giang 3 ngày 2 đêm nên đi như thế nào? | `0.809` | `0.875` | `0.667` | Retrieval | Chunking size quá rộng hoặc thiếu từ khóa đặc thù |
| 3 | Hà Giang có những điểm du lịch nổi tiếng nào? | `0.724` | `1.000` | `0.333` | Retrieval | Chunking size quá rộng hoặc thiếu từ khóa đặc thù |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng chunk overlap từ 50 lên 100 tokens trong Task 4 (Chunking & Indexing).
**Expected impact:** Giảm mất mát ngữ cảnh giữa các đoạn, tăng Context Recall lên ~5-8%.

### Cải tiến 2
**Action:** Tích hợp Cross-Encoder Reranker (Jina / BGE-Reranker) sau bước RRF.
**Expected impact:** Sắp xếp các đoạn tài liệu quan trọng nhất lên vị trí top 1-2, giúp tăng Context Precision và Answer Relevance.

### Cải tiến 3
**Action:** Mở rộng Golden Dataset thêm 20+ câu hỏi cạnh biên (edge cases) và câu hỏi đa chủ đề.
**Expected impact:** Giúp hệ thống tự động calibrate chính xác hơn ngưỡng Fallback (Cosine Threshold) cho PageIndex.
