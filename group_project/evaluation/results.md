# RAG Evaluation Results

## Framework sử dụng

> Framework đã chọn: **Heuristic RAG Evaluation Framework** (dựa trên 4 chỉ số chuẩn RAGAS: Faithfulness, Answer Relevance, Context Recall, Context Precision qua Keyword Overlap & Token Intersections để tránh bị Rate Limit khi gọi API).

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | `0.7850` | `0.7120` | `+0.0730` |
| Answer Relevance | `0.8520` | `0.7980` | `+0.0540` |
| Context Recall | `0.6470` | `0.5290` | `+0.1180` |
| Context Precision | `0.9410` | `0.8240` | `+0.1170` |
| **Average** | **`0.8063`** | **`0.7158`** | **`+0.0905`** |

---

## A/B Comparison Analysis

**Config A:**
> Hybrid Search kết hợp giữa Dense Search (ChromaDB - Cosine Similarity với model BAAI/bge-m3) và Sparse Search (BM25 Lexical). Kết quả được tổng hợp qua thuật toán RRF (Reciprocal Rank Fusion, $k=60$) và tích hợp cơ chế PageIndex Vectorless Fallback khi điểm Cosine Similarity < 0.48.

**Config B:**
> Dense-only Retrieval: Chỉ sử dụng Semantic Search đơn thuần dựa trên Cosine Similarity với ChromaDB, không áp dụng BM25 hay thuật toán Reranking RRF.

**Kết luận:**
> **Config A (Hybrid + RRF)** đạt điểm số trung bình **`0.8063`**, vượt trội hơn **Config B (Dense-only)** đạt **`0.7158`** (tăng **`+0.0905`**). Việc kết hợp cả từ khóa chính xác (BM25) và ngữ nghĩa (Dense Search) giúp tăng tỉ lệ tìm thấy thông tin cần thiết (**Context Recall +11.8%**) và làm cho câu trả lời bám sát thực tế tốt hơn rõ rệt.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Mùa đẹp nhất để đi du lịch Hà Nội là khi nào? | 0.650 | 0.714 | 0.500 | Retrieval | Chunk size tương đối lớn chứa nhiều thông tin về lịch sử, gây pha loãng từ khóa mùa vụ. |
| 2 | Những lăng tẩm vua triều Nguyễn nào ở Huế nên tham quan? | 0.680 | 0.750 | 0.500 | Retrieval | Tên các lăng tẩm (Minh Mạng, Khải Định, Tự Đức) rải rác ở nhiều đoạn khác nhau trong file Markdown. |
| 3 | Phương tiện di chuyển tốt nhất để khám phá Hà Giang là gì? | 0.724 | 0.833 | 0.333 | Generation | Câu trả lời sinh ra tóm tắt ngắn hơn so với chi tiết bảng giá thuê xe máy trong tài liệu gốc. |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng `chunk_overlap` từ 50 lên 100 tokens và giảm `chunk_size` từ 500 xuống 350 tokens trong Task 4 (Chunking & Indexing).
**Expected impact:** Tăng **Context Recall** thêm ~5–8% đối với các câu hỏi về danh sách địa điểm/mùa vụ.

### Cải tiến 2
**Action:** Tích hợp Cross-Encoder Reranker (Jina / BGE-Reranker-Large) ở Task 7 sau bước gộp danh sách RRF.
**Expected impact:** Sắp xếp lại thứ tự ưu tiên chính xác hơn, giúp tăng **Context Precision** và giảm nhiễu cho LLM khi sinh câu trả lời.

### Cải tiến 3
**Action:** Tinh chỉnh ngưỡng Cosine Fallback (score threshold) linh hoạt theo độ dài và loại hình câu hỏi (từ 0.48 xuống 0.42 cho query ngắn).
**Expected impact:** Kích hoạt PageIndex Vectorless Fallback chính xác hơn đối với các câu hỏi mở ngoài miền dữ liệu hiện có.
