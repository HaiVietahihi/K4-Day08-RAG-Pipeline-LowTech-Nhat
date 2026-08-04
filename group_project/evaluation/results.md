# RAG Evaluation Results — Trợ Lý Du Lịch Việt Nam
> Đánh giá ngày: 2026-08-04 18:15:46
> Corpus: du lịch Việt Nam (17 câu hỏi)

## 1. Tổng quan Metrics (A/B Comparison)
| Metric | Config A: Hybrid+RRF | Config B: Dense-only | Winner |
|--------|---------------------|---------------------|--------|
| Faithfulness | 🟢 `0.7502` | 🔴 `0.0000` | **A** ✅ |
| Answer Relevance | 🟢 `0.8714` | 🔴 `0.0000` | **A** ✅ |
| Context Recall | 🟡 `0.6255` | 🔴 `0.0000` | **A** ✅ |
| Context Precision | 🟢 `1.0000` | 🔴 `0.0000` | **A** ✅ |
| **Overall** | `0.8118` | `0.0000` | **A** ✅ |

## 2. Kết quả Per-Question (Config A: Hybrid+RRF)

| # | Câu hỏi | Faith | Relev | Recall | Precis |
|---|---------|-------|-------|--------|--------|
| 1 | Hà Giang có những điểm du lịch nổi tiếng... | 0.769 | 1.000 | 0.333 | 1.000 |
| 2 | Lịch trình du lịch Hà Giang 3 ngày 2 đêm... | 0.776 | 0.875 | 0.667 | 1.000 |
| 3 | Phương tiện di chuyển tốt nhất để khám p... | 0.761 | 0.727 | 0.667 | 1.000 |
| 4 | Hà Nội có những khu phố cổ và di tích lị... | 0.814 | 1.000 | 0.667 | 1.000 |
| 5 | Ẩm thực đặc trưng của Hà Nội gồm những m... | 0.872 | 0.778 | 0.571 | 1.000 |
| 6 | Mùa đẹp nhất để đi du lịch Hà Nội là khi... | 0.692 | 1.000 | 0.667 | 1.000 |
| 7 | Huế có những di sản văn hóa thế giới nào... | 0.588 | 1.000 | 0.571 | 1.000 |
| 8 | Những lăng tẩm vua triều Nguyễn nào ở Hu... | 0.811 | 0.625 | 0.625 | 1.000 |
| 9 | Đặc sản ẩm thực Huế có gì nổi tiếng?... | 0.787 | 0.875 | 0.625 | 1.000 |
| 10 | Hải Phòng có những điểm tham quan nào đá... | 0.836 | 1.000 | 0.667 | 1.000 |
| 11 | Đặc sản hải sản ở Hải Phòng có gì ngon?... | 0.797 | 0.667 | 0.667 | 1.000 |
| 12 | Đảo Cát Bà Hải Phòng có những hoạt động ... | 0.688 | 0.900 | 0.636 | 1.000 |
| 13 | Du lịch Kiên Giang có gì nổi bật?... | 0.830 | 0.857 | 0.700 | 1.000 |
| 14 | Phú Quốc (Kiên Giang) có những bãi biển ... | 0.544 | 0.875 | 0.667 | 1.000 |
| 15 | Cà Mau có những địa điểm du lịch sinh th... | 0.759 | 1.000 | 0.667 | 1.000 |
| 16 | Đặc sản Cà Mau có gì ngon và lạ?... | 0.702 | 0.857 | 0.667 | 1.000 |
| 17 | Nên đặt phòng khách sạn loại nào khi đi ... | 0.727 | 0.778 | 0.571 | 1.000 |

## 3. Worst Performers (câu hỏi trả lời kém nhất).

**Q**: Phú Quốc (Kiên Giang) có những bãi biển nào đẹp nhất?

- Faithfulness: `0.544` | Relevance: `0.875`
- Answer snippet: _Chào bạn! Rất vui được hỗ trợ bạn tìm hiểu về vẻ đẹp của Đảo Ngọc Phú Quốc.

Dựa trên các tài liệu hiện có, tại Phú Quốc_

**Q**: Những lăng tẩm vua triều Nguyễn nào ở Huế nên tham quan?

- Faithfulness: `0.811` | Relevance: `0.625`
- Answer snippet: _Chào bạn! Rất vui được hỗ trợ bạn tìm hiểu về các lăng tẩm của triều Nguyễn tại Huế. Dựa trên các tài liệu hiện có, dưới_

**Q**: Đặc sản hải sản ở Hải Phòng có gì ngon?

- Faithfulness: `0.797` | Relevance: `0.667`
- Answer snippet: _Chào bạn! Rất vui được hỗ trợ bạn khám phá ẩm thực của thành phố Hoa Phượng Đỏ. Dựa trên các tài liệu hiện có, tôi xin c_

## 4. Phân tích & Đề xuất Cải tiến

### Kết luận
- Config A (Hybrid+RRF) đạt overall score **0.8118**
- Config B (Dense-only) đạt overall score **0.0000**
- RRF Reranking cải thiện kết quả so với Dense-only

### Điểm mạnh
- Retrieval pipeline kết hợp Semantic + BM25 cho độ phủ tốt
- PageIndex fallback đảm bảo không bỏ lỡ câu hỏi ngoài domain
- GSAP UI giúp trải nghiệm người dùng mượt mà

### Đề xuất cải tiến
1. **Tăng chunk overlap** trong Task 4 để cải thiện Context Recall
2. **Cross-encoder reranking** thay RRF để cải thiện Context Precision
3. **Mở rộng corpus** thêm tỉnh thành mới (Đà Nẵng, Nha Trang, Phú Yên)
4. **Fine-tune threshold** 0.48 theo từng query type
5. **Conversation memory** để handle follow-up questions tốt hơn

---
*Evaluation bằng heuristic (keyword overlap) — không dùng LLM để tránh rate limit.*
