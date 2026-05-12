# Failure Cluster Analysis — Phase A.3

RAG pipeline: _(mô tả ngắn — ví dụ: LangChain + ChromaDB + gpt-4o-mini, top_k=3)_  
Evaluation run: `ragas_results.csv` — 50 questions  
Ngày phân tích: ___________

---

## Bottom 10 Questions (thấp nhất theo average 4 metrics)

> Cách tính: `avg = (faithfulness + answer_relevancy + context_precision + context_recall) / 4`  
> Sort ascending, lấy 10 hàng đầu.

| # | Question (tóm tắt) | Type | F | AR | CP | CR | Avg | Cluster |
|---|-------------------|------|---|----|----|----|----|---------|
| 1 | _(dán question, truncate ~60 ký tự)_ | reasoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | C1 |
| 2 | | | | | | | | |
| 3 | | | | | | | | |
| 4 | | | | | | | | |
| 5 | | | | | | | | |
| 6 | | | | | | | | |
| 7 | | | | | | | | |
| 8 | | | | | | | | |
| 9 | | | | | | | | |
| 10 | | | | | | | | |

_F = Faithfulness, AR = Answer Relevancy, CP = Context Precision, CR = Context Recall_

---

## Clusters

### Cluster C1 — Multi-hop reasoning failures

**Pattern:** Câu hỏi yêu cầu kết hợp thông tin từ 2+ documents hoặc suy luận nhiều bước.
Retriever chỉ lấy top-3 chunks, không đủ context để trả lời.

**Số câu trong cluster:** X / 10

**Metrics đặc trưng:**
- Context Recall thấp nhất (CR < 0.5): retriever bỏ sót chunks quan trọng
- Faithfulness thường thấp theo: LLM hallucinate khi thiếu context

**Example questions:**
1. "_(câu hỏi multi-hop ví dụ 1)_"
2. "_(câu hỏi multi-hop ví dụ 2)_"
3. "_(thêm nếu có)_"

**Root cause:**
Retriever dùng dense vector similarity với `top_k=3`. Các câu multi-hop cần ít nhất 2 chunks
từ các documents khác nhau, nhưng similarity score của chunk thứ 2 thường thấp hơn threshold.

**Proposed fixes (theo thứ tự ưu tiên):**

1. **Tăng `top_k` từ 3 → 5–7:** Giảm miss rate, cost tăng nhẹ vì context dài hơn.
2. **Thêm re-ranker (Cohere Rerank hoặc `cross-encoder/ms-marco-MiniLM-L-6-v2`):**
   Re-rank top-10 candidates, giữ top-5 relevance cao nhất.
3. **Hybrid search (BM25 + vector):** BM25 bắt được exact keyword match mà dense search bỏ sót.
4. **Multi-query retrieval:** Decompose câu hỏi thành 2–3 sub-queries, retrieve riêng rồi merge.

**Expected improvement:** CR tăng 0.10–0.20 với top_k=5 + re-ranker.

---

### Cluster C2 — Off-topic context retrieval

**Pattern:** Retriever trả về chunks không liên quan (context precision thấp).
LLM buộc phải trả lời dựa trên context sai → faithfulness và answer relevancy cùng thấp.

**Số câu trong cluster:** X / 10

**Metrics đặc trưng:**
- Context Precision thấp nhất (CP < 0.4): phần lớn retrieved chunks không related
- Answer Relevancy thấp theo: câu trả lời lạc đề do context sai

**Example questions:**
1. "_(câu hỏi bị retrieve sai context ví dụ 1)_"
2. "_(câu hỏi bị retrieve sai context ví dụ 2)_"

**Root cause:**
Embedding model (`text-embedding-3-small`) encode câu hỏi vào một representation chung chung,
cosine similarity bắt được topic gần nhưng không phải exact concept. Một số terms trong corpus
có nghĩa đa dạng gây ra false positives.

**Proposed fixes:**

1. **Thêm metadata filter:** Filter theo `source_document` hoặc `section_type` trước khi vector search.
2. **Chunking strategy:** Tăng chunk size từ 500 → 1000 tokens để giữ ngữ cảnh đầy đủ hơn trong mỗi chunk.
3. **Fine-tune embedding model:** Nếu corpus domain-specific, fine-tune trên domain data.
4. **Câu hỏi clarification:** Thêm step clarify ambiguous questions trước khi retrieve.

**Expected improvement:** CP tăng 0.10–0.15 với metadata filtering.

---

### Cluster C3 — Hallucination khi ground truth ngắn (nếu có)

> _(Điền nếu phát hiện thêm pattern. Có thể bỏ qua nếu chỉ có 2 clusters rõ ràng.)_

**Pattern:** _(mô tả)_

**Số câu trong cluster:** X / 10

**Example questions:**
1. 
2. 

**Root cause:** _(phân tích)_

**Proposed fixes:**
1. 
2. 

---

## Tổng kết & Hành động

| Cluster | Câu | Root cause chính | Fix ưu tiên |
|---------|-----|-----------------|-------------|
| C1 — Multi-hop | X | top_k=3 quá thấp | Tăng top_k + re-ranker |
| C2 — Off-topic context | X | Embedding không đủ precision | Metadata filter + chunk size |
| C3 — ... | X | ... | ... |

**Tác động dự kiến nếu apply fixes:**
- Faithfulness: X.XX → ~X.XX (+X.XX)
- Context Recall: X.XX → ~X.XX (+X.XX)
- Context Precision: X.XX → ~X.XX (+X.XX)

**Kết luận:** RAG pipeline Day 18 hoạt động ổn với simple questions (F/AR cao),
nhưng cần cải thiện retrieval layer để handle multi-hop và domain-specific queries.
