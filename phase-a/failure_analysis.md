# Failure Analysis — Phase A RAGAS Evaluation

**Source:** `ragas_summary.json` — 10 worst-performing questions (sorted by avg score)
**Overall:** faithfulness=0.6333 (FAIL), answer_relevancy=0.3666 (FAIL), context_precision=0.8638 (PASS), context_recall=0.5472 (FAIL)

---

## Failure Clusters

### Cluster 1 — Out-of-Scope / Definition Questions (4 cases)

Questions asking for basic definitions not explicitly defined in the corpus:

| Question | Issue |
|----------|-------|
| "Giới tính là gì trong dữ liệu cá nhân?" | "Gender" definition not in corpus |
| "Quốc tịch là gì trong dữ liệu cá nhân?" | "Nationality" definition not in corpus |
| "What are the financial sources outlined in Điều 31...?" | Điều 31 not retrieved |
| "What are the main regulations established by CHÍNH PHỦ...?" | Too broad, no specific chunks |

**All scores: 0.0 faithfulness, 0.0 answer_relevancy, 0.0 context_precision**
**Root cause:** RAG returns "Không tìm thấy." → RAGAS evaluates as complete failure.

**Fix:** (1) Expand corpus to include definitions section; (2) Filter definition questions from testset.

---

### Cluster 2 — Cross-Border Transfer Questions (2 cases, duplicate)

Two nearly identical questions about data transfer abroad (Điều 25):
- "What are the requirements for transferring personal data of Vietnamese citizens abroad?"
- "What are the requirements for transferring personal data of Vietnamese citizens abroad according to the regulations?"

| Metric | Q1 | Q2 |
|--------|----|----|
| faithfulness | 0.0 | 0.0 |
| answer_relevancy | 0.0 | 0.0 |
| context_precision | 0.0 | 1.0 |
| context_recall | 0.5 | 0.22 |

**Root cause:** RAGAS faithfulness=0 suggests the LLM generates hallucinated answers not supported
by retrieved Điều 25 text. Context Precision varies (0 vs 1.0) across duplicates — inconsistent
chunk retrieval for essentially same question.

**Fix:** Improve Điều 25 chunking (article-level split instead of sliding window).

---

### Cluster 3 — Rights and Obligations (3 cases)

Questions about rights of data subjects and controller obligations:

| Question | worst_metric | context_precision | context_recall |
|----------|-------------|------------------|----------------|
| "...rights and obligations...Bên Kiểm soát..." | faithfulness | 0.833 | 0.0 |
| "How does Bộ luật Dân sự relate to...data subjects?" | faithfulness | 1.0 | 0.0 |
| "What responsibilities does the government have...?" | faithfulness | 1.0 | 0.5 |

**Pattern:** High context_precision (0.83–1.0) but faithfulness=0.0 AND context_recall=0.
This means retrieval finds relevant chunks (precision good) but LLM either: (a) generates
answer from parametric knowledge ignoring context, or (b) RAGAS judge incorrectly evaluates
Vietnamese legal answers as unfaithful.

**Fix:** Strengthen RAG generation prompt: "Answer ONLY using the provided context. If not in context, say 'Không tìm thấy.'" — currently this instruction exists but LLM ignores it for these complex questions.

---

## Summary

| Cluster | Count | Root Cause | Priority Fix |
|---------|-------|------------|-------------|
| Out-of-scope questions | 4 | Missing corpus coverage | Filter testset; expand corpus |
| Duplicate cross-border | 2 | Hallucination + chunking | Better chunking strategy |
| Rights/obligations | 3 | Generation ignores context | Stricter system prompt |
| High context_precision but 0 faithfulness | 5 | RAGAS judge bias on Vietnamese legal text | Use multilingual judge model |

## RAGAS Score Interpretation for This Corpus

- **Context Precision 0.86 (PASS):** BM25 + dense hybrid retrieval works well for Vietnamese legal text
- **Faithfulness 0.63 (FAIL):** Generator prompt + RAGAS judge both struggle with Vietnamese legal nuance
- **Answer Relevancy 0.37 (FAIL):** LLM often returns "Không tìm thấy" which scores 0 on relevancy; inflates failures
- **Context Recall 0.55 (FAIL):** Chunking strategy misses some facts spread across multiple articles

**Recommendation for production:** Use Vietnamese-specific judge (GPT-4o or Claude with Vietnamese legal context)
and filter trivially-unanswerable questions from continuous eval testset.
