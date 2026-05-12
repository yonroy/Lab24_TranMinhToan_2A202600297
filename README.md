# Lab 24 — Full Evaluation & Guardrail System

**Tác giả:** Tran Minh Toan — 2A202600297
**Lab:** AICB-P2T3 · Day 24 · VinUniversity

## Overview

Bài lab xây dựng hệ thống evaluation và guardrail production-ready cho RAG pipeline (Day 18),
gồm 4 phases: RAGAS automated evaluation, LLM-as-Judge với bias mitigation, input/output guardrails
(PII redaction + topic validation + Llama Guard via Groq), và production blueprint.

RAG pipeline sử dụng LangChain + Qdrant in-memory + nomic-embed-text-v1.5 + BAAI/bge-reranker-v2-m3
+ gpt-4o-mini trên corpus luật bảo vệ dữ liệu cá nhân Việt Nam (Day 18).

## Setup

```bash
pip install -r requirements.txt

set OPENAI_API_KEY=sk-...
set GROQ_API_KEY=gsk_...
```

Verify:

```bash
python -c "from rag.adapter import my_rag_pipeline; a,c = my_rag_pipeline('test'); print('OK', len(c))"
```

## Repo Structure

```
Lab24_TranMinhToan/
├── README.md
├── requirements.txt
├── prompts.md                          # AI prompts log (academic integrity)
├── config.py
├── rag/                                # RAG pipeline (Day 18)
├── phase-a/                            # RAGAS Evaluation
│   ├── generate_testset.py
│   ├── run_ragas.py
│   ├── testset_v1.csv
│   ├── ragas_results.csv
│   ├── ragas_summary.json
│   ├── testset_review_notes.md
│   └── failure_analysis.md
├── phase-b/                            # LLM-as-Judge
│   ├── judge_pipeline.py
│   ├── kappa_analysis.py
│   ├── pairwise_results.csv
│   ├── absolute_scores.csv
│   ├── human_labels.csv
│   └── judge_bias_report.md
├── phase-c/                            # Guardrails
│   ├── input_guard.py
│   ├── output_guard.py
│   ├── full_pipeline.py
│   ├── adversarial_test_results.csv
│   └── latency_benchmark.csv
├── phase-d/
│   └── blueprint.md
└── .github/workflows/
    └── eval-gate.yml
```

## Results Summary

### Phase A — RAGAS Evaluation (52 questions)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Faithfulness | 0.6333 | ≥ 0.85 | FAIL |
| Answer Relevancy | 0.3666 | ≥ 0.80 | FAIL |
| Context Precision | 0.8638 | ≥ 0.75 | PASS |
| Context Recall | 0.5472 | ≥ 0.75 | FAIL |

**Nguyên nhân thất bại:** Corpus tiếng Việt (luật bảo vệ dữ liệu) gây khó khăn cho RAGAS LLM judge
vì model gpt-4o-mini đánh giá faithfulness/relevancy không chính xác với nội dung pháp luật chuyên ngành.
Context Precision cao (0.86) cho thấy retrieval tốt; vấn đề nằm ở generation và judge evaluation.

Xem phân tích: [`phase-a/failure_analysis.md`](phase-a/failure_analysis.md)

### Phase B — LLM-as-Judge (30 pairs: top_k=3 vs top_k=5)

| Metric | Result |
|--------|--------|
| Cohen's kappa (human vs judge) | 0.4828 — moderate agreement |
| A (top_k=3) wins | 3/30 (10%) |
| B (top_k=5) wins | 8/30 (27%) |
| Ties | 19/30 (63%) |
| Position bias (A win rate run1) | 26.7% — NO bias detected |
| Length bias diff | +32.4% — YES detected |

**Kết luận:** top_k=5 nhỉnh hơn top_k=3. Length bias đáng kể (+32.4%) — LLM judge ưu tiên câu trả lời dài hơn.
Swap-and-average giảm position bias hiệu quả (tie rate tăng từ 37% → 63%).

Xem chi tiết: [`phase-b/judge_bias_report.md`](phase-b/judge_bias_report.md)

### Phase C — Guardrails

| Layer | Component | Result |
|-------|-----------|--------|
| L1 | PII Redaction (VN regex + Presidio) | Active |
| L1 | Topic Validator (gpt-4o-mini zero-shot) | Active |
| L3 | Safety Classifier (llama-3.1-8b-instant / Groq) | Active |

**Adversarial test (30 cases):**
- Blocked: 17/20 adversarial (85%)
- False positives: 0/10 legitimate (0%)

**Latency benchmark (n=10, local CPU):**

| Layer | P50 | P95 | SLO |
|-------|-----|-----|-----|
| L1 (PII + topic) | 860ms | 1128ms | < 50ms (FAIL — local Presidio cold start) |
| L2 (RAG) | 5328ms | 5899ms | < 2000ms (FAIL — CPU reranker) |
| L3 (Groq) | 440ms | 614ms | < 100ms (FAIL — network) |
| Total | 6461ms | 7270ms | < 2500ms |

> SLO thất bại là expected trên local CPU. Production với GPU + Presidio caching → L1 ~30ms, L2 ~800ms, L3 ~80ms.

### Phase D — Blueprint

Kiến trúc defense-in-depth 4 layers, SLO definitions, alert playbook, cost analysis ($5.10 lab / $277/tháng production).
Xem: [`phase-d/blueprint.md`](phase-d/blueprint.md)

## Lessons Learned

**Về RAGAS:** RAGAS metrics phụ thuộc nhiều vào LLM judge quality. Với corpus tiếng Việt chuyên ngành (pháp luật),
gpt-4o-mini đánh giá faithfulness thấp hơn thực tế vì model khó verify nội dung pháp luật cụ thể. Context Precision
cao (0.86) nhưng Answer Relevancy thấp (0.37) gợi ý vấn đề nằm ở generation prompt, không phải retrieval.

**Về LLM-as-Judge:** Swap-and-average là kỹ thuật đơn giản nhưng hiệu quả để giảm position bias.
Length bias (+32.4%) khó tránh hơn — cần thêm length-normalization instruction vào judge prompt hoặc
dùng absolute scoring làm cross-validation. kappa=0.48 (moderate) cho thấy LLM judge đủ tin cậy cho
automated screening nhưng không thay thế được human review hoàn toàn.

**Về Guardrails:** Llama Guard 3 đã bị deprecated trên Groq — cần chuyển sang llama-3.1-8b-instant với
custom safety classifier prompt. Format prompt quan trọng hơn model choice: multi-turn conversation format
không hoạt động, cần single-task classification prompt. Presidio cold start trên CPU (~900ms) là bottleneck
chính của L1; production fix là pre-warm + model caching.

## Demo Video

> _(Thêm link YouTube sau khi quay)_

Nội dung: RAGAS live eval → LLM Judge comparison → 3 adversarial attacks bị block → latency benchmark output.

## API Costs (thực tế)

| Phase | Model | Approx. cost |
|-------|-------|-------------|
| A — Testset generation (52 Q) | gpt-4o-mini | ~$0.60 |
| A — RAGAS eval (52 Q × 4 metrics) | gpt-4o-mini | ~$1.80 |
| B — Judge pipeline (30 pairs × 2 runs) | gpt-4o-mini | ~$0.90 |
| C — Topic validator (adversarial + benchmark) | gpt-4o-mini | ~$0.40 |
| C — Output guard | Groq (free) | $0.00 |
| **Total** | | **~$3.70** |
