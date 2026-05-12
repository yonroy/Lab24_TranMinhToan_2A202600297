# Lab 24 — Full Evaluation & Guardrail System

## Overview

Bài lab này xây dựng một hệ thống evaluation và guardrail production-ready cho RAG pipeline từ Day 18,
bao gồm RAGAS evaluation tự động, LLM-as-Judge với bias mitigation, input/output guardrails (PII redaction,
topic validation, Llama Guard 3 qua Groq API), và một blueprint document cho production deployment.

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
export HUGGINGFACE_TOKEN=hf_...
```

Verify setup:

```bash
python --version                            # >= 3.10
python -c "import ragas; print(ragas.__version__)"   # >= 0.2.0
python -m your_rag_module.test_query "What is X?"
```

## Repo Structure

```
lab24-eval-guardrails-<ten-cua-ban>/
├── README.md
├── requirements.txt
├── prompts.md
├── phase-a/
│   ├── generate_testset.py
│   ├── run_ragas.py
│   ├── testset_v1.csv
│   ├── testset_review_notes.md
│   ├── ragas_results.csv
│   ├── ragas_summary.json
│   └── failure_analysis.md
├── phase-b/
│   ├── judge_pipeline.py
│   ├── pairwise_results.csv
│   ├── absolute_scores.csv
│   ├── human_labels.csv
│   ├── kappa_analysis.py
│   └── judge_bias_report.md
├── phase-c/
│   ├── input_guard.py
│   ├── output_guard.py
│   ├── full_pipeline.py
│   ├── pii_test_results.csv
│   ├── adversarial_test_results.csv
│   └── latency_benchmark.csv
├── phase-d/
│   └── blueprint.md
├── .github/workflows/
│   └── eval-gate.yml
└── demo/
    └── demo-video.mp4  (hoặc link YouTube trong README)
```

## Results Summary

### Phase A — RAGAS Evaluation

| Metric             | Score | Target | Status |
|--------------------|-------|--------|--------|
| Faithfulness       | —     | ≥ 0.85 | —      |
| Answer Relevancy   | —     | ≥ 0.80 | —      |
| Context Precision  | —     | ≥ 0.70 | —      |
| Context Recall     | —     | ≥ 0.75 | —      |

- Test set: 50 questions (50% simple, 25% reasoning, 25% multi-context)
- Total eval cost: $X.XX
- Failure clusters: xem [`phase-a/failure_analysis.md`](phase-a/failure_analysis.md)

### Phase B — LLM-as-Judge

- Cohen's kappa vs human: X.XX (interpretation: ...)
- Position bias: A wins as first X% (expected ~50%)
- Length bias: B wins when longer X/Y cases
- Mitigation applied: swap-and-average cho position bias

### Phase C — Guardrails

| Layer | Component | Detection Rate | Latency P95 |
|-------|-----------|---------------|-------------|
| L1 | PII Redaction (Presidio + VN regex) | X/10 (X%) | Xms |
| L1 | Topic Validator | X/20 (X%) | Xms |
| L1 | Adversarial Defense | X/20 (X%) | — |
| L3 | Llama Guard 3 (Groq API) | X/10 unsafe | Xms |

- Full stack P50/P95/P99: Xms / Xms / Xms
- Baseline (no guardrail) P95: Xms
- Overhead: +Xms (~X%)

### Phase D — Blueprint

Xem chi tiết: [`phase-d/blueprint.md`](phase-d/blueprint.md)

## Lessons Learned

> _(Điền sau khi hoàn thành lab — 2–3 đoạn về những gì bạn học được)_

**Về RAGAS:** ...

**Về LLM-as-Judge:** ...

**Về Guardrails:** ...

## Demo Video

> Link YouTube (unlisted): https://youtu.be/...

Nội dung video (5 phút):
1. RAGAS chạy live trên 5 questions (1')
2. LLM-Judge so sánh 2 versions (1')
3. Adversarial test — 3 attacks bị block (2')
4. Latency benchmark P50/P95/P99 output (1')

## API Costs

| Phase | Model | Calls | Cost |
|-------|-------|-------|------|
| A — Test set generation | gpt-4o-mini | ~200 | ~$0.XX |
| A — RAGAS eval | gpt-4o-mini | ~600 | ~$0.XX |
| B — Pairwise judge | gpt-4o-mini | ~60 | ~$0.XX |
| B — Absolute scoring | gpt-4o-mini | ~30 | ~$0.XX |
| C — Topic validator | gpt-4o-mini | ~120 | ~$0.XX |
| C — Llama Guard | Groq (free) | ~120 | $0.00 |
| **Total** | | | **~$X.XX** |
