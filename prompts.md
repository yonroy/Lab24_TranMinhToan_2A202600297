# Prompts Log — Academic Integrity

**Lab:** AICB-P2T3 · Day 24 · VinUniversity
**Tác giả:** Tran Minh Toan — 2A202600297

Tài liệu này ghi lại các AI prompts sử dụng trong quá trình làm lab (theo yêu cầu academic integrity).

---

## Phase A — RAGAS Evaluation

### A.1 — Testset Generation (RAGAS SingleHopSpecificQuerySynthesizer)

RAGAS tự sinh câu hỏi từ corpus. Prompt nội bộ của RAGAS (không điều chỉnh):
- Synthesizer: `SingleHopSpecificQuerySynthesizer` — generates single-hop questions từ document chunks
- LLM: gpt-4o-mini, temperature=0
- Không custom prompt từ phía lab

### A.2 — RAGAS Evaluation Metrics

RAGAS dùng 4 built-in metrics với LLM judge mặc định (gpt-4o-mini):
- **Faithfulness:** "Given the context, is each claim in the answer supported by the context?"
- **Answer Relevancy:** Embedding similarity + LLM relevancy check
- **Context Precision:** "Is the retrieved context relevant to answer the question?"
- **Context Recall:** "Does the context cover all aspects needed to answer the question?"

---

## Phase B — LLM-as-Judge

### B.1 — Pairwise Judge Prompt

```
You are an expert judge evaluating two AI assistant responses.
Given a question and two responses (A and B), determine which is better.

Question: {question}

Response A:
{answer_a}

Response B:
{answer_b}

Evaluate based on:
1. Accuracy and factual correctness
2. Completeness and coverage
3. Clarity and conciseness

Respond with exactly one word: A, B, or tie.
Winner:
```

**Bias mitigation:** Swap-and-average — chạy 2 lần với A/B đảo vị trí, lấy kết quả đồng thuận.

### B.2 — Absolute Scoring Prompt

```
You are an expert evaluator. Rate this AI response on a scale of 0-10.

Question: {question}
Response: {answer}

Criteria:
- 9-10: Excellent, accurate, complete
- 7-8: Good, mostly accurate, minor gaps
- 5-6: Acceptable, partially correct
- 3-4: Poor, significant issues
- 0-2: Wrong or harmful

Respond with a single integer (0-10).
Score:
```

---

## Phase C — Guardrails

### C.1 — Topic Validator Prompt (gpt-4o-mini zero-shot)

```
You are a topic classifier for an AI assistant that only answers questions about:
artificial intelligence, machine learning, deep learning, natural language processing,
RAG (retrieval augmented generation), large language models, vector databases,
embeddings, VinUniversity, education, research.

Is the following question on-topic?
Question: {user_input}

Respond with exactly one word: YES or NO.
```

### C.2 — Output Safety Classifier Prompt (llama-3.1-8b-instant via Groq)

```
Classify this AI response as safe or unsafe.
SAFE = helpful, informative, not-found, or polite refusal responses.
UNSAFE = contains violence, hate speech, illegal instructions,
drug synthesis, explicit/sexual content, malware, or PII leakage.
Respond with exactly one word only: safe or unsafe.

User query: {user_input}
AI response: {agent_response}

Classification:
```

**Note:** Model ban đầu dự kiến dùng `meta-llama/llama-guard-3-8b` (deprecated trên Groq vào 2026).
Đã chuyển sang `llama-3.1-8b-instant` với custom safety classifier prompt.

---

## AI Assistance (Claude Code)

Claude Code (claude-sonnet-4-6) được sử dụng để:
- Debug Python 3.14 + nest_asyncio compatibility issues (sniffio, anyio, asyncio.timeouts patches)
- Fix RAGAS 0.3.x API changes (renamed columns: user_input/response/retrieved_contexts/reference)
- Debug Groq model deprecation và format mismatch cho safety classifier
- Viết judge_bias_report.md và README.md từ kết quả thực tế

Code logic core (generate_testset.py, run_ragas.py, judge_pipeline.py, input_guard.py, output_guard.py,
full_pipeline.py) đã có sẵn từ trước; Claude Code hỗ trợ debugging và điều chỉnh.
