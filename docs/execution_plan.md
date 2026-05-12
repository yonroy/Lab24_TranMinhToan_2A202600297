# Kế hoạch thực hiện Lab 24

AICB-P2T3 · VinUniversity · Tháng 5, 2026  
Setup: RAG Day 18 ✓ | Corpus ✓ | API keys ✓ | Groq (thay GPU) | Intermediate Python

---

## Chuẩn bị (trước khi bắt đầu — 10 phút)

- [ ] Đăng ký tài khoản Groq tại https://groq.com → lấy API key (`GROQ_API_KEY`)
- [ ] Set environment variables:
  ```bash
  set GROQ_API_KEY=gsk_...
  set OPENAI_API_KEY=sk-...
  ```
- [ ] Tạo repo Lab 24:
  ```bash
  git init lab24-eval-guardrails-<ten>
  cd lab24-eval-guardrails-<ten>
  ```
- [ ] Copy thư mục RAG Day 18 vào repo:
  ```bash
  xcopy "D:\AI engineer\Vinuni\Stage2\Day 9\Lab\Day18-Track3-Production-RAG\src" ".\rag" /E /I
  ```
  Kết quả:
  ```
  lab24-eval-guardrails/
  └── rag/
      ├── __init__.py
      ├── pipeline.py       ← chứa run_query
      ├── m1_chunking.py
      ├── m2_search.py      ← HybridSearch
      ├── m3_rerank.py      ← CrossEncoderReranker
      ├── m4_eval.py
      └── m5_enrichment.py
  ```
- [ ] Tạo file `rag/adapter.py` — interface dùng cho toàn bộ Lab 24:
  ```python
  from rag.pipeline import run_query
  from rag.m2_search import HybridSearch
  from rag.m3_rerank import CrossEncoderReranker

  # Khởi tạo 1 lần, tái dùng cho Phase A, B, C
  # Điền args giống cách Day 18 khởi tạo
  _search = HybridSearch(...)
  _reranker = CrossEncoderReranker(...)

  def my_rag_pipeline(question: str) -> tuple[str, list[str]]:
      """Interface chuẩn cho Lab 24 — return (answer, contexts)."""
      return run_query(question, _search, _reranker)
  ```
- [ ] Verify RAG adapter chạy được:
  ```bash
  python -c "
  from rag.adapter import my_rag_pipeline
  answer, contexts = my_rag_pipeline('test question')
  assert isinstance(answer, str)
  assert isinstance(contexts, list) and len(contexts) > 0
  print('RAG OK —', len(contexts), 'chunks retrieved')
  print('Answer preview:', answer[:100])
  "
  ```
- [ ] Verify Python và RAGAS:
  ```bash
  python --version
  python -c "import ragas; print(ragas.__version__)"  # >= 0.2.0
  ```
- [ ] Cài packages Lab 24:
  ```bash
  pip install "ragas>=0.2.0" presidio-analyzer presidio-anonymizer scikit-learn --break-system-packages
  ```
- [ ] `git add -A && git commit -m "init: RAG Day 18 copied + adapter ready"`

---

## Phase A — RAGAS Evaluation (60 phút · 30 điểm)

**Mục tiêu:** Build automated eval pipeline cho RAG Day 18.

### Task A.1 — Synthetic Test Set (15' · 8đ)

- [ ] Tạo `phase-a/generate_testset.py` với `TestsetGenerator`
  - `generator_llm` = `ChatOpenAI(model="gpt-4o-mini")`
  - `critic_llm` = `ChatOpenAI(model="gpt-4o-mini")`
  - Distribution: `{simple: 0.5, reasoning: 0.25, multi_context: 0.25}`
  - `test_size=50`
- [ ] Chạy script → save `phase-a/testset_v1.csv`
- [ ] Verify: `df['evolution_type'].value_counts()` → distribution đúng không?
- [ ] Verify: file có ≥ 50 rows và 4 cột (`question, ground_truth, contexts, evolution_type`)
- [ ] Mở CSV, manual review 10 câu → ghi vào `testset_review_notes.md`
- [ ] Chỉnh sửa ít nhất 1 câu (ghi rõ trước/sau trong review notes)
- [ ] `git commit -m "A.1: testset generated + manual review"`

**Ghi chú nếu bị lỗi:**
- `RateLimitError` → thêm `max_concurrent=2`
- `OutOfMemoryError` → split corpus thành chunks 500–1000 tokens trước khi load
- Questions kỳ lạ → đây là lý do cần manual review, xóa câu xấu và regenerate

---

### Task A.2 — Run RAGAS 4 Metrics (20' · 10đ)

- [ ] Tạo `phase-a/run_ragas.py`
- [ ] Loop 50 questions → chạy RAG pipeline Day 18 → collect `answer, contexts`
- [ ] Build `Dataset` từ list of dicts
- [ ] Gọi `evaluate()` với 4 metrics: `faithfulness, answer_relevancy, context_precision, context_recall`
- [ ] Save `phase-a/ragas_results.csv` (50 rows, 4 metric columns)
- [ ] Save `phase-a/ragas_summary.json` (4 aggregate scores)
- [ ] Ghi total API cost vào `README.md` (estimate từ token usage)
- [ ] Nếu metric nào < 0.5: ghi observation vào README

**Benchmark targets (self-assess):**

| Metric | Target | Min OK |
|--------|--------|--------|
| Faithfulness | ≥ 0.85 | 0.75 |
| Answer Relevancy | ≥ 0.80 | 0.70 |
| Context Precision | ≥ 0.70 | 0.60 |
| Context Recall | ≥ 0.75 | 0.65 |

- [ ] `git commit -m "A.2: RAGAS eval complete — F=X.XX AR=X.XX CP=X.XX CR=X.XX"`

---

### Task A.3 — Failure Cluster Analysis (15' · 8đ)

- [ ] Tính `avg_score = mean([faithfulness, answer_relevancy, context_precision, context_recall])` mỗi row
- [ ] Sort ascending, lấy bottom 10
- [ ] Nhóm thành ≥ 2 clusters với pattern rõ ràng
- [ ] Điền vào `phase-a/failure_analysis.md`:
  - Bảng bottom 10 với scores + cluster label
  - Mỗi cluster: ≥ 2 examples + root cause + proposed fix kỹ thuật
- [ ] Verify: proposed fix phải technical (top_k, re-ranker, hybrid search...) — không phải "improve prompt"
- [ ] `git commit -m "A.3: failure cluster analysis — 2 clusters identified"`

---

### Task A.4 — CI/CD Integration Plan (10' · 4đ)

- [ ] Tạo `.github/workflows/eval-gate.yml` theo template
- [ ] Tạo `scripts/run_eval.py` với threshold gate:
  ```python
  if scores['faithfulness'] < 0.85:
      sys.exit(1)
  ```
- [ ] Validate YAML: `yamllint .github/workflows/eval-gate.yml`
- [ ] Đảm bảo có `upload-artifact` step
- [ ] `git commit -m "A.4: CI/CD eval gate workflow"`

---

## Phase B — LLM-as-Judge & Calibration (60 phút · 25 điểm)

**Mục tiêu:** Build judge pipeline với bias mitigation và human calibration.

**Cần 2 versions RAG để so sánh:**
→ Cách nhanh nhất: RAG Day 18 với `top_k=3` (version A) vs `top_k=5` (version B)

### Task B.1 — Pairwise Judge Pipeline (20' · 10đ)

- [ ] Tạo `phase-b/judge_pipeline.py`
- [ ] Implement `pairwise_judge_with_swap(question, ans1, ans2, judge_llm)`:
  - Run 1: ans1 as A, ans2 as B
  - Run 2: ans2 as A, ans1 as B → **flip winner** khi aggregate
  - Aggregate: cả 2 đồng ý → winner đó; khác nhau → "tie"
- [ ] Implement `parse_judge_output(text)` với robust JSON parsing (strip ```json fences)
- [ ] Chạy trên ≥ 30 questions từ test set
- [ ] Save `phase-b/pairwise_results.csv` với cột: `question, answer_a, answer_b, run1_winner, run2_winner, winner_after_swap`
- [ ] `git commit -m "B.1: pairwise judge with swap-and-average"`

---

### Task B.2 — Absolute Scoring (10' · 5đ)

- [ ] Implement `absolute_score(question, answer, judge_llm)`:
  - 4 dimensions: accuracy, relevance, conciseness, helpfulness (1–5 scale)
  - overall = mean của 4 dims
- [ ] Chạy trên 30 questions
- [ ] Save `phase-b/absolute_scores.csv`
- [ ] `git commit -m "B.2: absolute scoring 4 dimensions"`

---

### Task B.3 — Human Calibration (20' · 8đ)

- [ ] Sample 10 cặp: `df.sample(10, random_state=42)`
- [ ] Export `to_label.csv` với `question, answer_a, answer_b`
- [ ] Mở file, tự đọc và label 10 cặp thủ công
- [ ] Save `phase-b/human_labels.csv` với cột: `question_id, human_winner, confidence, notes`
  - confidence: high / medium / low
- [ ] Normalize labels về "A" / "B" / "tie" (không phải "answer_a", "Answer B", etc.)
- [ ] Compute: `cohen_kappa_score(human_labels, judge_labels)`
- [ ] Interpret theo bảng kappa scale, ghi kết quả vào README
- [ ] Nếu kappa < 0.6: viết root cause analysis trong `judge_bias_report.md`
- [ ] `git commit -m "B.3: human calibration — kappa=X.XX"`

---

### Task B.4 — Bias Report (10' · 2đ)

- [ ] Đo position bias: `A_wins_rate = (df['run1_winner'] == 'A').mean()`
- [ ] Đo length bias: so sánh B win rate khi B dài hơn vs ngắn hơn
- [ ] Vẽ ít nhất 1 chart (matplotlib bar chart, save PNG)
- [ ] Điền vào `phase-b/judge_bias_report.md`:
  - Ít nhất 2 biases với numbers
  - Mitigation strategy
- [ ] `git commit -m "B.4: bias analysis — position X% length X%"`

---

## Phase C — Guardrails Stack (90 phút · 35 điểm)

**Mục tiêu:** Build complete defense-in-depth với latency budget.

### Task C.1 — PII Redaction (20' · 8đ)

- [ ] Tạo `phase-c/input_guard.py` với class `InputGuard`
- [ ] Layer 1 — VN regex:
  ```python
  VN_PII = {
      "cccd": r"\b\d{12}\b",
      "phone_vn": r"(\+84|0)\d{9,10}",
      "tax_code": r"\b\d{10}(-\d{3})?\b",
      "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
  }
  ```
- [ ] Layer 2 — Presidio NER (`AnalyzerEngine` + `AnonymizerEngine`)
- [ ] Method `sanitize(text)` trả về `(output, latency_ms)`
- [ ] Build test set 10 inputs (mix EN/VN, bao gồm: empty, 5000 chars, multiple PII)
- [ ] Chạy test → save `phase-c/pii_test_results.csv`
- [ ] Verify: detection ≥ 80%, latency P95 < 50ms
- [ ] `git commit -m "C.1: PII guardrail — detection X/10"`

---

### Task C.2 — Topic Validator (15' · 6đ)

- [ ] Chọn **Option 2 (LLM zero-shot)** — nhanh nhất cho intermediate:
  ```python
  def topic_check_llm(text, allowed_topics, llm):
      prompt = f"Is this question about one of: {allowed_topics}? Answer YES or NO only.\nQuestion: {text}"
      response = llm.invoke(prompt).content.strip()
      return response.upper().startswith("YES"), response
  ```
- [ ] Define `allowed_topics` phù hợp với domain RAG Day 18 của bạn
- [ ] Build 20 test inputs (10 on-topic, 10 off-topic)
- [ ] Verify: accuracy ≥ 75%
- [ ] Implement graceful fallback message: "Câu hỏi này nằm ngoài phạm vi hỗ trợ của hệ thống. Tôi có thể giúp bạn về [domain]. Bạn có muốn thử lại không?"
- [ ] `git commit -m "C.2: topic validator — accuracy X/20"`

---

### Task C.3 — Adversarial Testing (15' · 6đ)

- [ ] Build 20 adversarial inputs trong `phase-c/full_pipeline.py`:
  - 5 DAN variants
  - 5 roleplay attacks
  - 3 payload splitting
  - 3 encoding attacks (Base64, Unicode)
  - 4 indirect injection
- [ ] Run qua `input_guard.sanitize()` + `topic_check_llm()` → track blocked/passed
- [ ] Build 10 legitimate queries để kiểm tra false positives
- [ ] Save `phase-c/adversarial_test_results.csv`
- [ ] Verify: detection rate ≥ 70%, FP ≤ 10% trên legitimate queries
- [ ] `git commit -m "C.3: adversarial testing — defense X/20"`

---

### Task C.4 — Llama Guard 3 via Groq (20' · 8đ)

- [ ] Tạo `phase-c/output_guard.py` với class `OutputGuardAPI` (Option B — Groq)
  ```python
  self.url = "https://api.groq.com/openai/v1/chat/completions"
  # model: "llama-guard-3-8b"
  ```
- [ ] Method `check(user_input, agent_response)` → trả về `(is_safe, result, latency_ms)`
- [ ] Build test set:
  - 10 unsafe outputs (craft thủ công theo FAQ Q10: violence, self-harm, hate, misinfo)
  - 10 safe outputs (câu trả lời bình thường của RAG)
- [ ] Verify: detection ≥ 80% unsafe, FP ≤ 20% safe
- [ ] Measure và record latency P95
- [ ] `git commit -m "C.4: Llama Guard via Groq — detection X/10"`

---

### Task C.5 — Full Stack Integration (20' · 7đ)

- [ ] Implement `async def guarded_pipeline(user_input)` trong `phase-c/full_pipeline.py`:
  ```python
  # L1: parallel (PII + Topic)
  pii_task = asyncio.create_task(...)
  topic_task = asyncio.create_task(...)
  sanitized, _ = await pii_task
  topic_ok, _ = await topic_task
  timings['L1'] = ...

  # L2: RAG
  answer = await rag_pipeline_async(sanitized)
  timings['L2'] = ...

  # L3: Llama Guard
  safe, _, _ = await output_guard.check_async(sanitized, answer)
  timings['L3'] = ...

  # L4: fire-and-forget
  asyncio.create_task(audit_log(...))
  ```
- [ ] Implement `async def benchmark(n=100)` → collect timings
- [ ] Chạy benchmark 100 requests
- [ ] Compute P50/P95/P99 cho L1, L2, L3
- [ ] Save `phase-c/latency_benchmark.csv`
- [ ] Verify: L1 P95 < 50ms, L3 P95 < 100ms
- [ ] Ghi overhead vs baseline vào README
- [ ] `git commit -m "C.5: full stack — P95=Xms overhead=+Xms"`

---

## Phase D — Blueprint Document (30 phút · 10 điểm)

**File:** `phase-d/blueprint.md` (4–6 trang)

- [ ] **Section 1 — SLO (2đ):** Điền ≥ 5 SLOs với số thực từ lab của bạn
- [ ] **Section 2 — Architecture diagram (3đ):** Copy/paste Mermaid diagram, điền latency annotation thực tế từ C.5 benchmark
- [ ] **Section 3 — Alert Playbook (3đ):** Review 3 incidents, customize theo domain và pipeline của bạn
- [ ] **Section 4 — Cost Analysis (2đ):** Điền số thực từ API cost log trong lab
- [ ] `git commit -m "D: blueprint complete"`

---

## Submission Checklist (trước khi nộp)

### Files bắt buộc

- [ ] `README.md` — overview 200–300 từ, results summary điền đầy đủ
- [ ] `requirements.txt` — pinned versions (`pip freeze > requirements.txt`)
- [ ] `prompts.md` — log tất cả AI prompts đã dùng trong lab
- [ ] `phase-a/testset_v1.csv` — ≥ 50 rows, 4 cột
- [ ] `phase-a/testset_review_notes.md` — 10 reviews, ≥ 1 chỉnh sửa
- [ ] `phase-a/ragas_results.csv` — 50 rows, 4 metric columns
- [ ] `phase-a/ragas_summary.json` — 4 aggregate scores
- [ ] `phase-a/failure_analysis.md` — bottom 10, ≥ 2 clusters
- [ ] `phase-b/pairwise_results.csv` — run1, run2, winner_after_swap columns
- [ ] `phase-b/absolute_scores.csv` — 30 rows, 4 dimensions + overall
- [ ] `phase-b/human_labels.csv` — 10 labels với confidence + notes
- [ ] `phase-b/judge_bias_report.md` — ≥ 2 biases với numbers + chart
- [ ] `phase-c/input_guard.py`
- [ ] `phase-c/output_guard.py`
- [ ] `phase-c/full_pipeline.py`
- [ ] `phase-c/pii_test_results.csv`
- [ ] `phase-c/adversarial_test_results.csv`
- [ ] `phase-c/latency_benchmark.csv`
- [ ] `phase-d/blueprint.md`
- [ ] `.github/workflows/eval-gate.yml` — valid YAML
- [ ] `demo/demo-video.mp4` hoặc YouTube link

### Final checks

- [ ] Commit history có ít nhất 8 commits rõ ràng
- [ ] `yamllint .github/workflows/eval-gate.yml` → pass
- [ ] `python -m pytest phase-c/test_guards.py` → pass (nếu có tests)
- [ ] Tổng API cost ghi trong README ≤ $20
- [ ] Demo video 5 phút: RAGAS (1') + Judge (1') + Adversarial (2') + Benchmark (1')
- [ ] `git push origin main`

---

## Ghi chú nhanh

| Cần làm gì | Lệnh |
|------------|------|
| Install packages | `pip install -r requirements.txt --break-system-packages` |
| Check API keys | `echo $OPENAI_API_KEY \| head -c 10` |
| Validate YAML | `yamllint .github/workflows/eval-gate.yml` |
| Commit | `git add -A && git commit -m "message"` |
| Check costs | Xem OpenAI dashboard → Usage |
| Stuck > 20 phút | Slack #lab24-eval-guardrails |
