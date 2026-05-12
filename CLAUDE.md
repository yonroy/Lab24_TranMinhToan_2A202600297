# Lab 24 — Full Evaluation & Guardrail System

## Second Brain
Đọc các file sau trước khi làm bất cứ gì:
- `d:\SecondBrain\_projects\Lab24_TranMinhToan\MEMORY\CONTEXT.md` — trạng thái dự án
- `d:\SecondBrain\_global\my-stack.md` — preferences cá nhân

Nếu MEMORY\CONTEXT.md chưa tồn tại, thông báo cho người dùng chạy init-project-prompt.md.

---

## Tổng quan dự án
Hệ thống evaluation và guardrail production-ready cho RAG pipeline (từ Day 18).
Gồm: RAGAS automated eval, LLM-as-Judge với bias mitigation, input/output guardrails (PII redaction, topic validation, Llama Guard 3 via Groq), và production blueprint.
Lab: AICB-P2T3 · Day 24 · VinUniversity.

## Tech Stack
- **Language:** Python ≥ 3.10
- **RAG base:** LangChain + ChromaDB + gpt-4o-mini (kế thừa Day 18)
- **Eval:** RAGAS ≥ 0.2.0
- **PII:** Microsoft Presidio + custom VN regex
- **Output guard:** Llama Guard 3 8B via Groq API (free tier)
- **Topic validator:** gpt-4o-mini zero-shot
- **APIs:** OPENAI_API_KEY, GROQ_API_KEY (set environment variables, không hardcode)

## Cấu trúc quan trọng
```
Lab24_TranMinhToan/
├── rag/                ← RAG Day 18 + adapter.py interface
├── phase-a/            ← RAGAS eval pipeline
├── phase-b/            ← LLM-as-Judge + bias analysis
├── phase-c/            ← Guardrails (input_guard, output_guard, full_pipeline)
├── phase-d/            ← blueprint.md production document
├── .github/workflows/  ← eval-gate.yml CI/CD
└── CLAUDE.md
```

---

## Quy tắc bắt buộc

### Không được làm
- ❌ Không hardcode API keys — dùng environment variables
- ❌ Không tự `git commit` hoặc `git push` khi chưa được xác nhận rõ ràng
- ❌ Không commit thẳng vào main/master
- ❌ Không dùng `any` trong type hints Python nếu không có lý do
- ❌ Không để print() debug trong code nộp — dùng logging

### Luôn phải làm
- ✅ Validate input tại mọi API boundary
- ✅ Mỗi method trả về latency_ms cùng với kết quả (theo pattern của phase-c)
- ✅ Sau khi hoàn thành task, nhắc người dùng commit
- ✅ Total API cost ≤ $20 cho toàn lab
- ✅ Groq free tier cho Llama Guard — không gọi OpenAI cho output guard

### Code style (Python)
- snake_case cho variables và functions, PascalCase cho class
- Type hints bắt buộc cho function signatures
- Import order: stdlib → external (ragas, presidio, openai) → internal → relative
- Mỗi phase có CSV output + optional JSON summary

---

## Patterns của dự án này

### RAG Adapter Interface (chuẩn cho tất cả phases)
```python
def my_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """Return (answer, contexts) — contexts là list of chunk strings."""
    return run_query(question, _search, _reranker)
```

### Guardrail Method Pattern
```python
def sanitize(self, text: str) -> tuple[str, float]:
    """Return (sanitized_text, latency_ms)."""
    start = time.time()
    # ... processing ...
    return result, (time.time() - start) * 1000
```

### Async Full Pipeline Pattern
```python
async def guarded_pipeline(user_input: str) -> dict:
    # L1: parallel
    pii_task = asyncio.create_task(input_guard.sanitize_async(user_input))
    topic_task = asyncio.create_task(topic_check_async(user_input))
    sanitized, _ = await pii_task
    topic_ok, _ = await topic_task
    # L2: RAG
    answer = await rag_pipeline_async(sanitized)
    # L3: Llama Guard
    safe, _, _ = await output_guard.check_async(sanitized, answer)
    # L4: fire-and-forget
    asyncio.create_task(audit_log(...))
```

### LLM Judge Swap-and-Average
```python
def pairwise_judge_with_swap(question, ans1, ans2, judge_llm):
    run1 = judge(question, ans1, ans2)          # A=ans1, B=ans2
    run2 = judge(question, ans2, ans1)          # A=ans2, B=ans1 → flip winner
    if run1.winner == run2.winner_flipped:
        return run1.winner  # agreement
    return "tie"            # disagreement
```

---

## Files quan trọng — đọc trước khi sửa
- `execution_plan.md` — checklist chi tiết mọi task, đọc trước khi code bất kỳ phase nào
- `blueprint.md` — SLOs và architecture chuẩn, tham chiếu khi thiết kế guardrails
- `rag/adapter.py` — interface dùng chung, không thay đổi signature
- `d:\SecondBrain\_projects\Lab24_TranMinhToan\MEMORY\DECISIONS.md` — lịch sử quyết định

---

## Build & Run
```bash
# Setup env
set OPENAI_API_KEY=sk-...
set GROQ_API_KEY=gsk_...

# Install
pip install "ragas>=0.2.0" presidio-analyzer presidio-anonymizer scikit-learn --break-system-packages

# Verify RAG adapter
python -c "from rag.adapter import my_rag_pipeline; a,c = my_rag_pipeline('test'); print('OK', len(c))"

# Phase A
python phase-a/generate_testset.py
python phase-a/run_ragas.py

# Phase C full pipeline
python phase-c/full_pipeline.py

# Validate CI YAML
yamllint .github/workflows/eval-gate.yml
```

---

## Session Management — Bắt buộc

### Cuối mỗi session
Khi người dùng nói "kết thúc", "done", "xong":

1. Cập nhật `d:\SecondBrain\_projects\Lab24_TranMinhToan\MEMORY\CONTEXT.md`
2. Tạo session log: `d:\SecondBrain\_projects\Lab24_TranMinhToan\MEMORY\sessions\YYYY-MM-DD-[tên-ngắn].md`
3. Nếu có bug đã fix → append vào `MISTAKES.md`
4. Nếu có quyết định kiến trúc → append vào `DECISIONS.md`
5. Hiển thị checklist:
   - [ ] CONTEXT.md đã cập nhật
   - [ ] Session log đã tạo
   - [ ] TODO phiên sau đã rõ

### Khi mắc lỗi
Append vào `MISTAKES.md`:
```
### [YYYY-MM-DD] <tên lỗi>
**Lỗi:** <làm gì sai>
**Fix:** <đã sửa thế nào>
**Lesson:** <rule cần nhớ>
```

---

## Context Monitoring
Sau mỗi câu trả lời dài:
```
📊 Context: ~XX% | Bloat: [nguồn lớn nhất nếu > 40%]
```
- > 70%: đề xuất `/clear` ngay
