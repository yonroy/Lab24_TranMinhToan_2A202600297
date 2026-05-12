"""Phase B — LLM-as-Judge Pipeline with Swap-and-Average + Absolute Scoring.

Runs two RAG variants (top_k=3 vs top_k=5) on 30 questions from testset_v1.csv,
judges pairwise with bias mitigation, scores absolutely on 4 dimensions.

Outputs:
  phase-b/pairwise_results.csv   — question, answer_a, answer_b, run1_winner,
                                    run2_winner, winner_after_swap
  phase-b/absolute_scores.csv    — accuracy, relevance, conciseness, helpfulness, overall
  phase-b/human_labels.csv       — 10 sample pairs for manual labeling

Usage:
    python phase-b/judge_pipeline.py
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TESTSET_CSV = Path(__file__).parent.parent / "phase-a" / "testset_v1.csv"
PAIRWISE_CSV = Path(__file__).parent / "pairwise_results.csv"
ABSOLUTE_CSV = Path(__file__).parent / "absolute_scores.csv"
HUMAN_LABELS_CSV = Path(__file__).parent / "human_labels.csv"

N_QUESTIONS = 30
TOP_K_A = 3
TOP_K_B = 5

Winner = Literal["A", "B", "tie"]

# ─── RAG variants ─────────────────────────────────────────


_search = None
_reranker = None


def _build_rag() -> None:
    global _search, _reranker
    import rag.adapter as adapter

    logger.info("Building RAG pipeline...")
    adapter.build()
    _search = adapter._search
    _reranker = adapter._reranker
    logger.info("RAG ready")


def _generate_answer(query: str, contexts: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    context_str = "\n\n".join(contexts)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Trả lời CHỈ dựa trên context cho sẵn. "
                    "Nếu context không có thông tin → trả lời 'Không tìm thấy.'"
                ),
            },
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def rag_version_a(question: str) -> tuple[str, list[str]]:
    """RAG variant A — top_k=3 (default)."""
    assert _search and _reranker, "Call _build_rag() first"
    results = _search.search(question)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = _reranker.rerank(question, docs, top_k=TOP_K_A)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:TOP_K_A]]
    return _generate_answer(question, contexts), contexts


def rag_version_b(question: str) -> tuple[str, list[str]]:
    """RAG variant B — top_k=5 (wider retrieval)."""
    assert _search and _reranker, "Call _build_rag() first"
    results = _search.search(question)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = _reranker.rerank(question, docs, top_k=TOP_K_B)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:TOP_K_B]]
    return _generate_answer(question, contexts), contexts


# ─── Judge prompts ────────────────────────────────────────

_PAIRWISE_SYSTEM = """You are an impartial evaluator. Compare two answers to the same question.
IMPORTANT: Do NOT favor longer answers. Focus ONLY on:
- Factual accuracy and groundedness
- Direct relevance to the question
- Clarity and conciseness

Ignore formatting, length, and writing style.
Respond with valid JSON only: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}"""

_PAIRWISE_USER = """Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Respond with JSON only."""

_ABSOLUTE_SYSTEM = """You are an impartial evaluator. Score the answer on 4 dimensions (1-5 scale):
- accuracy: Is the answer factually correct and grounded?
- relevance: Does it directly address the question?
- conciseness: Is it appropriately brief without losing key info?
- helpfulness: Would a real user find this useful?

Respond with valid JSON only:
{"accuracy": <1-5>, "relevance": <1-5>, "conciseness": <1-5>, "helpfulness": <1-5>, "reason": "<one sentence>"}"""

_ABSOLUTE_USER = """Question: {question}

Answer:
{answer}

Score this answer. Respond with JSON only."""


# ─── JSON parsing ─────────────────────────────────────────


def parse_judge_output(text: str) -> dict:
    """Robust JSON extraction from LLM output (handles ```json fences)."""
    # Strip code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try full parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Keyword fallback
    text_upper = text.upper()
    if '"A"' in text or "ANSWER A" in text_upper or "WINNER: A" in text_upper:
        return {"winner": "A", "reason": text[:200]}
    if '"B"' in text or "ANSWER B" in text_upper or "WINNER: B" in text_upper:
        return {"winner": "B", "reason": text[:200]}

    return {"winner": "tie", "reason": text[:200]}


# ─── Pairwise judge ───────────────────────────────────────


def _call_judge(question: str, answer_a: str, answer_b: str) -> tuple[Winner, str]:
    """Single judge call → (winner, reason)."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _PAIRWISE_SYSTEM},
            {
                "role": "user",
                "content": _PAIRWISE_USER.format(
                    question=question, answer_a=answer_a, answer_b=answer_b
                ),
            },
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = parse_judge_output(resp.choices[0].message.content)
    winner = str(parsed.get("winner", "tie")).upper()
    if winner not in ("A", "B"):
        winner = "tie"
    return winner, parsed.get("reason", "")  # type: ignore[return-value]


def _flip(winner: Winner) -> Winner:
    """Flip A↔B for run2 (ans1/ans2 positions are swapped)."""
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


def pairwise_judge_with_swap(
    question: str, ans1: str, ans2: str
) -> tuple[Winner, Winner, Winner]:
    """Run swap-and-average. Returns (run1_winner, run2_winner, winner_after_swap).

    run1: ans1=A, ans2=B
    run2: ans1=B (now A), ans2=A (now B) → flip result back to original ordering
    Aggregate: both agree → that winner; disagree → tie.
    """
    # Run 1: ans1 listed as A
    run1_winner, _ = _call_judge(question, ans1, ans2)

    # Run 2: positions swapped — ans2 listed as A
    run2_raw, _ = _call_judge(question, ans2, ans1)
    run2_winner = _flip(run2_raw)  # flip back to original ordering

    # Aggregate
    if run1_winner == run2_winner:
        final: Winner = run1_winner  # type: ignore[assignment]
    else:
        final = "tie"

    return run1_winner, run2_winner, final


# ─── Absolute scoring ─────────────────────────────────────


def absolute_score(question: str, answer: str) -> dict:
    """Score answer on 4 dimensions (1-5). Returns dict with scores + overall."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _ABSOLUTE_SYSTEM},
            {
                "role": "user",
                "content": _ABSOLUTE_USER.format(question=question, answer=answer),
            },
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = parse_judge_output(resp.choices[0].message.content)

    def _safe_int(val: object, default: int = 3) -> int:
        try:
            v = int(float(str(val)))  # type: ignore[arg-type]
            return max(1, min(5, v))
        except (TypeError, ValueError):
            return default

    accuracy = _safe_int(parsed.get("accuracy"))
    relevance = _safe_int(parsed.get("relevance"))
    conciseness = _safe_int(parsed.get("conciseness"))
    helpfulness = _safe_int(parsed.get("helpfulness"))
    overall = round((accuracy + relevance + conciseness + helpfulness) / 4, 2)

    return {
        "accuracy": accuracy,
        "relevance": relevance,
        "conciseness": conciseness,
        "helpfulness": helpfulness,
        "overall": overall,
        "reason": str(parsed.get("reason", ""))[:300],
    }


# ─── Load testset ─────────────────────────────────────────


def load_questions(n: int = N_QUESTIONS) -> list[str]:
    if not TESTSET_CSV.exists():
        raise FileNotFoundError(
            f"{TESTSET_CSV} not found. Run phase-a/generate_testset.py first."
        )
    questions: list[str] = []
    with open(TESTSET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            if q:
                questions.append(q)
            if len(questions) >= n:
                break
    logger.info("Loaded %d questions from testset", len(questions))
    return questions


# ─── Save outputs ─────────────────────────────────────────


def save_pairwise(rows: list[dict], path: Path = PAIRWISE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "question", "answer_a", "answer_b",
              "run1_winner", "run2_winner", "winner_after_swap"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Pairwise results → %s (%d rows)", path, len(rows))


def save_absolute(rows: list[dict], path: Path = ABSOLUTE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "question", "version", "answer",
              "accuracy", "relevance", "conciseness", "helpfulness", "overall", "reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Absolute scores → %s (%d rows)", path, len(rows))


def save_human_labels_template(pairwise_rows: list[dict], path: Path = HUMAN_LABELS_CSV) -> None:
    """Save 10 random sample pairs for manual human labeling."""
    import random

    random.seed(42)
    sample = random.sample(pairwise_rows, min(10, len(pairwise_rows)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["question_id", "question", "answer_a", "answer_b",
              "human_winner", "confidence", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sample:
            w.writerow({
                "question_id": row["question_id"],
                "question": row["question"],
                "answer_a": row["answer_a"],
                "answer_b": row["answer_b"],
                "human_winner": "",        # ← fill manually
                "confidence": "",          # ← high / medium / low
                "notes": "",
            })
    logger.info(
        "Human labels template → %s (fill 'human_winner' column: A / B / tie)", path
    )


# ─── Main ─────────────────────────────────────────────────


def main() -> None:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Exiting.")
        sys.exit(1)

    _build_rag()
    questions = load_questions(N_QUESTIONS)

    pairwise_rows: list[dict] = []
    absolute_rows: list[dict] = []

    total = len(questions)
    for i, question in enumerate(questions, 1):
        logger.info("[%d/%d] %s", i, total, question[:70])

        # Get answers from both RAG variants
        t0 = time.time()
        ans_a, _ = rag_version_a(question)
        ans_b, _ = rag_version_b(question)
        rag_ms = (time.time() - t0) * 1000

        # Pairwise judge with swap
        run1, run2, final = pairwise_judge_with_swap(question, ans_a, ans_b)

        pairwise_rows.append({
            "question_id": i,
            "question": question,
            "answer_a": ans_a,
            "answer_b": ans_b,
            "run1_winner": run1,
            "run2_winner": run2,
            "winner_after_swap": final,
        })

        # Absolute scores for both versions
        score_a = absolute_score(question, ans_a)
        score_b = absolute_score(question, ans_b)

        for version, answer, score in [("A_top3", ans_a, score_a), ("B_top5", ans_b, score_b)]:
            absolute_rows.append({
                "question_id": i,
                "question": question,
                "version": version,
                "answer": answer,
                **score,
            })

        logger.info(
            "  RAG: %.0fms | run1=%s run2=%s final=%s | A_overall=%.2f B_overall=%.2f",
            rag_ms, run1, run2, final,
            score_a["overall"], score_b["overall"],
        )

    save_pairwise(pairwise_rows)
    save_absolute(absolute_rows)
    save_human_labels_template(pairwise_rows)

    # Summary stats
    finals = [r["winner_after_swap"] for r in pairwise_rows]
    a_wins = finals.count("A")
    b_wins = finals.count("B")
    ties = finals.count("tie")
    logger.info("=" * 55)
    logger.info("PAIRWISE SUMMARY (top_k=3 vs top_k=5)")
    logger.info("  A (top_k=3) wins: %d  B (top_k=5) wins: %d  ties: %d", a_wins, b_wins, ties)

    # Position bias check
    run1_a_wins = sum(1 for r in pairwise_rows if r["run1_winner"] == "A")
    pos_bias = run1_a_wins / len(pairwise_rows) * 100 if pairwise_rows else 0
    logger.info("  Position bias (A win rate run1): %.1f%%  (expected ~50%%)", pos_bias)
    logger.info("=" * 55)
    logger.info("Next: fill human_labels.csv, then run kappa_analysis.py")


if __name__ == "__main__":
    main()
