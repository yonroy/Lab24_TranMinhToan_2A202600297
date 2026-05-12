"""Phase A.2 — Run RAGAS evaluation on testset_v1.csv.

Loads testset, runs each question through the RAG pipeline (rag.adapter),
evaluates with 4 RAGAS metrics, saves per-question CSV + summary JSON.

Usage:
    python phase-a/run_ragas.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TESTSET_CSV = Path(__file__).parent / "testset_v1.csv"
RESULTS_CSV = Path(__file__).parent / "ragas_results.csv"
SUMMARY_JSON = Path(__file__).parent / "ragas_summary.json"

# SLOs from blueprint.md
SLOS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

METRICS = list(SLOS.keys())


# ─── Load testset ──────────────────────────────────────────


def load_testset(path: Path = TESTSET_CSV) -> list[dict]:
    """Load testset_v1.csv → list of {question, ground_truth, contexts}."""
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("Loaded %d test questions from %s", len(rows), path)
    return rows


# ─── RAG pipeline ─────────────────────────────────────────


def _run_pipeline(rows: list[dict]) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Run all questions through RAG adapter.

    Returns (questions, answers, all_contexts, ground_truths).
    """
    import rag.adapter as adapter

    logger.info("Building RAG pipeline (this takes ~30s)...")
    adapter.build()

    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []

    total = len(rows)
    for i, row in enumerate(rows, 1):
        q = row["question"].strip()
        gt = row["ground_truth"].strip()
        if not q:
            continue

        t0 = time.time()
        answer, contexts = adapter.my_rag_pipeline(q)
        latency_ms = (time.time() - t0) * 1000

        questions.append(q)
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(gt)

        logger.info("[%d/%d] %.0fms — %s", i, total, latency_ms, q[:60])

    return questions, answers, all_contexts, ground_truths


# ─── RAGAS evaluation ──────────────────────────────────────


def _build_ragas_components():
    """Return (llm, embeddings) wrapped for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
        )
        emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        )
        return llm, emb
    except ImportError:
        # ragas 0.2.x style — return raw langchain objects
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
        emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        return llm, emb


def run_ragas(
    questions: list[str],
    answers: list[str],
    all_contexts: list[list[str]],
    ground_truths: list[str],
) -> tuple[list[dict], dict]:
    """Evaluate with RAGAS. Returns (per_question_rows, summary)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    llm, emb = _build_ragas_components()

    # Assign LLM/embeddings to each metric
    for metric in [faithfulness, context_precision, context_recall]:
        metric.llm = llm  # type: ignore[attr-defined]
    answer_relevancy.llm = llm  # type: ignore[attr-defined]
    answer_relevancy.embeddings = emb  # type: ignore[attr-defined]

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": all_contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Running RAGAS evaluate() on %d samples...", len(questions))
    t0 = time.time()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False,
    )
    elapsed = time.time() - t0
    logger.info("RAGAS evaluation done in %.1fs", elapsed)

    df = result.to_pandas()

    def _safe_float(val: object) -> float:
        try:
            v = float(val)  # type: ignore[arg-type]
            return v if v == v else 0.0  # NaN check
        except (TypeError, ValueError):
            return 0.0

    per_question: list[dict] = []
    for _, row in df.iterrows():
        per_question.append(
            {
                "question": str(row.get("question", "")),
                "ground_truth": str(row.get("ground_truth", "")),
                "answer": str(row.get("answer", "")),
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
                "context_recall": _safe_float(row.get("context_recall")),
            }
        )

    summary: dict = {}
    for m in METRICS:
        vals = [r[m] for r in per_question]
        summary[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return per_question, summary


# ─── Output ────────────────────────────────────────────────


def save_results(per_question: list[dict], path: Path = RESULTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question", "ground_truth", "answer",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question)
    logger.info("Per-question results → %s", path)


def save_summary(summary: dict, per_question: list[dict], path: Path = SUMMARY_JSON) -> None:
    failures = _top_failures(per_question, n=10)
    payload = {
        "aggregate": summary,
        "num_questions": len(per_question),
        "slo_pass": {m: summary[m] >= SLOS[m] for m in METRICS},
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Summary → %s", path)


def _top_failures(per_question: list[dict], n: int = 10) -> list[dict]:
    def avg_score(r: dict) -> float:
        return sum(r[m] for m in METRICS) / len(METRICS)

    sorted_rows = sorted(per_question, key=avg_score)[:n]
    return [
        {
            "question": r["question"],
            "worst_metric": min(METRICS, key=lambda m: r[m]),
            "scores": {m: r[m] for m in METRICS},
        }
        for r in sorted_rows
    ]


# ─── Report ────────────────────────────────────────────────


def print_report(summary: dict) -> None:
    logger.info("=" * 55)
    logger.info("RAGAS RESULTS (Phase A)")
    logger.info("=" * 55)
    for m in METRICS:
        score = summary[m]
        threshold = SLOS[m]
        status = "PASS" if score >= threshold else "FAIL"
        logger.info("  [%s] %-22s %.4f  (SLO ≥ %.2f)", status, m, score, threshold)
    logger.info("=" * 55)


# ─── Main ──────────────────────────────────────────────────


def main() -> None:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Exiting.")
        sys.exit(1)

    if not TESTSET_CSV.exists():
        logger.error("%s not found. Run generate_testset.py first.", TESTSET_CSV)
        sys.exit(1)

    rows = load_testset()
    questions, answers, all_contexts, ground_truths = _run_pipeline(rows)

    per_question, summary = run_ragas(questions, answers, all_contexts, ground_truths)

    save_results(per_question)
    save_summary(summary, per_question)
    print_report(summary)

    logger.info("Done. Outputs: %s, %s", RESULTS_CSV.name, SUMMARY_JSON.name)


if __name__ == "__main__":
    main()
