"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


_RAGAS_EMBEDDINGS = None
_RAGAS_LLM = None


def _get_ragas_embeddings():
    """Build an OpenAI embedding backend with LangChain interface (embed_query/embed_documents)."""
    global _RAGAS_EMBEDDINGS
    if _RAGAS_EMBEDDINGS is None:
        from langchain_openai import OpenAIEmbeddings
        _RAGAS_EMBEDDINGS = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
        )
    return _RAGAS_EMBEDDINGS


def _get_ragas_llm():
    """Build an OpenAI LLM backend for RAGAS metrics."""
    global _RAGAS_LLM
    if _RAGAS_LLM is None:
        from openai import OpenAI
        from ragas.llms import llm_factory
        client = OpenAI(api_key=OPENAI_API_KEY)
        _RAGAS_LLM = llm_factory("gpt-4o-mini", client=client)
    return _RAGAS_LLM


def _build_ragas_metrics():
    """Build metric objects compatible with ragas.evaluate() (must be Metric instances)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall
        )

    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    faithfulness.llm = llm
    context_precision.llm = llm
    context_recall.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    return [faithfulness, answer_relevancy, context_precision, context_recall]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation on (Q, A, contexts, GT) tuples."""
    empty = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if not questions:
        return empty

    try:
        from ragas import evaluate
        from datasets import Dataset
    except ImportError as e:
        print(f"[m4_eval] RAGAS not available: {e}")
        return empty

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    try:
        result = evaluate(
            dataset,
            metrics=_build_ragas_metrics(),
            raise_exceptions=False,
        )
        df = result.to_pandas()
    except Exception as e:
        print(f"[m4_eval] RAGAS evaluate failed: {e}")
        return empty

    def _f(row, key):
        v = row.get(key, 0.0)
        try:
            return float(v) if v == v else 0.0  # NaN-safe
        except (TypeError, ValueError):
            return 0.0

    per_question = [
        EvalResult(
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            contexts=row.get("contexts", []) if isinstance(row.get("contexts"), list)
                    else list(row.get("contexts", [])),
            ground_truth=row.get("ground_truth", ""),
            faithfulness=_f(row, "faithfulness"),
            answer_relevancy=_f(row, "answer_relevancy"),
            context_precision=_f(row, "context_precision"),
            context_recall=_f(row, "context_recall"),
        )
        for _, row in df.iterrows()
    ]

    def _mean(key):
        vals = [getattr(r, key) for r in per_question]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
        "per_question": per_question,
    }


_DIAGNOSIS = [
    ("faithfulness", 0.85, "LLM hallucinating",
     "Tighten prompt: 'Trả lời CHỈ dựa trên context'. Lower temperature to 0."),
    ("context_recall", 0.75, "Missing relevant chunks",
     "Improve chunking (try hierarchical) or add BM25 to catch keyword matches."),
    ("context_precision", 0.75, "Too many irrelevant chunks",
     "Add cross-encoder reranking or metadata filtering before generation."),
    ("answer_relevancy", 0.80, "Answer doesn't match question intent",
     "Improve prompt template; ensure question is restated in the answer."),
]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    metrics_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def avg(r: EvalResult) -> float:
        return sum(getattr(r, k) for k in metrics_keys) / len(metrics_keys)

    sorted_results = sorted(eval_results, key=avg)[:bottom_n]

    failures: list[dict] = []
    for r in sorted_results:
        scores = {k: getattr(r, k) for k in metrics_keys}
        worst_metric = min(scores, key=scores.get)
        worst_score = scores[worst_metric]

        diagnosis = "Mixed failure mode"
        suggested_fix = "Inspect contexts and answer manually."
        for metric, threshold, diag, fix in _DIAGNOSIS:
            if metric == worst_metric and worst_score < threshold:
                diagnosis, suggested_fix = diag, fix
                break

        failures.append({
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "scores": scores,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
