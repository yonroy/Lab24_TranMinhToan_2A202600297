"""Phase B.3/B.4 — Cohen's Kappa + Bias Analysis.

Reads pairwise_results.csv + human_labels.csv, computes inter-rater agreement,
analyses position bias and length bias, prints structured report.

Usage:
    python phase-b/kappa_analysis.py

Prerequisites:
    1. Run phase-b/judge_pipeline.py first
    2. Fill human_winner column in phase-b/human_labels.csv (A / B / tie)
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PAIRWISE_CSV = Path(__file__).parent / "pairwise_results.csv"
HUMAN_LABELS_CSV = Path(__file__).parent / "human_labels.csv"


# ─── Load ─────────────────────────────────────────────────


def load_pairwise(path: Path = PAIRWISE_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run judge_pipeline.py first.")
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("Loaded %d pairwise rows", len(rows))
    return rows


def load_human_labels(path: Path = HUMAN_LABELS_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run judge_pipeline.py first, then fill human_winner column."
        )
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ─── Cohen's Kappa ────────────────────────────────────────


def _normalize_label(label: str) -> str:
    label = label.strip().upper()
    if label in ("A", "B"):
        return label
    return "tie"


def compute_kappa(human_labels: list[dict], pairwise_rows: list[dict]) -> float:
    """Compute Cohen's kappa between human labels and judge labels.

    Matches on question_id. Skips rows with empty human_winner.
    """
    # Build judge lookup: question_id → winner_after_swap
    judge_map: dict[str, str] = {
        str(r["question_id"]): _normalize_label(r["winner_after_swap"])
        for r in pairwise_rows
    }

    human: list[str] = []
    judge: list[str] = []

    for row in human_labels:
        h = row.get("human_winner", "").strip()
        if not h:
            continue  # skip unfilled rows
        qid = str(row.get("question_id", ""))
        if qid not in judge_map:
            continue
        human.append(_normalize_label(h))
        judge.append(judge_map[qid])

    if len(human) < 2:
        logger.warning("Fewer than 2 matched human labels — kappa not computable.")
        return float("nan")

    try:
        from sklearn.metrics import cohen_kappa_score

        kappa = float(cohen_kappa_score(human, judge))
    except Exception as e:
        logger.warning("sklearn kappa failed (%s), computing manually.", e)
        kappa = _kappa_manual(human, judge)

    return kappa


def _kappa_manual(y1: list[str], y2: list[str]) -> float:
    """Manual Cohen's kappa for 3 classes."""
    labels = ["A", "B", "tie"]
    n = len(y1)
    if n == 0:
        return float("nan")

    observed = sum(a == b for a, b in zip(y1, y2)) / n

    expected = sum(
        (y1.count(lbl) / n) * (y2.count(lbl) / n) for lbl in labels
    )

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def interpret_kappa(kappa: float) -> str:
    if kappa != kappa:  # NaN
        return "not computable"
    if kappa < 0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight agreement"
    if kappa < 0.40:
        return "fair agreement"
    if kappa < 0.60:
        return "moderate agreement"
    if kappa < 0.80:
        return "substantial agreement (production-ready ✓)"
    return "almost perfect agreement"


# ─── Position bias ────────────────────────────────────────


def analyze_position_bias(rows: list[dict]) -> dict:
    """Measure how often A wins in run1 (A is listed first)."""
    total = len(rows)
    if total == 0:
        return {}

    run1_a = sum(1 for r in rows if _normalize_label(r.get("run1_winner", "")) == "A")
    run1_b = sum(1 for r in rows if _normalize_label(r.get("run1_winner", "")) == "B")
    run1_tie = total - run1_a - run1_b

    a_rate = run1_a / total * 100
    has_bias = a_rate > 55.0

    # After swap: ties should increase if bias existed
    final_a = sum(1 for r in rows if _normalize_label(r.get("winner_after_swap", "")) == "A")
    final_b = sum(1 for r in rows if _normalize_label(r.get("winner_after_swap", "")) == "B")
    final_tie = total - final_a - final_b

    tie_increase = (final_tie - run1_tie) / total * 100

    return {
        "total": total,
        "run1_a_wins": run1_a,
        "run1_b_wins": run1_b,
        "run1_ties": run1_tie,
        "a_win_rate_run1_pct": round(a_rate, 1),
        "has_position_bias": has_bias,
        "final_a_wins": final_a,
        "final_b_wins": final_b,
        "final_ties": final_tie,
        "tie_increase_pct": round(tie_increase, 1),
    }


# ─── Length bias ──────────────────────────────────────────


def analyze_length_bias(rows: list[dict]) -> dict:
    """Measure if judge favours the longer answer."""
    b_longer_wins = 0
    b_longer_total = 0
    b_shorter_wins = 0
    b_shorter_total = 0

    for r in rows:
        len_a = len(str(r.get("answer_a", "")))
        len_b = len(str(r.get("answer_b", "")))
        winner = _normalize_label(r.get("winner_after_swap", ""))

        if len_b > len_a:
            b_longer_total += 1
            if winner == "B":
                b_longer_wins += 1
        elif len_b < len_a:
            b_shorter_total += 1
            if winner == "B":
                b_shorter_wins += 1

    def _rate(wins: int, total: int) -> float:
        return round(wins / total * 100, 1) if total > 0 else 0.0

    longer_rate = _rate(b_longer_wins, b_longer_total)
    shorter_rate = _rate(b_shorter_wins, b_shorter_total)
    diff = round(longer_rate - shorter_rate, 1)
    has_bias = diff > 15.0  # >15% difference = notable length bias

    return {
        "b_longer_total": b_longer_total,
        "b_longer_wins": b_longer_wins,
        "b_longer_win_rate_pct": longer_rate,
        "b_shorter_total": b_shorter_total,
        "b_shorter_wins": b_shorter_wins,
        "b_shorter_win_rate_pct": shorter_rate,
        "length_bias_diff_pct": diff,
        "has_length_bias": has_bias,
    }


# ─── Report ───────────────────────────────────────────────


def print_report(
    kappa: float,
    pos: dict,
    length: dict,
    n_human: int,
) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("PHASE B — BIAS REPORT")
    logger.info(sep)

    # Kappa
    logger.info("\n[1] COHEN'S KAPPA (human vs judge)")
    logger.info("    Pairs labelled by human: %d", n_human)
    if kappa == kappa:
        logger.info("    kappa = %.4f → %s", kappa, interpret_kappa(kappa))
    else:
        logger.info("    kappa = n/a (fill human_labels.csv first)")

    # Position bias
    logger.info("\n[2] POSITION BIAS (A listed first in run1)")
    if pos:
        logger.info("    A win rate (run1): %.1f%%  (expected ~50%%)", pos["a_win_rate_run1_pct"])
        logger.info(
            "    Bias detected: %s", "YES ⚠" if pos["has_position_bias"] else "NO ✓"
        )
        logger.info(
            "    After swap-and-average: tie rate +%.1f%% (%d → %d ties)",
            pos["tie_increase_pct"], pos["run1_ties"], pos["final_ties"],
        )

    # Length bias
    logger.info("\n[3] LENGTH BIAS (B favoured when longer)")
    if length:
        logger.info(
            "    B win rate when B longer: %.1f%%  when B shorter: %.1f%%",
            length["b_longer_win_rate_pct"], length["b_shorter_win_rate_pct"],
        )
        logger.info(
            "    Difference: %.1f%%  Bias detected: %s",
            length["length_bias_diff_pct"],
            "YES ⚠" if length["has_length_bias"] else "NO ✓",
        )

    logger.info(sep)
    logger.info("Fill in judge_bias_report.md with the numbers above.")


# ─── Main ─────────────────────────────────────────────────


def main() -> None:
    rows = load_pairwise()

    # Position + length bias (no human labels needed)
    pos = analyze_position_bias(rows)
    length = analyze_length_bias(rows)

    # Kappa (requires human labels)
    kappa = float("nan")
    n_human = 0
    try:
        human_rows = load_human_labels()
        filled = [r for r in human_rows if r.get("human_winner", "").strip()]
        n_human = len(filled)
        if n_human > 0:
            kappa = compute_kappa(filled, rows)
        else:
            logger.warning(
                "human_labels.csv has no filled rows yet. "
                "Open the file and fill the 'human_winner' column (A / B / tie)."
            )
    except FileNotFoundError as e:
        logger.warning("%s", e)

    print_report(kappa, pos, length, n_human)


if __name__ == "__main__":
    main()
