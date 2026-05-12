# Phase B — LLM-as-Judge Bias Report

## Setup
- **Model compared:** top_k=3 (A) vs top_k=5 (B) — same RAG pipeline, different retrieval depth
- **Questions evaluated:** 30 pairwise comparisons
- **Judge LLM:** gpt-4o-mini with swap-and-average bias mitigation

---

## 1. Inter-Rater Agreement (Cohen's Kappa)

| Metric | Value |
|--------|-------|
| Human-labelled pairs | 10 |
| Cohen's kappa | **0.4828** |
| Interpretation | **Moderate agreement** |

**Analysis:** kappa = 0.48 indicates moderate agreement between human evaluators and the LLM judge. The judge is reliable enough for automated screening but should not replace human review for high-stakes decisions. Production-ready threshold is kappa >= 0.60 (substantial agreement).

---

## 2. Position Bias

| Metric | Value |
|--------|-------|
| A win rate in run1 (A listed first) | 26.7% |
| Expected if unbiased | ~50% |
| Position bias detected | **NO** |
| Tie rate before swap-and-average | 36.7% (11/30) |
| Tie rate after swap-and-average | 63.3% (19/30) |
| Tie rate increase | +26.7% |

**Analysis:** Counter-intuitively, the A win rate in run1 is *below* 50% (26.7%), suggesting a slight recency bias (B is listed second = more recent in context, slightly favoured). The swap-and-average effectively converted disagreements to ties, increasing tie rate by 26.7 percentage points. No systematic position bias detected (threshold: A win rate > 55%).

---

## 3. Length Bias

| Condition | B win rate |
|-----------|-----------|
| B answer longer than A | 46.7% |
| B answer shorter than A | 14.3% |
| Difference | **+32.4%** |
| Length bias detected | **YES** |

**Analysis:** The LLM judge shows significant length bias (+32.4% > 15% threshold). When B's answer is longer, B wins 46.7% of the time; when B's answer is shorter, B wins only 14.3% of the time. This aligns with known LLM judge behaviour where longer, more detailed answers are perceived as higher quality regardless of accuracy.

**Mitigation strategies:**
- Add explicit length-penalty instruction in judge prompt
- Normalize answer length before judging
- Use absolute scoring (quality rubric) alongside pairwise to cross-validate

---

## 4. Pairwise Summary

| Outcome | Count | % |
|---------|-------|---|
| A (top_k=3) wins | 3 | 10.0% |
| B (top_k=5) wins | 8 | 26.7% |
| Ties | 19 | 63.3% |

**Conclusion:** top_k=5 outperforms top_k=3 in 26.7% of cases with swap-and-average applied. The larger retrieval window (k=5) provides more context, enabling more complete answers — consistent with the length bias finding (B tends to produce longer answers when k=5 retrieves more chunks).

---

## 5. Absolute Scoring Summary

Absolute scores (0–10 scale) were collected for both versions across 30 questions (60 total rows in `absolute_scores.csv`). Absolute scoring cross-validates pairwise results and is less susceptible to length bias since it evaluates each answer independently.

---

*Generated: 2026-05-12 | Phase B — Lab 24 AICB-P2T3*
