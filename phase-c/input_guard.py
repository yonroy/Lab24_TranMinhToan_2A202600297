"""Phase C.1/C.2 — Input Guard: PII Redaction + Topic Validator.

PII redaction uses two layers:
  Layer 1: VN-specific regex patterns (CCCD, phone, tax code, email)
  Layer 2: Microsoft Presidio NER (PERSON, LOCATION, ORG, CREDIT_CARD, ...)

Topic validator uses LLM zero-shot (gpt-4o-mini).

All methods return (result, latency_ms) per CLAUDE.md pattern.

Usage:
    python phase-c/input_guard.py          # runs PII + topic tests, saves CSVs
"""

from __future__ import annotations

import csv
import logging
import re
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

PII_CSV = Path(__file__).parent / "pii_test_results.csv"

# ─── Domain config ────────────────────────────────────────

ALLOWED_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "RAG",
    "retrieval augmented generation",
    "large language models",
    "vector databases",
    "embeddings",
    "VinUniversity",
    "education",
    "research",
]

OUT_OF_SCOPE_MESSAGE = (
    "Câu hỏi này nằm ngoài phạm vi hỗ trợ của hệ thống. "
    "Tôi có thể giúp bạn về AI, Machine Learning, RAG, và các chủ đề liên quan. "
    "Bạn có muốn thử lại không?"
)

# ─── VN PII regex patterns ────────────────────────────────

_VN_PII: dict[str, re.Pattern[str]] = {
    "CCCD":      re.compile(r"\b\d{12}\b"),
    "PHONE_VN":  re.compile(r"(\+84|0)\d{9,10}"),
    "TAX_CODE":  re.compile(r"\b\d{10}(-\d{3})?\b"),
    "EMAIL":     re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b"),
}

_PLACEHOLDER: dict[str, str] = {
    "CCCD":     "<CCCD_REDACTED>",
    "PHONE_VN": "<PHONE_REDACTED>",
    "TAX_CODE": "<TAX_CODE_REDACTED>",
    "EMAIL":    "<EMAIL_REDACTED>",
}


def _redact_vn_regex(text: str) -> tuple[str, list[str]]:
    """Apply VN regex PII patterns. Returns (redacted_text, list_of_detected_types)."""
    detected: list[str] = []
    for label, pattern in _VN_PII.items():
        if pattern.search(text):
            detected.append(label)
            text = pattern.sub(_PLACEHOLDER[label], text)
    return text, detected


# ─── Presidio ─────────────────────────────────────────────

_presidio_analyzer = None
_presidio_anonymizer = None


def _get_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            _presidio_analyzer = AnalyzerEngine()
            _presidio_anonymizer = AnonymizerEngine()
        except ImportError:
            logger.warning("presidio not installed — Presidio layer skipped.")
    return _presidio_analyzer, _presidio_anonymizer


def _redact_presidio(text: str) -> tuple[str, list[str]]:
    """Apply Presidio NER PII detection. Returns (redacted_text, list_of_entity_types)."""
    analyzer, anonymizer = _get_presidio()
    if analyzer is None or anonymizer is None:
        return text, []

    try:
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return text, []
        detected = list({r.entity_type for r in results})
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, detected
    except Exception as e:
        logger.warning("Presidio error: %s", e)
        return text, []


# ─── InputGuard class ─────────────────────────────────────


class InputGuard:
    """PII redaction + topic validation for user inputs."""

    def __init__(self, allowed_topics: list[str] | None = None) -> None:
        self._allowed_topics = allowed_topics or ALLOWED_TOPICS
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0,
            )
        return self._llm

    # ── PII sanitize ─────────────────────────────────────

    def sanitize(self, text: str) -> tuple[str, float]:
        """Redact PII using VN regex + Presidio.

        Returns:
            (sanitized_text, latency_ms)
        """
        t0 = time.time()
        if not text or not text.strip():
            return text, (time.time() - t0) * 1000

        # Layer 1: VN regex
        redacted, _ = _redact_vn_regex(text)

        # Layer 2: Presidio NER
        redacted, _ = _redact_presidio(redacted)

        return redacted, (time.time() - t0) * 1000

    async def sanitize_async(self, text: str) -> tuple[str, float]:
        """Async wrapper for sanitize()."""
        import asyncio
        return await asyncio.to_thread(self.sanitize, text)

    def detect_pii_types(self, text: str) -> list[str]:
        """Return list of PII entity types found (without redacting)."""
        detected: list[str] = []
        for label, pattern in _VN_PII.items():
            if pattern.search(text):
                detected.append(label)
        _, presidio_types = _redact_presidio(text)
        detected.extend(presidio_types)
        return list(set(detected))

    # ── Topic validation ──────────────────────────────────

    def topic_check(self, text: str) -> tuple[bool, float]:
        """LLM zero-shot topic classifier.

        Returns:
            (is_on_topic, latency_ms)
        """
        t0 = time.time()
        if not text or not text.strip():
            return True, (time.time() - t0) * 1000

        topics_str = ", ".join(self._allowed_topics)
        prompt = (
            f"Is this question about one of these topics: {topics_str}?\n"
            f"Answer YES or NO only.\n"
            f"Question: {text}"
        )
        try:
            llm = self._get_llm()
            response = llm.invoke(prompt).content.strip()
            is_on_topic = response.upper().startswith("YES")
        except Exception as e:
            logger.warning("Topic check LLM error: %s — defaulting to True", e)
            is_on_topic = True

        return is_on_topic, (time.time() - t0) * 1000

    async def topic_check_async(self, text: str) -> tuple[bool, float]:
        """Async wrapper for topic_check()."""
        import asyncio
        return await asyncio.to_thread(self.topic_check, text)


# ─── PII test set ─────────────────────────────────────────

_PII_TESTS: list[dict] = [
    # id, input, expected_pii_types (partial — for reference)
    {"id": 1,  "input": "Số CCCD của tôi là 012345678901.",                 "has_pii": True},
    {"id": 2,  "input": "Liên hệ qua 0912345678 hoặc +84987654321.",        "has_pii": True},
    {"id": 3,  "input": "Email: nguyen.van.a@gmail.com để biết thêm.",       "has_pii": True},
    {"id": 4,  "input": "Mã số thuế doanh nghiệp: 0123456789.",              "has_pii": True},
    {"id": 5,  "input": "John Smith lives at 123 Main St.",                  "has_pii": True},
    {"id": 6,  "input": "CCCD 034567890123, SĐT 0398765432, email a@b.vn.", "has_pii": True},
    {"id": 7,  "input": "",                                                   "has_pii": False},
    {"id": 8,  "input": "A" * 5000 + " gọi 0912000001 để biết thêm.",       "has_pii": True},
    {"id": 9,  "input": "Hệ thống RAG sử dụng ChromaDB và LangChain.",      "has_pii": False},
    {"id": 10, "input": "Câu hỏi về mô hình ngôn ngữ lớn gpt-4o-mini.",     "has_pii": False},
]

_TOPIC_TESTS: list[dict] = [
    # On-topic (10)
    {"id": 1,  "input": "What is RAG and how does it work?",                              "expected": True},
    {"id": 2,  "input": "Explain vector embeddings for NLP.",                             "expected": True},
    {"id": 3,  "input": "What is a large language model?",                                "expected": True},
    {"id": 4,  "input": "How does ChromaDB store embeddings?",                            "expected": True},
    {"id": 5,  "input": "What are the evaluation metrics for RAG pipelines?",             "expected": True},
    {"id": 6,  "input": "Giải thích retrieval augmented generation bằng tiếng Việt.",    "expected": True},
    {"id": 7,  "input": "VinUniversity AI research focus areas?",                         "expected": True},
    {"id": 8,  "input": "How to fine-tune a language model?",                             "expected": True},
    {"id": 9,  "input": "What is RAGAS evaluation framework?",                            "expected": True},
    {"id": 10, "input": "Explain the difference between BM25 and dense retrieval.",       "expected": True},
    # Off-topic (10)
    {"id": 11, "input": "What is the recipe for chocolate cake?",                         "expected": False},
    {"id": 12, "input": "Who won the 2026 World Cup?",                                    "expected": False},
    {"id": 13, "input": "How do I fix my car engine?",                                    "expected": False},
    {"id": 14, "input": "What is the best investment strategy for stocks?",               "expected": False},
    {"id": 15, "input": "Tell me about the history of ancient Rome.",                     "expected": False},
    {"id": 16, "input": "What are the symptoms of flu?",                                  "expected": False},
    {"id": 17, "input": "How to grow tomatoes in a garden?",                              "expected": False},
    {"id": 18, "input": "What is the population of Vietnam?",                             "expected": False},
    {"id": 19, "input": "How do I apply for a visa to Japan?",                            "expected": False},
    {"id": 20, "input": "Recommend a good action movie.",                                 "expected": False},
]


def run_pii_tests(guard: InputGuard) -> list[dict]:
    """Run 10 PII test cases. Returns result rows."""
    rows: list[dict] = []
    detected_count = 0

    for case in _PII_TESTS:
        text = case["input"]
        sanitized, latency_ms = guard.sanitize(text)
        pii_types = guard.detect_pii_types(text) if text else []
        detected = len(pii_types) > 0
        changed = sanitized != text

        if case["has_pii"] and detected:
            detected_count += 1

        rows.append({
            "id": case["id"],
            "input_preview": text[:80].replace("\n", " "),
            "has_pii_expected": case["has_pii"],
            "pii_types_detected": "|".join(pii_types),
            "detected": detected,
            "redacted": changed,
            "output_preview": sanitized[:80].replace("\n", " "),
            "latency_ms": round(latency_ms, 2),
        })
        logger.info(
            "[PII %d] detected=%s redacted=%s latency=%.1fms types=%s",
            case["id"], detected, changed, latency_ms, pii_types,
        )

    pii_cases = sum(1 for c in _PII_TESTS if c["has_pii"])
    logger.info("PII detection: %d/%d (%.0f%%)", detected_count, pii_cases,
                detected_count / pii_cases * 100 if pii_cases else 0)
    return rows


def run_topic_tests(guard: InputGuard) -> list[dict]:
    """Run 20 topic test cases. Returns result rows."""
    rows: list[dict] = []
    correct = 0

    for case in _TOPIC_TESTS:
        is_on_topic, latency_ms = guard.topic_check(case["input"])
        hit = is_on_topic == case["expected"]
        if hit:
            correct += 1
        rows.append({
            "id": case["id"],
            "input": case["input"],
            "expected_on_topic": case["expected"],
            "predicted_on_topic": is_on_topic,
            "correct": hit,
            "latency_ms": round(latency_ms, 2),
        })
        logger.info(
            "[Topic %d] expected=%s predicted=%s %s latency=%.1fms",
            case["id"], case["expected"], is_on_topic,
            "✓" if hit else "✗", latency_ms,
        )

    logger.info("Topic accuracy: %d/20 (%.0f%%)", correct, correct / 20 * 100)
    return rows


def save_pii_csv(rows: list[dict], path: Path = PII_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "input_preview", "has_pii_expected", "pii_types_detected",
              "detected", "redacted", "output_preview", "latency_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("PII test results → %s", path)


def main() -> None:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    guard = InputGuard()

    logger.info("=== C.1 PII Redaction Tests ===")
    pii_rows = run_pii_tests(guard)
    save_pii_csv(pii_rows)

    logger.info("\n=== C.2 Topic Validation Tests ===")
    run_topic_tests(guard)

    logger.info("\nDone. Next: python phase-c/output_guard.py")


if __name__ == "__main__":
    main()
