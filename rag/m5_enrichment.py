"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _has_openai() -> bool:
    return bool(OPENAI_API_KEY)


def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """LLM tóm tắt chunk → giảm noise. Extractive fallback nếu không có API."""
    if not text.strip():
        return ""

    if _has_openai():
        try:
            client = _openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[summarize_chunk] OpenAI failed, falling back: {e}")

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return (". ".join(sentences[:2]) + ".") if sentences else text[:200]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate câu hỏi mà chunk có thể trả lời. Bridge vocabulary gap."""
    if not text.strip():
        return []

    if _has_openai():
        try:
            client = _openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": (f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn "
                                 f"có thể trả lời. Mỗi câu hỏi 1 dòng, không đánh số.")},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ").strip()
                    for q in raw if q.strip()][:n_questions]
        except Exception as e:
            print(f"[generate_hypothesis_questions] OpenAI failed, falling back: {e}")

    # Fallback: build trivial template questions from the first sentence
    first = re.split(r'(?<=[.!?])\s+', text.strip())[0] if text.strip() else ""
    return [f"Đoạn văn nói về điều gì? ({first[:60]}...)"][:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend 1-câu mô tả vị trí + chủ đề. Anthropic: -49% retrieval failure."""
    if not text.strip():
        return text

    if _has_openai():
        try:
            client = _openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": ("Viết 1 câu ngắn (tiếng Việt) mô tả đoạn văn này nằm ở "
                                 "tài liệu nào và nói về chủ đề gì. CHỈ trả về 1 câu.")},
                    {"role": "user",
                     "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"[contextual_prepend] OpenAI failed, falling back: {e}")

    prefix = f"Trích từ tài liệu: {document_title}." if document_title else "Trích từ tài liệu."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """LLM extract: topic, entities, category, language."""
    if not text.strip():
        return {}

    if _has_openai():
        try:
            client = _openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": ('Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ: '
                                 '{"topic": "...", "entities": ["..."], '
                                 '"category": "policy|hr|it|finance|other", '
                                 '"language": "vi|en"}')},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"[extract_metadata] OpenAI failed, falling back: {e}")

    # Heuristic fallback
    has_vi = bool(re.search(r'[ăâđêôơưĂÂĐÊÔƠƯ]', text))
    return {
        "topic": "",
        "entities": [],
        "category": "other",
        "language": "vi" if has_vi else "en",
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Chạy enrichment pipeline trên list chunks."""
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    do_summary = "summary" in methods or "full" in methods
    do_hyqa = "hyqa" in methods or "full" in methods
    do_contextual = "contextual" in methods or "full" in methods
    do_metadata = "metadata" in methods or "full" in methods

    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {}) or {}
        title = meta.get("source", "")

        summary = summarize_chunk(text) if do_summary else ""
        questions = generate_hypothesis_questions(text) if do_hyqa else []
        enriched_text = contextual_prepend(text, title) if do_contextual else text
        auto_meta = extract_metadata(text) if do_metadata else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text or text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = ("Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
              "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác.")

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên VinUni 2024')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
