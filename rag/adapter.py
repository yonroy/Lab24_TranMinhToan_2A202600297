"""RAG Adapter — interface chuẩn cho Lab 24.

Khởi tạo pipeline 1 lần, tái dùng cho Phase A, B, C.
Gọi build() trước khi dùng my_rag_pipeline().
"""

from __future__ import annotations

from rag.pipeline import build_pipeline, run_query
from rag.m2_search import HybridSearch
from rag.m3_rerank import CrossEncoderReranker

_search: HybridSearch | None = None
_reranker: CrossEncoderReranker | None = None


def build() -> None:
    """Load documents, build index, load reranker. Gọi 1 lần khi startup."""
    global _search, _reranker
    _search, _reranker = build_pipeline(use_enrichment=False)


def my_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """Interface chuẩn cho Lab 24.

    Returns:
        (answer, contexts) — contexts là list of retrieved chunk strings.
    """
    if _search is None or _reranker is None:
        raise RuntimeError("RAG chưa được khởi tạo. Gọi rag.adapter.build() trước.")
    return run_query(question, _search, _reranker)
