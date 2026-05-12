"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


_SEMANTIC_ENCODER = None


def _get_semantic_encoder():
    global _SEMANTIC_ENCODER
    if _SEMANTIC_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_ENCODER


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    encoder = _get_semantic_encoder()
    embeddings = encoder.encode(sentences, show_progress_bar=False)

    from numpy import dot
    from numpy.linalg import norm

    def cosine_sim(a, b):
        denom = norm(a) * norm(b)
        return float(dot(a, b) / denom) if denom else 0.0

    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Default recommendation cho production RAG.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current = ""
    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            ))
            current = ""
        current += para + "\n\n"
    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))

    for parent in parents:
        pid = parent.metadata["parent_id"]
        ptext = parent.text
        step = max(child_size, 1)
        for start in range(0, len(ptext), step):
            child_text = ptext[start:start + step].strip()
            if not child_text:
                continue
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "chunk_index": len(children)},
                parent_id=pid,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists.
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush():
        nonlocal current_header, current_content
        body = current_content.strip()
        if not body and not current_header:
            return
        chunk_text = f"{current_header}\n{body}".strip() if current_header else body
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "section": current_header.strip() or "(no header)",
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))

    for part in sections:
        if re.match(r'^#{1,6}\s+', part or ""):
            if current_content.strip() or current_header:
                flush()
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part or ""
    if current_content.strip() or current_header:
        flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def _stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": sum(lengths) // len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }

def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and compare."""
    basic, semantic, hier_p, hier_c, structure = [], [], [], [], []
    for doc in documents:
        text, meta = doc["text"], doc.get("metadata", {})
        basic.extend(chunk_basic(text, metadata=meta))
        try:
            semantic.extend(chunk_semantic(text, metadata=meta))
        except Exception:
            pass
        p, c = chunk_hierarchical(text, metadata=meta)
        hier_p.extend(p)
        hier_c.extend(c)
        structure.extend(chunk_structure_aware(text, metadata=meta))

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(hier_c), "num_parents": len(hier_p)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<14} | {'Chunks':>7} | {'Avg':>5} | {'Min':>4} | {'Max':>5}")
    print("-" * 50)
    for name, s in results.items():
        chunks_label = (f"{s.get('num_parents', 0)}p/{s['num_chunks']}c"
                        if name == "hierarchical" else str(s["num_chunks"]))
        print(f"{name:<14} | {chunks_label:>7} | {s['avg_length']:>5} | "
              f"{s['min_length']:>4} | {s['max_length']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
