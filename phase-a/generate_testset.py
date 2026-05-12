"""Phase A.1 — Generate testset using RAGAS TestsetGenerator.

Loads corpus from DATA_DIR (Day 18 markdown files),
generates 50 Q/A/context triples, saves to phase-a/testset_v1.csv.

Usage:
    python phase-a/generate_testset.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OUTPUT_CSV = Path(__file__).parent / "testset_v1.csv"
TEST_SIZE = 50


def _load_langchain_documents() -> list:
    """Load markdown files from DATA_DIR as LangChain Document objects."""
    from langchain_core.documents import Document

    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {data_dir}")

    docs: list[Document] = []
    for fp in md_files:
        text = fp.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": fp.name}))

    logger.info("Loaded %d documents from %s", len(docs), data_dir)
    return docs


def generate_testset(test_size: int = TEST_SIZE) -> list[dict]:
    """Generate test set using RAGAS TestsetGenerator.

    Returns list of dicts with keys: question, ground_truth, contexts, source.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    docs = _load_langchain_documents()

    logger.info("Initialising RAGAS TestsetGenerator (test_size=%d)...", test_size)

    # ragas 0.3.x API
    try:
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        generator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
        )
        generator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        )
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        logger.info("Using ragas 0.3.x TestsetGenerator API")

    except (ImportError, TypeError):
        # Fallback: ragas 0.2.x API
        from ragas.testset.generator import TestsetGenerator  # type: ignore[no-redef]
        from ragas.testset.evolutions import simple, reasoning, multi_context  # type: ignore

        generator = TestsetGenerator.from_langchain(
            generator_llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
            critic_llm=ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY),
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=OPENAI_API_KEY
            ),
        )
        logger.info("Using ragas 0.2.x TestsetGenerator API (fallback)")

    t0 = time.time()
    logger.info("Generating %d test samples — this may take a few minutes...", test_size)

    try:
        testset = generator.generate_with_langchain_docs(
            docs, testset_size=test_size
        )
    except TypeError:
        # Some versions use test_size= parameter name
        testset = generator.generate_with_langchain_docs(docs, test_size=test_size)

    elapsed = time.time() - t0
    logger.info("Generation complete in %.1fs", elapsed)

    df = testset.to_pandas()
    logger.info("Generated %d samples", len(df))

    rows: list[dict] = []
    for _, row in df.iterrows():
        contexts = row.get("contexts", row.get("reference_contexts", []))
        if not isinstance(contexts, list):
            contexts = list(contexts) if contexts is not None else []

        rows.append(
            {
                "question": str(row.get("user_input", row.get("question", ""))).strip(),
                "ground_truth": str(
                    row.get("reference", row.get("ground_truth", ""))
                ).strip(),
                "contexts": " ||| ".join(str(c) for c in contexts),
                "source": str(row.get("source", "")),
            }
        )

    return rows


def save_csv(rows: list[dict], path: Path = OUTPUT_CSV) -> None:
    """Save testset rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "ground_truth", "contexts", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Testset saved → %s (%d rows)", path, len(rows))


def main() -> None:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set. Exiting.")
        sys.exit(1)

    rows = generate_testset(test_size=TEST_SIZE)
    save_csv(rows)

    logger.info("Done. Next step: python phase-a/run_ragas.py")


if __name__ == "__main__":
    main()
