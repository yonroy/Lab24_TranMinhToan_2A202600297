"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4(+M5)."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.m1_chunking import load_documents, chunk_hierarchical
from rag.m2_search import HybridSearch
from rag.m3_rerank import CrossEncoderReranker
from rag.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from rag.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


_OPENAI_CLIENT = None


def _get_openai():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None and OPENAI_API_KEY:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _OPENAI_CLIENT


def build_pipeline(use_enrichment: bool = False):
    """Build production RAG pipeline. Set use_enrichment=True to run M5 (slow, costs API)."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Chunking (M1)
    print("\n[1/4] Chunking documents (hierarchical)...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    print(f"  {len(all_chunks)} child chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    if use_enrichment:
        print("\n[2/4] Enriching chunks (M5)...")
        enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
        if enriched:
            all_chunks = [
                {"text": e.enriched_text, "metadata": e.auto_metadata}
                for e in enriched
            ]
            print(f"  Enriched {len(enriched)} chunks")
    else:
        print("\n[2/4] Skipping enrichment (set use_enrichment=True to enable)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoderReranker()

    return search, reranker


def _generate_answer(query: str, contexts: list[str]) -> str:
    """LLM answer generation grounded in retrieved contexts."""
    client = _get_openai()
    if client is None or not contexts:
        return contexts[0] if contexts else "Không tìm thấy thông tin."

    context_str = "\n\n".join(contexts)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": ("Trả lời CHỈ dựa trên context cho sẵn. "
                             "Nếu context không có thông tin → trả lời 'Không tìm thấy.'")},
                {"role": "user",
                 "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM fallback to top context: {e}]")
        return contexts[0]


def run_query(query: str, search: HybridSearch,
              reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:RERANK_TOP_K]]
    answer = _generate_answer(query, contexts)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    if not test_set:
        print("  Test set is empty — nothing to evaluate.")
        return {}

    questions, answers, all_contexts, ground_truths = [], [], [], []
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'OK' if s >= 0.75 else 'XX'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, path="reports/ragas_report.json")
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline(use_enrichment=False)
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
