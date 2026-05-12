"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import importlib
import math
import os, sys
from dataclasses import dataclass
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        underthesea = importlib.import_module("underthesea")
        return underthesea.word_tokenize(text, format="text")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"].lower()).split()
            for chunk in chunks
        ]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = None

    def _fallback_scores(self, tokenized_query: list[str]) -> list[float]:
        doc_count = len(self.corpus_tokens)
        doc_freq = Counter(
            token
            for tokens in self.corpus_tokens
            for token in set(tokens)
        )
        avg_doc_len = sum(len(tokens) for tokens in self.corpus_tokens) / max(doc_count, 1)
        k1 = 1.5
        b = 0.75
        scores = []

        for tokens in self.corpus_tokens:
            token_counts = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for token in tokenized_query:
                freq = token_counts[token]
                if freq == 0:
                    continue
                idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
                score += idf * freq * (k1 + 1) / denom
            scores.append(score)

        return scores

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []

        tokenized_query = segment_vietnamese(query.lower()).split()
        if not tokenized_query:
            return []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenized_query)
        else:
            scores = self._fallback_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        # Prefer remote Qdrant, fallback to in-memory for local runs.
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            self.client = client
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        qdrant_models = importlib.import_module("qdrant_client.models")

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, batch_size=8)
        points = [
            qdrant_models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        if points:
            self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()

        try:
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
            )
        except AttributeError:
            hits = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            ).points

        return [
            SearchResult(
                text=hit.payload["text"],
                score=float(hit.score),
                metadata=hit.payload,
                method="dense",
            )
            for hit in hits
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}

    for results in results_list:
        for rank, result in enumerate(results, start=1):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank)

    ranked = sorted(
        rrf_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]

    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
