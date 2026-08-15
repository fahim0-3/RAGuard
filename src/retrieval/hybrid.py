"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

RRF is used rather than a weighted score blend because dense cosine similarity
and BM25 scores live on incomparable scales. RRF only consumes ranks, so it
needs no per-corpus normalisation constant to tune, which keeps evaluation
runs comparable across ingestion changes.

    score(d) = sum over retrievers of 1 / (k + rank(d))
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

from src.config import get_settings
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.embeddings import embed_query
from src.retrieval.types import RetrievedChunk
from src.retrieval.vector_store import dense_search

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one. Later duplicates merge their scores."""
    fused: dict[int, RetrievedChunk] = {}
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = fused.get(chunk.chunk_id)
            if existing is None:
                fused[chunk.chunk_id] = replace(chunk)
            else:
                # Keep whichever stage-score each retriever contributed.
                if chunk.dense_score is not None:
                    existing.dense_score = chunk.dense_score
                if chunk.sparse_score is not None:
                    existing.sparse_score = chunk.sparse_score

    for chunk_id, score in scores.items():
        fused[chunk_id].fusion_score = score

    ordered = sorted(fused.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)
    return ordered[:top_k] if top_k else ordered


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion. Reranking is a separate stage."""

    def __init__(
        self,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        fusion_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.dense_top_k = dense_top_k or settings.dense_top_k
        self.sparse_top_k = sparse_top_k or settings.sparse_top_k
        self.fusion_top_k = fusion_top_k or settings.fusion_top_k
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        dense_hits = dense_search(embed_query(query), self.dense_top_k)
        sparse_hits = get_bm25_index().search(query, self.sparse_top_k)
        logger.debug(
            "query=%r dense=%d sparse=%d", query, len(dense_hits), len(sparse_hits)
        )
        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=self.rrf_k,
            top_k=top_k or self.fusion_top_k,
        )

    def retrieve_many(
        self, queries: list[str], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve for several query variants and fuse the results.

        Used by the self-healing loop after query rewriting: each variant is an
        independent retriever, so RRF composes over them without change.
        """
        rankings = [self.retrieve(q, top_k=self.fusion_top_k) for q in queries]
        return reciprocal_rank_fusion(
            rankings, k=self.rrf_k, top_k=top_k or self.fusion_top_k
        )


_retriever: HybridRetriever | None = None
_lock = threading.Lock()


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever
