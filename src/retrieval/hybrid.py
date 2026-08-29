"""Hybrid retrieval: BM25 + pgvector, fused with RRF, then deduplicated.

Pipeline for a single query:

    BM25 top-20  ─┐
                  ├─ RRF (k=60) ─ deduplicate ─ final top-20
    dense top-20 ─┘

Fusion runs over the *full* union rather than a pre-truncated list, so that
deduplication removes candidates from the tail rather than leaving holes in the
returned top-20.

Reranking is deliberately not part of this module. Retrieval quality must be
measurable on its own, without loading a 568 M parameter cross-encoder, so that
the retrieval baseline is reproducible on modest hardware and in CI.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.deduplication import DeduplicationResult, deduplicate, deduplication_config
from src.retrieval.embeddings import embed_query
from src.retrieval.rrf import reciprocal_rank_fusion, rrf_config
from src.retrieval.types import RetrievedChunk
from src.retrieval.vector_store import dense_search
from src.timing import elapsed_ms

logger = logging.getLogger(__name__)

__all__ = [
    "HybridRetriever",
    "RetrievalDiagnostics",
    "get_hybrid_retriever",
    "reciprocal_rank_fusion",
]


@dataclass(slots=True)
class RetrievalDiagnostics:
    """Per-stage record of one retrieval, for reports and failure analysis."""

    query: str
    dense_hits: list[RetrievedChunk] = field(default_factory=list)
    sparse_hits: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    deduplication: DeduplicationResult | None = None
    results: list[RetrievedChunk] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "dense_hit_count": len(self.dense_hits),
            "sparse_hit_count": len(self.sparse_hits),
            "fused_count": len(self.fused),
            "dropped_by_deduplication": (
                [d.to_dict() for d in self.deduplication.dropped] if self.deduplication else []
            ),
            "result_count": len(self.results),
            "timings_ms": dict(self.timings_ms),
        }


class HybridRetriever:
    """BM25 + dense retrieval, RRF fusion, deduplication."""

    def __init__(
        self,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        final_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.dense_top_k = dense_top_k or settings.dense_top_k
        self.sparse_top_k = sparse_top_k or settings.sparse_top_k
        self.final_top_k = final_top_k or settings.fusion_top_k
        self.rrf_k = rrf_k or settings.rrf_k
        self.dedup_enabled = settings.dedup_enabled
        self.near_duplicate_threshold = settings.dedup_near_duplicate_threshold
        self.adjacent_threshold = settings.dedup_adjacent_threshold
        # A configured 0 means "no cap"; deduplicate() expects None for that.
        self.max_adjacent_run = settings.dedup_max_adjacent_run or None

    # -- stages ------------------------------------------------------------

    def _dedupe(self, chunks: list[RetrievedChunk]) -> DeduplicationResult:
        if not self.dedup_enabled:
            return DeduplicationResult(kept=list(chunks), dropped=[])
        return deduplicate(
            chunks,
            near_duplicate_threshold=self.near_duplicate_threshold,
            adjacent_threshold=self.adjacent_threshold,
            max_adjacent_run=self.max_adjacent_run,
        )

    # -- public API --------------------------------------------------------

    def retrieve_with_diagnostics(
        self, query: str, top_k: int | None = None
    ) -> RetrievalDiagnostics:
        """Retrieve and return every intermediate stage."""
        started = time.perf_counter()
        embedding = embed_query(query)
        query_embedding_ms = elapsed_ms(started)

        started = time.perf_counter()
        dense_hits = dense_search(embedding, self.dense_top_k)
        vector_search_ms = elapsed_ms(started)

        started = time.perf_counter()
        sparse_hits = get_bm25_index().search(query, self.sparse_top_k)
        bm25_search_ms = elapsed_ms(started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion({"dense": dense_hits, "sparse": sparse_hits}, k=self.rrf_k)
        rrf_fusion_ms = elapsed_ms(started)

        started = time.perf_counter()
        dedup = self._dedupe(fused)
        results = dedup.kept[: top_k or self.final_top_k]
        deduplication_ms = elapsed_ms(started)

        logger.debug(
            "query=%r dense=%d sparse=%d fused=%d dropped=%d final=%d",
            query,
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            dedup.dropped_count,
            len(results),
        )
        return RetrievalDiagnostics(
            query=query,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused=fused,
            deduplication=dedup,
            results=results,
            timings_ms={
                "query_embedding": query_embedding_ms,
                "vector_search": vector_search_ms,
                "bm25_search": bm25_search_ms,
                "rrf_fusion": rrf_fusion_ms,
                "deduplication": deduplication_ms,
            },
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retrieve_with_diagnostics(query, top_k=top_k).results

    def retrieve_many(self, queries: list[str], top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve for several query variants and fuse the results.

        Used by the self-healing loop after query rewriting: each variant is an
        independent retriever, so RRF composes over them without change.
        """
        rankings = {f"query_{i}": self.retrieve(q) for i, q in enumerate(queries)}
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        return self._dedupe(fused).kept[: top_k or self.final_top_k]

    def config(self) -> dict[str, Any]:
        """Full configuration block recorded in evaluation reports."""
        settings = get_settings()
        return {
            "retrieval": {
                "strategy": "hybrid_bm25_dense_rrf",
                "dense_top_k": self.dense_top_k,
                "sparse_top_k": self.sparse_top_k,
                "final_top_k": self.final_top_k,
                "reranking_applied": False,
            },
            "bm25": get_bm25_index().config(),
            "vector": {
                "embedding_model": settings.embedding_model,
                "dimension": settings.vector_dimension,
                "metric": "cosine",
                "operator": "<=>",
                "index": "hnsw (vector_cosine_ops)",
                "normalised_embeddings": True,
                "top_k": self.dense_top_k,
            },
            "rrf": rrf_config(self.rrf_k),
            "deduplication": {
                "enabled": self.dedup_enabled,
                **deduplication_config(
                    near_duplicate_threshold=self.near_duplicate_threshold,
                    adjacent_threshold=self.adjacent_threshold,
                    max_adjacent_run=self.max_adjacent_run,
                ),
            },
        }


_retriever: HybridRetriever | None = None
_lock = threading.Lock()


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever


def reset_hybrid_retriever() -> None:
    """Drop the cached retriever so new settings take effect."""
    global _retriever
    with _lock:
        _retriever = None
