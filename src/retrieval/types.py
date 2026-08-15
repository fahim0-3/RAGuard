"""Shared retrieval data structures.

`RetrievedChunk` is the single currency passed between retrieval, reranking,
self-healing, generation, and evaluation. Keeping one type avoids the dict-shape
drift that makes RAG pipelines hard to test.

Every stage writes its own score field and leaves the others untouched, so a
finished result carries the whole audit trail: which retriever found it, at what
rank, with what score, and how fusion combined them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # Document-level identifier parsed at ingestion, for example REF-001.
    doc_id: str = ""

    # Stage scores. Each stage fills its own field so the trace stays auditable.
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    normalised_rerank_score: float | None = None

    # Rank this chunk held in each contributing retriever's own ranking,
    # 1-based. Retained for debugging fusion behaviour.
    retriever_ranks: dict[str, int] = field(default_factory=dict)

    # Set when the chunk was recovered by a rewritten query rather than the original.
    retrieved_by: str = "original"

    @property
    def citation_label(self) -> str:
        """Stable, human-readable handle used in prompts and citations."""
        return f"{self.source}#{self.chunk_index}"

    @property
    def policy_id(self) -> str:
        """Document identifier, falling back to metadata then to the filename."""
        return self.doc_id or self.metadata.get("doc_id") or self.source

    @property
    def dense_rank(self) -> int | None:
        return self.retriever_ranks.get("dense")

    @property
    def sparse_rank(self) -> int | None:
        return self.retriever_ranks.get("sparse")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "citation_label": self.citation_label,
            "policy_id": self.policy_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "normalised_rerank_score": self.normalised_rerank_score,
            "retriever_ranks": self.retriever_ranks,
            "retrieved_by": self.retrieved_by,
        }
