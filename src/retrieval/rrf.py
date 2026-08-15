"""Reciprocal Rank Fusion.

    RRF_score(d) = sum over retrievers of  1 / (k + rank(d))     rank starts at 1

RRF is used rather than a weighted score blend because dense cosine similarity
and BM25 scores live on incomparable scales. RRF consumes only ranks, so there
is no per-corpus normalisation constant to tune, and evaluation runs stay
comparable across ingestion changes.

Two properties matter for the evaluation gate:

- **Score preservation.** Fusion never discards the contributing scores. A
  fused chunk keeps its dense score, its BM25 score, and the rank it held in
  each retriever, so a bad result can be attributed to the retriever that
  produced it.
- **Deterministic ordering.** Ties are broken on `chunk_id`, not on dictionary
  insertion order. Without this, two runs over identical data could emit
  different rankings and the metrics would drift for no reason.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def rrf_contribution(rank: int, k: int = DEFAULT_RRF_K) -> float:
    """Contribution of a single retriever to one document's fused score."""
    if rank < 1:
        raise ValueError(f"RRF ranks are 1-based; received rank={rank}")
    return 1.0 / (k + rank)


def _as_named(
    rankings: Mapping[str, Sequence[RetrievedChunk]] | Sequence[Sequence[RetrievedChunk]],
) -> dict[str, Sequence[RetrievedChunk]]:
    if isinstance(rankings, Mapping):
        return dict(rankings)
    return {f"retriever_{i}": ranking for i, ranking in enumerate(rankings)}


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievedChunk]] | Sequence[Sequence[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one.

    `rankings` may be a mapping of retriever name to ranking (preferred, since
    the names are recorded in `retriever_ranks`), or a bare sequence of
    rankings, which are auto-named.
    """
    named = _as_named(rankings)

    fused: dict[int, RetrievedChunk] = {}
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for name, ranking in named.items():
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contribution(rank, k)
            ranks.setdefault(chunk_id, {})[name] = rank

            existing = fused.get(chunk_id)
            if existing is None:
                fused[chunk_id] = replace(chunk)
            else:
                # Carry across whichever score each retriever contributed, so a
                # chunk found by both retrievers ends up holding both scores.
                if chunk.dense_score is not None:
                    existing.dense_score = chunk.dense_score
                if chunk.sparse_score is not None:
                    existing.sparse_score = chunk.sparse_score

    for chunk_id, chunk in fused.items():
        chunk.fusion_score = scores[chunk_id]
        chunk.retriever_ranks = dict(ranks[chunk_id])

    ordered = sorted(
        fused.values(),
        key=lambda c: (-(c.fusion_score or 0.0), c.chunk_id),
    )
    return ordered[:top_k] if top_k else ordered


def rrf_config(k: int = DEFAULT_RRF_K) -> dict[str, object]:
    """Configuration block recorded in evaluation reports."""
    return {
        "algorithm": "reciprocal_rank_fusion",
        "formula": "sum(1 / (k + rank))",
        "k": k,
        "rank_origin": 1,
        "tie_break": "ascending chunk_id",
        "scores_preserved": ["dense_score", "sparse_score", "fusion_score"],
        "ranks_preserved": True,
    }
