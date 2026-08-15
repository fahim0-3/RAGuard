"""Retrieval tests (Phase B).

Fast tier covers the pure logic: RRF arithmetic, rank and score preservation,
deterministic ordering, and every deduplication rule. Those run with no
database and no models.

Integration tier exercises the live hybrid retriever against pgvector and the
ingested corpus.
"""

from __future__ import annotations

import pytest

from src.retrieval.deduplication import (
    deduplicate,
    jaccard_similarity,
)
from src.retrieval.rrf import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
    rrf_contribution,
)
from src.retrieval.types import RetrievedChunk


def make_chunk(
    chunk_id: int,
    content: str = "placeholder content for retrieval testing",
    source: str = "refund_policy.txt",
    chunk_index: int = 0,
    **kwargs,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        chunk_index=chunk_index,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def test_rrf_contribution_uses_one_based_ranks():
    assert rrf_contribution(1) == pytest.approx(1 / 61)
    assert rrf_contribution(2) == pytest.approx(1 / 62)
    assert rrf_contribution(1, k=0) == pytest.approx(1.0)


def test_rrf_contribution_rejects_zero_rank():
    with pytest.raises(ValueError, match="1-based"):
        rrf_contribution(0)


def test_rrf_sums_contributions_across_retrievers():
    shared = make_chunk(1, dense_score=0.9)
    dense = [shared]
    sparse = [make_chunk(2, sparse_score=4.0), make_chunk(1, sparse_score=3.0)]

    fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse})
    by_id = {c.chunk_id: c for c in fused}

    # Chunk 1 is rank 1 in dense and rank 2 in sparse.
    assert by_id[1].fusion_score == pytest.approx(1 / 61 + 1 / 62)
    # Chunk 2 appears only in sparse, at rank 1.
    assert by_id[2].fusion_score == pytest.approx(1 / 61)


def test_rrf_preserves_both_retriever_scores():
    fused = reciprocal_rank_fusion(
        {
            "dense": [make_chunk(1, dense_score=0.87)],
            "sparse": [make_chunk(1, sparse_score=5.25)],
        }
    )
    assert len(fused) == 1
    assert fused[0].dense_score == pytest.approx(0.87)
    assert fused[0].sparse_score == pytest.approx(5.25)
    assert fused[0].fusion_score is not None


def test_rrf_preserves_retriever_ranks():
    fused = reciprocal_rank_fusion(
        {
            "dense": [make_chunk(9), make_chunk(1)],
            "sparse": [make_chunk(1)],
        }
    )
    by_id = {c.chunk_id: c for c in fused}
    assert by_id[1].retriever_ranks == {"dense": 2, "sparse": 1}
    assert by_id[1].dense_rank == 2
    assert by_id[1].sparse_rank == 1
    assert by_id[9].retriever_ranks == {"dense": 1}
    assert by_id[9].sparse_rank is None


def test_rrf_orders_by_score_then_chunk_id():
    """Ties must resolve deterministically, or metrics drift between runs."""
    fused = reciprocal_rank_fusion({"dense": [make_chunk(7)], "sparse": [make_chunk(3)]})
    assert [c.fusion_score for c in fused] == pytest.approx([1 / 61, 1 / 61])
    assert [c.chunk_id for c in fused] == [3, 7]


def test_rrf_respects_top_k():
    ranking = [make_chunk(i) for i in range(1, 11)]
    assert len(reciprocal_rank_fusion({"dense": ranking}, top_k=4)) == 4


def test_rrf_accepts_unnamed_rankings():
    fused = reciprocal_rank_fusion([[make_chunk(1)], [make_chunk(1)]])
    assert fused[0].retriever_ranks == {"retriever_0": 1, "retriever_1": 1}


def test_rrf_default_k_is_sixty():
    assert DEFAULT_RRF_K == 60


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_jaccard_identical_and_disjoint():
    assert jaccard_similarity("refund window thirty days", "refund window thirty days") == 1.0
    assert jaccard_similarity("refund window", "espresso descaling") == 0.0


def test_duplicate_chunk_ids_removed():
    result = deduplicate([make_chunk(1), make_chunk(1)])
    assert [c.chunk_id for c in result.kept] == [1]
    assert result.dropped[0].reason == "duplicate_chunk_id"


def test_near_identical_content_removed_keeping_highest_ranked():
    text = "concealed damage discovered after unboxing must be reported within seven days"
    first = make_chunk(1, content=text, source="damaged_product_policy.txt", chunk_index=0)
    second = make_chunk(2, content=text, source="return_policy.txt", chunk_index=5)

    result = deduplicate([first, second])
    assert [c.chunk_id for c in result.kept] == [1]
    assert result.dropped[0].reason == "near_duplicate_content"
    assert result.dropped[0].similar_to == 1


def test_genuinely_different_evidence_is_kept():
    result = deduplicate(
        [
            make_chunk(1, content="credit card refunds take five to seven business days"),
            make_chunk(2, content="espresso machine error E04 requires a descaling cycle"),
        ]
    )
    assert [c.chunk_id for c in result.kept] == [1, 2]
    assert not result.dropped


def test_adjacent_chunks_use_the_lower_threshold():
    """Consecutive chunks share an ingestion overlap window by construction."""
    base = "refund window thirty days payment card scanned warehouse policy"
    overlapping = "refund window thirty days payment card scanned warehouse exception"
    similarity = jaccard_similarity(base, overlapping)
    assert 0.70 <= similarity < 0.90, f"fixture drifted: similarity={similarity}"

    adjacent = deduplicate(
        [
            make_chunk(1, content=base, chunk_index=3),
            make_chunk(2, content=overlapping, chunk_index=4),
        ]
    )
    assert [c.chunk_id for c in adjacent.kept] == [1]
    assert adjacent.dropped[0].reason == "adjacent_overlap"

    # The same pair, not adjacent, falls below the near-duplicate threshold.
    separated = deduplicate(
        [
            make_chunk(1, content=base, chunk_index=0),
            make_chunk(2, content=overlapping, chunk_index=7),
        ]
    )
    assert [c.chunk_id for c in separated.kept] == [1, 2]


def test_contiguous_run_cap_limits_adjacent_chunks():
    chunks = [
        make_chunk(1, content="alpha refund eligibility window", chunk_index=0),
        make_chunk(2, content="beta processing times for cards", chunk_index=1),
        make_chunk(3, content="gamma partial refunds and shipping", chunk_index=2),
        make_chunk(4, content="delta escalation above five hundred", chunk_index=3),
    ]
    capped = deduplicate(chunks, max_adjacent_run=3)
    assert [c.chunk_id for c in capped.kept] == [1, 2, 3]
    assert capped.dropped[0].reason == "max_adjacent_run_exceeded"

    uncapped = deduplicate(chunks, max_adjacent_run=None)
    assert len(uncapped.kept) == 4


def test_run_cap_ignores_non_contiguous_indices():
    chunks = [
        make_chunk(1, content="alpha refund eligibility window", chunk_index=0),
        make_chunk(2, content="beta processing times for cards", chunk_index=2),
        make_chunk(3, content="gamma partial refunds and shipping", chunk_index=4),
        make_chunk(4, content="delta escalation above five hundred", chunk_index=6),
    ]
    result = deduplicate(chunks, max_adjacent_run=3)
    assert len(result.kept) == 4


def test_run_cap_is_per_source():
    chunks = [
        make_chunk(1, content="alpha refund eligibility", source="a.txt", chunk_index=0),
        make_chunk(2, content="beta refund processing", source="a.txt", chunk_index=1),
        make_chunk(3, content="gamma refund exclusions", source="a.txt", chunk_index=2),
        make_chunk(4, content="delta delivery lead times", source="b.txt", chunk_index=0),
        make_chunk(5, content="epsilon delivery tracking", source="b.txt", chunk_index=1),
    ]
    result = deduplicate(chunks, max_adjacent_run=3)
    assert len(result.kept) == 5


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def test_metric_primitives():
    from src.evaluation.retrieval_eval import (
        first_relevant_rank,
        hit_rate_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
    )

    ranked = ["delivery_policy.txt", "refund_policy.txt", "return_policy.txt"]
    expected = {"refund_policy.txt", "return_policy.txt"}

    assert hit_rate_at_k(ranked, expected, 1) == 0.0
    assert hit_rate_at_k(ranked, expected, 3) == 1.0
    assert recall_at_k(ranked, expected, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, expected, 5) == pytest.approx(1.0)
    assert first_relevant_rank(ranked, expected) == 2
    assert reciprocal_rank_at_k(ranked, expected, 5) == pytest.approx(0.5)
    # Outside the cut-off, the reciprocal rank is zero, not 1/rank.
    assert reciprocal_rank_at_k(ranked, expected, 1) == 0.0
    assert hit_rate_at_k(ranked, set(), 5) == 0.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.heavy
def test_hybrid_retriever_returns_deduplicated_top_k():
    from src.retrieval.hybrid import get_hybrid_retriever

    retriever = get_hybrid_retriever()
    diagnostics = retriever.retrieve_with_diagnostics("What is the refund window for items?")

    assert diagnostics.dense_hits, "dense retrieval returned nothing"
    assert diagnostics.sparse_hits, "BM25 returned nothing"
    assert diagnostics.results, "fusion returned nothing"
    assert len(diagnostics.results) <= retriever.final_top_k

    ids = [c.chunk_id for c in diagnostics.results]
    assert len(ids) == len(set(ids)), "duplicate chunk IDs survived deduplication"

    top = diagnostics.results[0]
    assert top.fusion_score is not None
    assert top.policy_id, "policy ID missing; doc_id was not loaded from the database"


@pytest.mark.integration
@pytest.mark.heavy
def test_bm25_recovers_exact_policy_identifier():
    """The case dense retrieval alone is expected to miss."""
    from src.retrieval.bm25 import get_bm25_index

    hits = get_bm25_index().search("What does policy REF-001 say?", 20)
    assert hits, "BM25 found nothing for an exact identifier query"
    assert any(h.source == "refund_policy.txt" for h in hits)


@pytest.mark.integration
@pytest.mark.heavy
@pytest.mark.parametrize(
    "query",
    [
        "What is the refund window for items?",
        "How long do I have to return something?",
        "What does policy REF-001 say?",
        "My payment was deducted but my order was not created.",
        "Can I get a replacement for a damaged product?",
    ],
)
def test_probe_queries_return_results(query):
    from src.retrieval.hybrid import get_hybrid_retriever

    results = get_hybrid_retriever().retrieve(query)
    assert results, f"no results for probe query: {query}"


@pytest.mark.integration
@pytest.mark.heavy
def test_baseline_evaluation_produces_metrics():
    from src.evaluation.retrieval_eval import evaluate_retrieval_baseline

    payload = evaluate_retrieval_baseline()

    assert payload["dataset"]["scored_cases"] > 0
    for name in ("hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
                 "recall_at_5", "recall_at_10", "mrr_at_5"):
        assert name in payload["measured_metrics"], f"missing metric {name}"
        assert 0.0 <= payload["measured_metrics"][name] <= 1.0
    assert payload["configuration"]["rrf"]["k"] == 60
    assert payload["configuration"]["retrieval"]["reranking_applied"] is False
