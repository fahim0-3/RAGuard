"""Phase C: cross-encoder reranking.

Every test in the fast tier runs against an injected fake model. Nothing here
downloads weights, so the contract — ordering, score preservation, top-20 to
top-5 truncation, and the degradation path — stays verifiable in CI on a
machine with no GPU and no model cache.

The single heavy test loads the real cross-encoder and is marked accordingly.
"""

from __future__ import annotations

import pytest

from src.reranking.cross_encoder import CrossEncoderReranker, sigmoid
from src.retrieval.types import RetrievedChunk


def make_chunk(
    chunk_id: int,
    content: str = "text",
    *,
    source: str = "policy.txt",
    chunk_index: int = 0,
    fusion_score: float | None = None,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    retriever_ranks: dict[str, int] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        chunk_index=chunk_index,
        doc_id=f"DOC-{chunk_id:03d}",
        fusion_score=fusion_score,
        dense_score=dense_score,
        sparse_score=sparse_score,
        retriever_ranks=retriever_ranks or {},
    )


class FakeModel:
    """Deterministic stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        if self.scores is not None:
            return self.scores[: len(pairs)]
        # Default: score by position of "relevant" in the content.
        return [float(len(content)) for _query, content in pairs]


def make_reranker(model: object, **kwargs) -> CrossEncoderReranker:
    return CrossEncoderReranker(model=model, **kwargs)


def test_explicit_reranker_device_overrides_the_shared_model_device():
    reranker = make_reranker(FakeModel(), device="cuda:0")

    assert reranker.device == "cuda:0"


def test_explicit_batch_size_overrides_the_device_default():
    model = FakeModel([1.0])
    reranker = make_reranker(model, batch_size=7)

    reranker.rerank_with_diagnostics("q", [make_chunk(1)], top_k=1)

    assert reranker.batch_size == 7
    assert model.calls == [[("q", "text")]]


def test_automatic_cuda_load_falls_back_to_cpu_without_switching_models(monkeypatch):
    """A driver/OOM failure must preserve the primary model before RRF fallback."""
    from src.reranking import cross_encoder

    class SettingsStub:
        reranker_model = "primary"
        reranker_fallback_model = "fallback"
        reranker_device = "auto"
        resolved_reranker_device = "cuda"
        resolved_reranker_model = "primary"
        reranker_enabled = True
        reranker_batch_size = 0
        reranker_max_length = 512
        reranker_max_concurrency = 1
        reranker_cpu_threads = 0
        reranker_warmup_inference = True
        rerank_top_k = 5
        rerank_candidate_top_k = 20

    attempted: list[tuple[str, str]] = []

    class CudaThenCpu(CrossEncoderReranker):
        def _load(self, model_name):
            attempted.append((model_name, self.device))
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return FakeModel([2.0])

    monkeypatch.setattr(cross_encoder, "get_settings", lambda: SettingsStub())
    reranker = CudaThenCpu()
    result = reranker.rerank_with_diagnostics("q", [make_chunk(1)], top_k=1)

    assert attempted == [("primary", "cuda"), ("primary", "cpu")]
    assert reranker.device == "cpu"
    assert reranker.batch_size == 32
    assert result.reranker_used is True
    assert result.model_name == "primary"
    assert result.fallback_model_used is False


def test_warmup_runs_a_synthetic_pair_before_the_first_query():
    model = FakeModel([1.0])
    reranker = make_reranker(model)

    assert reranker.warmup() is True
    assert model.calls == [[("warmup", "warmup")]]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_sigmoid_squashes_logits_into_unit_range():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(-40.0) == 0.0
    assert sigmoid(40.0) == 1.0
    assert 0.0 < sigmoid(-2.0) < 0.5 < sigmoid(2.0) < 1.0


def test_reranker_scores_every_candidate():
    chunks = [make_chunk(i) for i in range(1, 6)]
    reranker = make_reranker(FakeModel([5.0, 4.0, 3.0, 2.0, 1.0]))

    result = reranker.rerank_with_diagnostics("q", chunks, top_k=5)

    assert result.reranker_used is True
    assert result.candidate_count == 5
    assert all(c.rerank_score is not None for c in result.chunks)
    assert all(c.normalised_rerank_score is not None for c in result.chunks)


def test_normalised_score_is_sigmoid_of_raw_score():
    reranker = make_reranker(FakeModel([2.0]))

    chunk = reranker.rerank_with_diagnostics("q", [make_chunk(1)], top_k=1).chunks[0]

    assert chunk.rerank_score == pytest.approx(2.0)
    assert chunk.normalised_rerank_score == pytest.approx(sigmoid(2.0))


def test_fixed_order_scoring_uses_one_batch_without_reordering():
    model = FakeModel([-4.0, 8.0, 1.0])
    reranker = make_reranker(model)
    chunks = [make_chunk(3), make_chunk(1), make_chunk(2)]

    result = reranker.score_fixed_order_with_diagnostics("q", chunks)

    assert result.reranker_used is True
    assert [chunk.chunk_id for chunk in result.chunks] == [3, 1, 2]
    assert [chunk.rerank_score for chunk in result.chunks] == [-4.0, 8.0, 1.0]
    assert model.calls == [[("q", "text"), ("q", "text"), ("q", "text")]]
    assert result.bge_scoring_latency_ms >= 0
    assert result.bge_scoring_cpu_time_ms >= 0


def test_model_receives_query_and_chunk_content_pairs():
    chunks = [make_chunk(1, "first passage"), make_chunk(2, "second passage")]
    model = FakeModel([1.0, 2.0])

    make_reranker(model).rerank_with_diagnostics("why was I charged?", chunks)

    assert model.calls[0] == [
        ("why was I charged?", "first passage"),
        ("why was I charged?", "second passage"),
    ]


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def test_candidates_are_sorted_by_descending_rerank_score():
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    reranker = make_reranker(FakeModel([-1.0, 5.0, 2.0]))

    result = reranker.rerank_with_diagnostics("q", chunks, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [2, 3, 1]
    assert [c.rerank_score for c in result.chunks] == [5.0, 2.0, -1.0]


def test_reranking_promotes_a_low_ranked_candidate():
    """The Phase B failure mode: the right document sits below the wrong one."""
    chunks = [make_chunk(1, source="manual.txt"), make_chunk(2, source="damaged.txt")]
    reranker = make_reranker(FakeModel([0.1, 9.0]))

    result = reranker.rerank_with_diagnostics("arrived cracked", chunks, top_k=2)

    assert result.chunks[0].source == "damaged.txt"


def test_ties_break_on_chunk_id_for_determinism():
    chunks = [make_chunk(7), make_chunk(3), make_chunk(5)]
    reranker = make_reranker(FakeModel([1.0, 1.0, 1.0]))

    result = reranker.rerank_with_diagnostics("q", chunks, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [3, 5, 7]


# --------------------------------------------------------------------------
# top-20 -> top-5
# --------------------------------------------------------------------------


def test_twenty_candidates_are_scored_and_five_returned():
    chunks = [make_chunk(i) for i in range(1, 21)]
    model = FakeModel([float(i) for i in range(20)])

    result = make_reranker(model).rerank_with_diagnostics("q", chunks, top_k=5)

    assert len(model.calls[0]) == 20, "every candidate must be scored"
    assert result.candidate_count == 20
    assert len(result.chunks) == 5
    assert [c.chunk_id for c in result.chunks] == [20, 19, 18, 17, 16]


def test_candidate_pool_is_capped_before_scoring():
    chunks = [make_chunk(i) for i in range(1, 31)]
    model = FakeModel([float(i) for i in range(30)])

    result = make_reranker(model).rerank_with_diagnostics("q", chunks, top_k=5, candidate_top_k=20)

    assert len(model.calls[0]) == 20
    assert result.candidate_count == 20


def test_fewer_candidates_than_top_k_returns_what_exists():
    reranker = make_reranker(FakeModel([1.0, 2.0]))

    result = reranker.rerank_with_diagnostics("q", [make_chunk(1), make_chunk(2)], top_k=5)

    assert len(result.chunks) == 2


def test_empty_candidate_list_is_not_a_failure():
    result = make_reranker(FakeModel()).rerank_with_diagnostics("q", [], top_k=5)

    assert result.chunks == []
    assert result.reranker_used is False
    assert result.failure is None


# --------------------------------------------------------------------------
# Score preservation
# --------------------------------------------------------------------------


def test_retrieval_scores_survive_reranking():
    chunk = make_chunk(
        1,
        fusion_score=0.0328,
        dense_score=0.81,
        sparse_score=4.2,
        retriever_ranks={"dense": 3, "sparse": 1},
    )
    reranker = make_reranker(FakeModel([1.5]))

    reranked = reranker.rerank_with_diagnostics("q", [chunk], top_k=1).chunks[0]

    assert reranked.fusion_score == pytest.approx(0.0328)
    assert reranked.dense_score == pytest.approx(0.81)
    assert reranked.sparse_score == pytest.approx(4.2)
    assert reranked.retriever_ranks == {"dense": 3, "sparse": 1}
    assert reranked.dense_rank == 3
    assert reranked.sparse_rank == 1
    assert reranked.rerank_score == pytest.approx(1.5)


def test_identity_fields_survive_reranking():
    chunk = make_chunk(9, "body", source="refund_policy.txt", chunk_index=4)
    reranker = make_reranker(FakeModel([0.5]))

    reranked = reranker.rerank_with_diagnostics("q", [chunk], top_k=1).chunks[0]

    assert reranked.chunk_id == 9
    assert reranked.source == "refund_policy.txt"
    assert reranked.chunk_index == 4
    assert reranked.doc_id == "DOC-009"
    assert reranked.content == "body"
    assert reranked.citation_label == "refund_policy.txt#4"


def test_reranking_does_not_mutate_the_input_chunks():
    chunks = [make_chunk(1, fusion_score=0.03)]

    make_reranker(FakeModel([2.0])).rerank_with_diagnostics("q", chunks, top_k=1)

    assert chunks[0].rerank_score is None, "input must not be mutated in place"
    assert chunks[0].fusion_score == pytest.approx(0.03)


def test_to_dict_exposes_every_stage_score():
    reranker = make_reranker(FakeModel([1.0]))
    chunk = make_chunk(1, fusion_score=0.03, dense_score=0.9, sparse_score=3.0)

    payload = reranker.rerank_with_diagnostics("q", [chunk], top_k=1).chunks[0].to_dict()

    for key in (
        "sparse_score",
        "dense_score",
        "fusion_score",
        "rerank_score",
        "normalised_rerank_score",
        "retriever_ranks",
    ):
        assert key in payload


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------


class ExplodingLoader(CrossEncoderReranker):
    """Reranker whose model loading always fails."""

    def _load(self, model_name: str):
        raise RuntimeError(f"cannot load {model_name}")


def test_model_load_failure_returns_rrf_order_without_crashing():
    chunks = [make_chunk(i, fusion_score=1.0 / i) for i in range(1, 8)]

    result = ExplodingLoader().rerank_with_diagnostics("q", chunks, top_k=5)

    assert result.reranker_used is False
    assert result.failure_stage == "load"
    assert "cannot load" in result.failure
    assert [c.chunk_id for c in result.chunks] == [1, 2, 3, 4, 5], "RRF order preserved"


def test_rrf_scores_are_preserved_when_the_model_fails_to_load():
    chunk = make_chunk(1, fusion_score=0.0328, dense_score=0.81, sparse_score=4.2)

    result = ExplodingLoader().rerank_with_diagnostics("q", [chunk], top_k=5)

    kept = result.chunks[0]
    assert kept.fusion_score == pytest.approx(0.0328)
    assert kept.dense_score == pytest.approx(0.81)
    assert kept.sparse_score == pytest.approx(4.2)
    assert kept.rerank_score is None, "no score may be invented when the model is absent"
    assert kept.normalised_rerank_score is None


def test_fallback_model_is_tried_when_the_primary_fails():
    attempted: list[str] = []

    class FallbackOnly(CrossEncoderReranker):
        def _load(self, model_name: str):
            attempted.append(model_name)
            if model_name == "primary":
                raise OSError("primary weights unavailable")
            return FakeModel([3.0])

    reranker = FallbackOnly(model_name="primary", fallback_model_name="fallback")
    result = reranker.rerank_with_diagnostics("q", [make_chunk(1)], top_k=1)

    assert attempted == ["primary", "fallback"]
    assert result.reranker_used is True
    assert result.fallback_model_used is True
    assert result.model_name == "fallback"


def test_load_is_attempted_once_and_the_failure_remembered():
    class CountingLoader(CrossEncoderReranker):
        loads = 0

        def _load(self, model_name: str):
            type(self).loads += 1
            raise RuntimeError("nope")

    reranker = CountingLoader(model_name="a", fallback_model_name="b")
    for _ in range(3):
        reranker.rerank_with_diagnostics("q", [make_chunk(1)])

    assert CountingLoader.loads == 2, "primary and fallback tried once, then remembered"


def test_inference_failure_returns_rrf_order():
    class BrokenModel:
        def predict(self, pairs, **kwargs):
            raise ValueError("tensor shape mismatch")

    chunks = [make_chunk(1), make_chunk(2)]
    result = make_reranker(BrokenModel()).rerank_with_diagnostics("q", chunks, top_k=5)

    assert result.reranker_used is False
    assert result.failure_stage == "inference"
    assert "tensor shape mismatch" in result.failure
    assert [c.chunk_id for c in result.chunks] == [1, 2]


def test_score_count_mismatch_is_treated_as_failure():
    """A model returning the wrong number of scores must not be zipped blindly."""
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    result = make_reranker(FakeModel([1.0])).rerank_with_diagnostics("q", chunks, top_k=3)

    assert result.reranker_used is False
    assert result.failure_stage == "inference"
    assert [c.chunk_id for c in result.chunks] == [1, 2, 3]


def test_disabled_reranker_returns_rrf_order():
    chunks = [make_chunk(i) for i in range(1, 10)]

    result = make_reranker(FakeModel(), enabled=False).rerank_with_diagnostics("q", chunks, top_k=5)

    assert result.reranker_used is False
    assert result.failure_stage == "disabled"
    assert [c.chunk_id for c in result.chunks] == [1, 2, 3, 4, 5]


def test_rerank_returns_a_plain_list_for_existing_callers():
    """self_healing.pipeline calls rerank() and expects a list of chunks."""
    reranker = make_reranker(FakeModel([1.0, 2.0]))

    out = reranker.rerank("q", [make_chunk(1), make_chunk(2)], top_k=2)

    assert isinstance(out, list)
    assert [c.chunk_id for c in out] == [2, 1]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_config_reports_the_model_actually_loaded():
    reranker = make_reranker(FakeModel(), model_name="configured-model")

    config = reranker.config()

    assert config["configured_model"] == "configured-model"
    assert config["input_pair"] == "(original query, candidate chunk content)"
    assert "fusion_score" in config["scores_preserved"]
    assert "rerank_score" in config["scores_preserved"]


def test_config_records_the_load_error_after_a_failure():
    reranker = ExplodingLoader(model_name="a", fallback_model_name="b")
    reranker.rerank_with_diagnostics("q", [make_chunk(1)])

    config = reranker.config()

    assert config["loaded_model"] is None
    assert "cannot load" in config["load_error"]


def test_result_to_dict_records_failure_state():
    payload = ExplodingLoader().rerank_with_diagnostics("q", [make_chunk(1)]).to_dict()

    assert payload["reranker_used"] is False
    assert payload["failure_stage"] == "load"
    assert payload["candidate_count"] == 1


def test_result_exposes_queue_and_inference_latency_without_query_content():
    result = make_reranker(FakeModel([1.0])).rerank_with_diagnostics("sensitive query", [make_chunk(1)])
    payload = result.to_dict()

    assert payload["queue_wait_ms"] >= 0
    assert payload["inference_latency_ms"] >= 0
    assert "sensitive query" not in str(
        {key: value for key, value in payload.items() if key != "query"}
    )


# --------------------------------------------------------------------------
# Comparison evaluator
# --------------------------------------------------------------------------


def _case(case_id: str = "GC-001", **overrides) -> dict:
    case = {
        "id": case_id,
        "question": "q",
        "expected_sources": ["refund_policy.txt"],
        "should_abstain": False,
    }
    case.update(overrides)
    return case


def test_rank_change_labels():
    from src.evaluation.reranking_eval import _rank_change

    assert _rank_change(4, 1) == "improved: 4 -> 1"
    assert _rank_change(1, 3) == "regressed: 1 -> 3"
    assert _rank_change(2, 2) == "unchanged: 2"
    assert _rank_change(None, 2) == "recovered: absent -> 2"
    assert _rank_change(1, None) == "lost: 1 -> absent"


def test_score_case_uses_phase_b_metric_definitions():
    from src.evaluation.reranking_eval import score_case

    chunks = [
        make_chunk(1, source="delivery_policy.txt"),
        make_chunk(2, source="refund_policy.txt"),
    ]

    result = score_case("hybrid_rrf", _case(), chunks)

    assert result.first_relevant_rank == 2
    assert result.metrics["hit_rate_at_1"] == 0.0
    assert result.metrics["hit_rate_at_3"] == 1.0
    assert result.metrics["mrr_at_5"] == pytest.approx(0.5)
    assert result.metrics["recall_at_5"] == pytest.approx(1.0)


def test_abstention_cases_are_not_scored():
    from src.evaluation.reranking_eval import score_case

    result = score_case(
        "hybrid_rrf", _case(should_abstain=True), [make_chunk(1, source="refund_policy.txt")]
    )

    assert result.metrics == {}


def _verdict(before_ranks: dict[str, int | None], after_ranks: dict[str, int | None]):
    """Build a verdict from per-case ranks alone."""
    from src.evaluation.reranking_eval import (
        CONFIG_C,
        CONFIG_D,
        ConfigCaseResult,
        aggregate,
        build_verdict,
    )

    cases_by_id = {cid: _case(cid) for cid in before_ranks}

    def results(ranks):
        out = []
        for case_id, rank in ranks.items():
            sources = ["other.txt"] * 10
            if rank is not None:
                sources[rank - 1] = "refund_policy.txt"
            metrics = {
                "hit_rate_at_1": 1.0 if rank == 1 else 0.0,
                "hit_rate_at_3": 1.0 if rank and rank <= 3 else 0.0,
                "hit_rate_at_5": 1.0 if rank and rank <= 5 else 0.0,
                "recall_at_5": 1.0 if rank and rank <= 5 else 0.0,
                "mrr_at_5": 1.0 / rank if rank and rank <= 5 else 0.0,
            }
            out.append(
                ConfigCaseResult(
                    config="x",
                    case_id=case_id,
                    retrieved_sources=sources,
                    retrieved_policy_ids=sources,
                    first_relevant_rank=rank,
                    metrics=metrics,
                )
            )
        return out

    results_map = {CONFIG_C: results(before_ranks), CONFIG_D: results(after_ranks)}
    metrics = {
        CONFIG_C: aggregate(results_map[CONFIG_C], cases_by_id),
        CONFIG_D: aggregate(results_map[CONFIG_D], cases_by_id),
    }
    return build_verdict(metrics, results_map, cases_by_id)


def test_verdict_reports_improvement_when_ranks_rise():
    verdict = _verdict({"GC-002": 2, "GC-010": 4}, {"GC-002": 1, "GC-010": 1})

    assert verdict["mrr_at_5_improved"] is True
    assert verdict["no_regressions"] is True
    assert verdict["reranking_is_an_improvement"] is True
    assert {e["case_id"] for e in verdict["improved_cases"]} == {"GC-002", "GC-010"}


def test_verdict_refuses_improvement_when_a_case_regresses():
    """A better mean must not hide a case that got worse."""
    verdict = _verdict({"GC-001": 1, "GC-002": 4}, {"GC-001": 3, "GC-002": 1})

    assert verdict["no_regressions"] is False
    assert verdict["regressed_cases"] == [{"case_id": "GC-001", "before_rank": 1, "after_rank": 3}]
    assert verdict["reranking_is_an_improvement"] is False


def test_verdict_refuses_improvement_when_a_case_falls_out_of_the_window():
    verdict = _verdict({"GC-001": 3}, {"GC-001": None})

    assert verdict["recall_at_5_maintained"] is False
    assert verdict["reranking_is_an_improvement"] is False


def test_verdict_is_false_when_mrr_is_merely_unchanged():
    """Running the model is not evidence that it helped."""
    verdict = _verdict({"GC-001": 1, "GC-002": 2}, {"GC-001": 1, "GC-002": 2})

    assert verdict["mrr_at_5_improved"] is False
    assert verdict["no_regressions"] is True
    assert verdict["reranking_is_an_improvement"] is False


def test_verdict_ignores_cases_that_were_never_in_the_reranked_window():
    """A source at rank 9 cannot appear in a top-5, so its loss is truncation."""
    verdict = _verdict({"GC-001": 1, "GC-002": 9}, {"GC-001": 1, "GC-002": None})

    assert verdict["regressed_cases"] == []


# --------------------------------------------------------------------------
# Real model
# --------------------------------------------------------------------------


@pytest.mark.heavy
def test_real_cross_encoder_prefers_the_relevant_passage():
    """Loads real weights. Excluded from the fast tier."""
    reranker = CrossEncoderReranker()
    chunks = [
        make_chunk(
            1,
            "Carriers make two delivery attempts before returning the parcel.",
            source="delivery_policy.txt",
        ),
        make_chunk(
            2,
            "Customers are never asked to pay return shipping for a validated "
            "damage claim reported within 48 hours.",
            source="damaged_product_policy.txt",
        ),
    ]

    result = reranker.rerank_with_diagnostics(
        "The espresso machine arrived cracked. Do I have to pay to send it back?",
        chunks,
        top_k=2,
    )

    assert result.reranker_used is True, f"model unavailable: {result.failure}"
    assert result.chunks[0].source == "damaged_product_policy.txt"
    assert result.chunks[0].rerank_score > result.chunks[1].rerank_score
