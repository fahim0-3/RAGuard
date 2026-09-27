"""Mock-only contracts for the explicit hosted reranker profile."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx

from src.config import Settings
from src.reranking.cross_encoder import RerankResult, sigmoid
from src.reranking.provider import (
    CohereReranker,
    ConfiguredReranker,
    VoyageReranker,
)
from src.retrieval.types import RetrievedChunk


def chunk(chunk_id: int, content: str = "policy text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source=f"policy-{chunk_id}.txt",
        chunk_index=chunk_id,
        doc_id=f"POL-{chunk_id}",
        fusion_score=0.03 * chunk_id,
    )


def settings(**overrides):
    values = {
        "reranker_enabled": True,
        "reranker_provider": "voyage",
        "reranker_remote_allowed": True,
        "voyage_api_key": "v" * 40,
        "voyage_rerank_model": "rerank-2.5-lite",
        "hosted_rerank_timeout_seconds": 3.0,
        "hosted_rerank_max_retries": 1,
        "hosted_rerank_top_k": 5,
        "hosted_rerank_max_candidates": 20,
        "reranker_fallback_provider": "local",
        "rerank_top_k": 5,
        "rerank_candidate_top_k": 20,
        "reranker_confidence_profile": "unverified",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class Response:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, json):
        self.calls.append((url, json))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Local:
    def __init__(self, fixed_order_scores: list[float] | None = None):
        self.calls = 0
        self.fixed_order_calls = 0
        self.fixed_order_scores = fixed_order_scores
        self.is_model_loaded = False
        self.loaded_model_name = None
        self.load_error = None

    def warmup(self):
        self.is_model_loaded = True
        return True

    def rerank_with_diagnostics(self, query, chunks, *, top_k, candidate_top_k):
        self.calls += 1
        selected = chunks[:candidate_top_k][:top_k]
        return RerankResult(
            query=query,
            chunks=selected,
            reranker_used=True,
            model_name="BAAI/bge-reranker-v2-m3",
            candidate_count=len(chunks[:candidate_top_k]),
            inference_latency_ms=7.0,
        )

    def score_fixed_order_with_diagnostics(self, query, chunks):
        self.fixed_order_calls += 1
        scores = self.fixed_order_scores or [float(index + 1) for index in range(len(chunks))]
        scored = [
            replace(item, rerank_score=score, normalised_rerank_score=sigmoid(score))
            for item, score in zip(chunks, scores, strict=True)
        ]
        return RerankResult(
            query=query,
            chunks=scored,
            reranker_used=True,
            model_name="BAAI/bge-reranker-v2-m3",
            candidate_count=len(chunks),
            inference_latency_ms=3.0,
            bge_scoring_latency_ms=3.0,
            bge_scoring_cpu_time_ms=2.0,
        )

    def config(self):
        return {"local": True}


def test_voyage_reorders_by_provider_score_without_writing_bge_score_fields():
    client = Client(
        [
            Response(
                200,
                {
                    "data": [
                        {"index": 0, "relevance_score": 0.2},
                        {"index": 1, "relevance_score": 0.9},
                    ]
                },
            )
        ]
    )
    reranker = VoyageReranker(
        api_key="secret-not-for-logs",
        model_name="rerank-2.5-lite",
        timeout_seconds=3,
        max_retries=0,
        client=client,
    )
    first, second = chunk(1), chunk(2)

    result = reranker.rerank_with_diagnostics(
        "sensitive query", [first, second], top_k=2, candidate_top_k=2
    )

    assert [item.chunk_id for item in result.chunks] == [2, 1]
    assert result.provider_raw_scores == {2: 0.9, 1: 0.2}
    assert result.provider_order == [2, 1]
    assert all(item.rerank_score is None for item in result.chunks)
    assert all(item.normalised_rerank_score is None for item in result.chunks)
    assert result.confidence_score_source == "unverified_hosted_order_only"
    assert result.chunks[0].citation_label == "policy-2.txt#2"
    assert client.calls[0][1]["documents"] == [first.content, second.content]


def test_voyage_order_uses_bge_scores_for_confidence_not_voyage_scores():
    client = Client([Response(200, {"data": [{"index": 0, "relevance_score": 0.99}]})])
    local = Local(fixed_order_scores=[2.0])
    reranker = ConfiguredReranker(
        settings=settings(reranker_confidence_profile="voyage_candidate_profile"),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.chunks[0].rerank_score == 2.0
    assert result.chunks[0].normalised_rerank_score == sigmoid(2.0)
    assert result.provider_raw_scores == {1: 0.99}
    assert result.confidence_score_source == "bge_sigmoid_fixed_voyage_order"
    assert local.fixed_order_calls == 1


def test_voyage_retries_a_rate_limit_once_without_logging_response_body():
    client = Client(
        [
            Response(429, {"error": {"message": "do not expose me"}}),
            Response(200, {"data": [{"index": 0, "relevance_score": 0.8}]}),
        ]
    )
    delays: list[float] = []
    reranker = VoyageReranker(
        api_key="key",
        model_name="rerank-2.5-lite",
        timeout_seconds=3,
        max_retries=1,
        client=client,
        sleep=delays.append,
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)], top_k=1, candidate_top_k=1)

    assert result.reranker_used is True
    assert result.retry_count == 1
    assert len(client.calls) == 2
    assert delays == [0.25]
    assert "do not expose me" not in str(result.to_dict())


def test_timeout_uses_deterministic_local_fallback():
    local = Local()

    class TimeoutVoyage:
        def rerank_with_diagnostics(self, query, chunks, *, top_k, candidate_top_k):
            return RerankResult(
                query=query,
                chunks=chunks[:top_k],
                candidate_count=len(chunks[:candidate_top_k]),
                failure="voyage_timeout",
                failure_stage="hosted",
                requested_provider="voyage",
                actual_provider="voyage",
                hosted_latency_ms=12.0,
                retry_count=1,
            )

    reranker = ConfiguredReranker(
        settings=settings(), local_factory=lambda: local, voyage_factory=lambda **_: TimeoutVoyage()
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert result.reranker_used is True
    assert result.requested_provider == "voyage"
    assert result.actual_provider == "local"
    assert result.fallback_used is True
    assert result.failure == "voyage_timeout"
    assert result.retry_count == 1
    assert result.hosted_latency_ms == 12.0
    assert local.calls == 1


def test_malformed_hosted_response_uses_local_fallback():
    local = Local()
    client = Client([Response(200, {"data": [{"index": 99, "relevance_score": 0.9}]})])
    reranker = ConfiguredReranker(
        settings=settings(),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.actual_provider == "local"
    assert result.fallback_used is True
    assert result.failure == "voyage_malformed_response"
    assert local.calls == 1


def test_api_key_alone_never_enables_remote_transmission():
    local = Local()
    voyage_calls = []
    reranker = ConfiguredReranker(
        settings=settings(reranker_remote_allowed=False),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: voyage_calls.append(kwargs),
    )

    result = reranker.rerank_with_diagnostics("private policy question", [chunk(1)])

    assert voyage_calls == []
    assert local.calls == 1
    assert result.actual_provider == "local"
    assert result.failure == "voyage_remote_not_explicitly_enabled"


def test_successful_voyage_request_does_not_lazy_load_local_model():
    local_creations = []
    client = Client([Response(200, {"data": [{"index": 0, "relevance_score": 0.9}]})])
    reranker = ConfiguredReranker(
        settings=settings(),
        local_factory=lambda: local_creations.append(Local()) or local_creations[-1],
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.actual_provider == "voyage"
    assert result.reranker_used is True
    # BGE is loaded only after Voyage has successfully selected the evidence.
    assert len(local_creations) == 1
    assert local_creations[0].fixed_order_calls == 1


def test_voyage_preserves_the_top_twenty_to_top_five_contract():
    candidates = [chunk(index) for index in range(1, 26)]
    client = Client(
        [
            Response(
                200,
                {
                    "data": [
                        {"index": index, "relevance_score": float(20 - index)} for index in range(5)
                    ]
                },
            )
        ]
    )
    reranker = ConfiguredReranker(
        settings=settings(),
        local_factory=Local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", candidates)

    assert client.calls[0][1]["top_k"] == 5
    assert len(client.calls[0][1]["documents"]) == 20
    assert result.candidate_count == 20
    assert len(result.chunks) == 5
    assert result.bge_scoring_latency_ms == 3.0


def test_bge_scoring_does_not_reorder_voyage_top_five():
    local = Local(fixed_order_scores=[-5.0, 9.0])
    client = Client(
        [
            Response(
                200,
                {
                    "data": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                },
            )
        ]
    )
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_top_k=2),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert [item.chunk_id for item in result.chunks] == [2, 1]
    assert [item.rerank_score for item in result.chunks] == [-5.0, 9.0]
    assert result.provider_order == [2, 1]
    assert local.fixed_order_calls == 1


def test_voyage_hybrid_result_satisfies_existing_confidence_contract():
    from src.self_healing.confidence import score_retrieval

    local = Local(fixed_order_scores=[2.0, 1.0])
    client = Client(
        [
            Response(
                200,
                {
                    "data": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                },
            )
        ]
    )
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_top_k=2),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])
    confidence = score_retrieval(result.chunks)

    assert confidence.level == "high"
    assert confidence.supporting_chunks == 2
    assert result.confidence_score_source == "bge_sigmoid_fixed_voyage_order"


def test_bge_scoring_failure_reverts_to_full_local_top_twenty_path():
    class ScoreFailureLocal(Local):
        def score_fixed_order_with_diagnostics(self, query, chunks):
            self.fixed_order_calls += 1
            return RerankResult(
                query=query,
                chunks=chunks,
                candidate_count=len(chunks),
                failure="bge_score_failed",
                failure_stage="inference",
            )

    local = ScoreFailureLocal()
    client = Client([Response(200, {"data": [{"index": 1, "relevance_score": 0.9}]})])
    candidates = [chunk(index) for index in range(1, 26)]
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_top_k=1),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    result = reranker.rerank_with_diagnostics("q", candidates)

    assert local.fixed_order_calls == 1
    assert local.calls == 1
    assert result.actual_provider == "local"
    assert result.fallback_used is True
    assert result.candidate_count == 20


def test_rrf_fallback_does_not_load_local_model_when_configured():
    local_creations = []
    reranker = ConfiguredReranker(
        settings=settings(reranker_remote_allowed=False, reranker_fallback_provider="rrf"),
        local_factory=lambda: local_creations.append(Local()) or local_creations[-1],
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert result.actual_provider == "rrf"
    assert result.fallback_used is True
    assert [item.chunk_id for item in result.chunks] == [1, 2]
    assert local_creations == []


def test_voyage_warmup_loads_the_local_confidence_scorer_without_a_network_call():
    local = Local()
    reranker = ConfiguredReranker(settings=settings(), local_factory=lambda: local)

    assert reranker.warmup() is True
    assert reranker.is_model_loaded is True
    assert reranker.loaded_model_name == "rerank-2.5-lite"
    assert local.is_model_loaded is True


def test_settings_default_to_local_and_key_does_not_change_provider():
    configured = Settings(_env_file=None, voyage_api_key="v" * 40)

    assert configured.reranker_provider == "local"
    assert configured.reranker_remote_allowed is False


def test_timeout_response_is_recorded_without_exception_text():
    client = Client([httpx.ReadTimeout("secret endpoint detail")])
    reranker = VoyageReranker(
        api_key="key",
        model_name="rerank-2.5-lite",
        timeout_seconds=3,
        max_retries=0,
        client=client,
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)], top_k=1, candidate_top_k=1)

    assert result.failure == "voyage_timeout"
    assert "secret endpoint detail" not in str(result.to_dict())


# --------------------------------------------------------------------------
# Overlapped local scoring: same scores, off the critical path
# --------------------------------------------------------------------------


class PairwiseLocal(Local):
    """Scores depend only on the (query, chunk) pair, as a cross-encoder's do.

    The positional scores of `Local` would make batch composition matter, which
    is exactly the property that makes the overlap safe for the real model.
    """

    def __init__(self, *, degraded: bool = False):
        super().__init__()
        self.degraded = degraded
        self.scored_ids: list[list[int]] = []

    def score_fixed_order_with_diagnostics(self, query, chunks):
        self.fixed_order_calls += 1
        self.scored_ids.append([item.chunk_id for item in chunks])
        if self.degraded:
            return RerankResult(query=query, chunks=list(chunks), reranker_used=False)
        scored = [
            replace(
                item,
                rerank_score=item.chunk_id * 0.5,
                normalised_rerank_score=sigmoid(item.chunk_id * 0.5),
            )
            for item in chunks
        ]
        return RerankResult(
            query=query,
            chunks=scored,
            reranker_used=True,
            model_name="m",
            candidate_count=len(chunks),
        )


def _voyage_picks(indices: list[int]) -> Client:
    return Client(
        [
            Response(
                200,
                {
                    "data": [
                        {"index": i, "relevance_score": 1.0 - n * 0.1}
                        for n, i in enumerate(indices)
                    ]
                },
            )
        ]
    )


def _run(overlap: bool, local: PairwiseLocal):
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_top_k=3, reranker_overlap_local_scoring=overlap),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=_voyage_picks([4, 0, 2]), **kwargs),
    )
    try:
        return reranker.rerank_with_diagnostics("q", [chunk(i) for i in range(1, 7)])
    finally:
        reranker.close()


def test_overlapped_scoring_matches_the_sequential_pass_exactly():
    sequential = _run(False, PairwiseLocal())
    overlapped = _run(True, PairwiseLocal())

    assert [c.chunk_id for c in overlapped.chunks] == [c.chunk_id for c in sequential.chunks]
    assert [c.rerank_score for c in overlapped.chunks] == [
        c.rerank_score for c in sequential.chunks
    ]
    assert [c.normalised_rerank_score for c in overlapped.chunks] == [
        c.normalised_rerank_score for c in sequential.chunks
    ]
    assert overlapped.confidence_score_source == sequential.confidence_score_source
    assert overlapped.provider_order == sequential.provider_order


def test_overlap_scores_the_whole_pool_once_instead_of_after_voyage():
    local = PairwiseLocal()

    result = _run(True, local)

    assert local.fixed_order_calls == 1
    assert local.scored_ids == [[1, 2, 3, 4, 5, 6]], "the pool, scored while Voyage ran"
    assert [c.chunk_id for c in result.chunks] == [5, 1, 3], "still Voyage's order"


def test_a_degraded_prescore_falls_back_to_the_sequential_pass():
    """A pre-score that cannot cover every pick must never reach the evidence gate."""
    local = PairwiseLocal(degraded=True)

    result = _run(True, local)

    assert local.fixed_order_calls >= 2, "the sequential pass must run after a failed pre-score"
    assert result.chunks


def test_overlap_disabled_scores_only_voyages_picks():
    local = PairwiseLocal()

    _run(False, local)

    assert local.scored_ids == [[5, 1, 3]]


# --------------------------------------------------------------------------
# Voyage cooldown: one hosted timeout, not one per request
# --------------------------------------------------------------------------


def _voyage_reranker(client, local, **overrides):
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_cooldown_enabled=True, **overrides),
        local_factory=lambda: local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )
    now = [1_000.0]
    reranker._clock = lambda: now[0]
    return reranker, now


def test_a_voyage_timeout_sends_the_next_request_straight_to_local():
    import httpx as _httpx

    client = Client([_httpx.TimeoutException("slow"), _httpx.TimeoutException("slow")])
    local = Local()
    reranker, _now = _voyage_reranker(client, local, hosted_rerank_max_retries=1)

    first = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])
    calls_after_first = len(client.calls)
    second = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert first.fallback_used is True
    assert len(client.calls) == calls_after_first, "no second hosted timeout"
    assert second.actual_provider == "local"
    assert second.failure == "voyage_cooling_down"


def test_voyage_is_tried_again_once_its_cooldown_expires():
    import httpx as _httpx

    client = Client(
        [
            _httpx.TimeoutException("slow"),
            _httpx.TimeoutException("slow"),
            Response(200, {"data": [{"index": 0, "relevance_score": 0.9}]}),
        ]
    )
    reranker, now = _voyage_reranker(client, Local(), hosted_rerank_top_k=1)
    reranker.rerank_with_diagnostics("q", [chunk(1)])

    now[0] += 31.0
    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.actual_provider == "voyage"
    assert result.reranker_used is True


def test_a_malformed_voyage_body_does_not_open_the_circuit():
    """A bad response is not an outage; the next request should still try Voyage."""
    client = Client(
        [
            Response(200, {"data": "not-a-list"}),
            Response(200, {"data": [{"index": 0, "relevance_score": 0.9}]}),
        ]
    )
    reranker, _now = _voyage_reranker(client, Local(), hosted_rerank_top_k=1)
    reranker.rerank_with_diagnostics("q", [chunk(1)])

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.actual_provider == "voyage"


def test_the_voyage_cooldown_can_be_disabled():
    import httpx as _httpx

    client = Client([_httpx.TimeoutException("slow")] * 4)
    reranker = ConfiguredReranker(
        settings=settings(hosted_rerank_cooldown_enabled=False, hosted_rerank_max_retries=1),
        local_factory=Local,
        voyage_factory=lambda **kwargs: VoyageReranker(client=client, **kwargs),
    )

    reranker.rerank_with_diagnostics("q", [chunk(1)])
    reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert len(client.calls) == 4, "both requests tried Voyage twice"


# --------------------------------------------------------------------------
# Hosted chain: Voyage, then Cohere, then local
# --------------------------------------------------------------------------


def cohere_response(indices: list[int]) -> Response:
    """Cohere v2 returns `results`, not `data`."""
    return Response(
        200,
        {
            "results": [
                {"index": i, "relevance_score": 1.0 - n * 0.1} for n, i in enumerate(indices)
            ]
        },
    )


def chained(voyage_client, cohere_client, local=None, **overrides):
    reranker = ConfiguredReranker(
        settings=settings(
            reranker_provider="voyage",
            reranker_hosted_fallback="cohere",
            cohere_api_key="c" * 40,
            cohere_rerank_model="rerank-v3.5",
            hosted_rerank_top_k=2,
            hosted_rerank_cooldown_enabled=True,
            **overrides,
        ),
        local_factory=lambda: local or Local(),
        voyage_factory=lambda **kw: VoyageReranker(client=voyage_client, **kw),
        cohere_factory=lambda **kw: CohereReranker(client=cohere_client, **kw),
    )
    return reranker


def test_cohere_takes_over_when_voyage_is_rate_limited():
    """The case this chain exists for: Voyage allows 3 requests a minute."""
    voyage = Client([Response(429, {"detail": "rate limited"})])
    cohere = Client([cohere_response([1, 0])])
    reranker = chained(voyage, cohere, hosted_rerank_max_retries=0)

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert result.reranker_used is True
    assert result.actual_provider == "cohere"
    assert result.requested_provider == "voyage"
    assert result.fallback_used is True
    assert [c.chunk_id for c in result.chunks] == [2, 1], "Cohere's order, not Voyage's"


def test_cohere_results_are_parsed_from_its_own_envelope():
    """Voyage returns `data`; Cohere returns `results`. Both must work."""
    cohere = Client([cohere_response([0, 1])])
    reranker = chained(Client([Response(429, {})]), cohere, hosted_rerank_max_retries=0)

    result = reranker.rerank_with_diagnostics("q", [chunk(7), chunk(8)])

    assert [c.chunk_id for c in result.chunks] == [7, 8]
    assert result.provider_raw_scores == {7: 1.0, 8: 0.9}


def test_cohere_is_sent_top_n_not_top_k():
    cohere = Client([cohere_response([0, 1])])
    reranker = chained(Client([Response(429, {})]), cohere, hosted_rerank_max_retries=0)

    reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    _url, body = cohere.calls[0]
    assert body["top_n"] == 2
    assert "top_k" not in body


def test_both_hosted_providers_failing_falls_back_to_local():
    local = Local()
    reranker = chained(
        Client([Response(429, {})]),
        Client([Response(429, {})]),
        local=local,
        hosted_rerank_max_retries=0,
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1), chunk(2)])

    assert result.actual_provider == "local"
    assert result.fallback_used is True
    assert local.calls == 1


def test_an_unconfigured_cohere_key_is_skipped_without_a_request():
    """A key alone enables a provider; its absence must not cost a round trip."""
    cohere = Client([])
    local = Local()
    reranker = ConfiguredReranker(
        settings=settings(
            reranker_provider="voyage",
            reranker_hosted_fallback="cohere",
            cohere_api_key=None,
            hosted_rerank_max_retries=0,
        ),
        local_factory=lambda: local,
        voyage_factory=lambda **kw: VoyageReranker(client=Client([Response(429, {})]), **kw),
        cohere_factory=lambda **kw: CohereReranker(client=cohere, **kw),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert cohere.calls == [], "no request to a provider without a key"
    assert result.actual_provider == "local"


def test_a_cooling_voyage_goes_straight_to_cohere():
    voyage = Client([Response(429, {}), cohere_response([0])])
    cohere = Client([cohere_response([0]), cohere_response([0])])
    reranker = chained(voyage, cohere, hosted_rerank_max_retries=0)
    now = [1_000.0]
    reranker._clock = lambda: now[0]

    reranker.rerank_with_diagnostics("q", [chunk(1)])
    voyage_calls_after_first = len(voyage.calls)
    second = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert len(voyage.calls) == voyage_calls_after_first, "Voyage was not retried while cooling"
    assert second.actual_provider == "cohere"


def test_no_hosted_fallback_configured_keeps_the_original_behaviour():
    local = Local()
    reranker = ConfiguredReranker(
        settings=settings(
            reranker_provider="voyage", reranker_hosted_fallback="none", hosted_rerank_max_retries=0
        ),
        local_factory=lambda: local,
        voyage_factory=lambda **kw: VoyageReranker(client=Client([Response(429, {})]), **kw),
    )

    result = reranker.rerank_with_diagnostics("q", [chunk(1)])

    assert result.actual_provider == "local"
