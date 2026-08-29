"""Mock-only contracts for the explicit hosted reranker profile."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx

from src.config import Settings
from src.reranking.cross_encoder import RerankResult, sigmoid
from src.reranking.provider import ConfiguredReranker, VoyageReranker
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

    result = reranker.rerank_with_diagnostics("sensitive query", [first, second], top_k=2, candidate_top_k=2)

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
                        {"index": index, "relevance_score": float(20 - index)}
                        for index in range(5)
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


def test_voyage_warmup_checks_configuration_without_a_network_call():
    reranker = ConfiguredReranker(settings=settings())

    assert reranker.warmup() is True
    assert reranker.is_model_loaded is True
    assert reranker.loaded_model_name == "rerank-2.5-lite"


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
