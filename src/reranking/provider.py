"""Explicit local-or-Voyage reranker provider selection.

This module deliberately has no automatic routing.  A deployment selects one
provider in configuration for every graph run.  Voyage is usable only when both
``RERANKER_PROVIDER=voyage`` and ``RERANKER_REMOTE_ALLOWED=true`` are set;
merely adding an API key cannot transmit a query or policy passage.

Voyage relevance scores are useful evaluation artifacts, but they are not BGE
logits.  The adapter consequently uses them for ordering only and leaves both
``rerank_score`` and ``normalised_rerank_score`` untouched until a separately
evaluated confidence profile is implemented and approved.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx

from src.config import get_settings
from src.reranking.cross_encoder import CrossEncoderReranker, RerankResult
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

__all__ = [
    "ConfiguredReranker",
    "VoyageReranker",
    "VoyageRerankerError",
]


class VoyageRerankerError(RuntimeError):
    """Controlled operational failure. Its message is always safe to expose."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class VoyageReranker:
    """Small HTTP adapter for Voyage's rerank endpoint.

    The client is constructed without making a network request and contains no
    logging hooks, so the Authorization header and document contents cannot be
    written to application logs by this module. Tests inject a mock client and
    never contact Voyage.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        timeout_seconds: float,
        max_retries: int,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @staticmethod
    def _backoff_seconds(retry_number: int) -> float:
        """Short bounded backoff; a query never waits unboundedly on retries."""
        return min(0.25 * (2**retry_number), 1.0)

    def _request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        retries = 0
        started = time.perf_counter()
        while True:
            try:
                response = self._client.post(VOYAGE_RERANK_URL, json=payload)
            except httpx.TimeoutException:
                failure = VoyageRerankerError("voyage_timeout", retryable=True)
            except httpx.HTTPError:
                failure = VoyageRerankerError("voyage_unavailable", retryable=True)
            else:
                status_code = int(getattr(response, "status_code", 0))
                if status_code == 429:
                    failure = VoyageRerankerError("voyage_rate_limited", retryable=True)
                elif status_code >= 500:
                    failure = VoyageRerankerError("voyage_unavailable", retryable=True)
                elif status_code < 200 or status_code >= 300:
                    failure = VoyageRerankerError("voyage_request_rejected")
                else:
                    try:
                        body = response.json()
                    except (TypeError, ValueError):
                        failure = VoyageRerankerError("voyage_malformed_response")
                    else:
                        if not isinstance(body, dict):
                            failure = VoyageRerankerError("voyage_malformed_response")
                        else:
                            return body, retries, (time.perf_counter() - started) * 1000.0

            if not failure.retryable or retries >= self.max_retries:
                failure.args = (failure.code,)
                failure.retry_count = retries  # type: ignore[attr-defined]
                failure.latency_ms = (time.perf_counter() - started) * 1000.0  # type: ignore[attr-defined]
                raise failure
            self._sleep(self._backoff_seconds(retries))
            retries += 1

    @staticmethod
    def _parse_order(
        body: dict[str, Any], candidates: list[RetrievedChunk], top_k: int
    ) -> tuple[list[RetrievedChunk], dict[int, float], list[int]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) < top_k:
            raise VoyageRerankerError("voyage_malformed_response")

        indexed: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in data:
            if not isinstance(item, dict):
                raise VoyageRerankerError("voyage_malformed_response")
            index = item.get("index")
            score = item.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(candidates)
                or index in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                raise VoyageRerankerError("voyage_malformed_response")
            seen.add(index)
            indexed.append((index, float(score)))

        # Responses are normally score-sorted, but ordering explicitly makes
        # the contract deterministic even if an API implementation changes.
        indexed.sort(key=lambda item: (-item[1], candidates[item[0]].chunk_id))
        selected = indexed[:top_k]
        return (
            [candidates[index] for index, _score in selected],
            {candidates[index].chunk_id: score for index, score in indexed},
            [candidates[index].chunk_id for index, _score in selected],
        )

    def rerank_with_diagnostics(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int, candidate_top_k: int
    ) -> RerankResult:
        candidates = chunks[:candidate_top_k]
        effective_top_k = min(top_k, len(candidates))
        if not candidates:
            return RerankResult(
                query=query,
                chunks=[],
                model_name=self.model_name,
                requested_provider="voyage",
                actual_provider="voyage",
                confidence_score_source="unverified_hosted_order_only",
            )

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [chunk.content for chunk in candidates],
            "top_k": effective_top_k,
        }
        try:
            body, retries, latency_ms = self._request(payload)
            ordered, raw_scores, provider_order = self._parse_order(
                body, candidates, effective_top_k
            )
        except VoyageRerankerError as exc:
            return RerankResult(
                query=query,
                chunks=candidates[:effective_top_k],
                model_name=self.model_name,
                candidate_count=len(candidates),
                failure=exc.code,
                failure_stage="hosted",
                requested_provider="voyage",
                actual_provider="voyage",
                hosted_latency_ms=float(getattr(exc, "latency_ms", 0.0)),
                retry_count=int(getattr(exc, "retry_count", 0)),
                confidence_score_source="unverified_hosted_order_only",
            )

        return RerankResult(
            query=query,
            chunks=ordered,
            reranker_used=True,
            model_name=self.model_name,
            candidate_count=len(candidates),
            hosted_latency_ms=latency_ms,
            retry_count=retries,
            provider_raw_scores=raw_scores,
            provider_order=provider_order,
            requested_provider="voyage",
            actual_provider="voyage",
            confidence_score_source="unverified_hosted_order_only",
        )


class ConfiguredReranker:
    """Select the configured provider once, with an explicit local fallback."""

    def __init__(
        self,
        *,
        settings: Any | None = None,
        local_factory: Callable[[], CrossEncoderReranker] = CrossEncoderReranker,
        voyage_factory: Callable[..., VoyageReranker] = VoyageReranker,
    ) -> None:
        self.settings = settings or get_settings()
        self._local_factory = local_factory
        self._voyage_factory = voyage_factory
        self._local: CrossEncoderReranker | None = None
        self._local_lock = threading.Lock()

    def _local_reranker(self) -> CrossEncoderReranker:
        if self._local is None:
            with self._local_lock:
                if self._local is None:
                    self._local = self._local_factory()
        return self._local

    @property
    def _voyage_configured(self) -> bool:
        return bool(self.settings.reranker_remote_allowed and self.settings.voyage_api_key)

    @property
    def is_model_loaded(self) -> bool:
        if self.settings.reranker_provider == "voyage" and self._local is None:
            # Hosted readiness is configuration readiness; no startup request
            # is made because that would send data and consume quota.
            return self._voyage_configured
        return bool(self._local and self._local.is_model_loaded)

    @property
    def load_error(self) -> str | None:
        if self.settings.reranker_provider == "voyage" and not self._voyage_configured:
            return "voyage_remote_not_explicitly_enabled"
        return self._local.load_error if self._local else None

    @property
    def loaded_model_name(self) -> str | None:
        if self._local and self._local.loaded_model_name:
            return self._local.loaded_model_name
        if self.settings.reranker_provider == "voyage" and self._voyage_configured:
            return self.settings.voyage_rerank_model
        return None

    def warmup(self) -> bool:
        if not self.settings.reranker_enabled:
            return True
        if self.settings.reranker_provider == "voyage":
            # Preserve the hosted fast path: do not allocate local model RAM
            # unless a hosted request actually fails and needs fallback.
            return self._voyage_configured
        return self._local_reranker().warmup()

    def _voyage(self) -> VoyageReranker:
        return self._voyage_factory(
            api_key=self.settings.voyage_api_key or "",
            model_name=self.settings.voyage_rerank_model,
            timeout_seconds=self.settings.hosted_rerank_timeout_seconds,
            max_retries=self.settings.hosted_rerank_max_retries,
        )

    def _local_result(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int,
        candidate_top_k: int,
        requested_provider: str,
        fallback_used: bool = False,
        hosted_result: RerankResult | None = None,
    ) -> RerankResult:
        result = self._local_reranker().rerank_with_diagnostics(
            query, chunks, top_k=top_k, candidate_top_k=candidate_top_k
        )
        return replace(
            result,
            requested_provider=requested_provider,
            actual_provider="local",
            fallback_used=fallback_used,
            hosted_latency_ms=hosted_result.hosted_latency_ms if hosted_result else 0.0,
            retry_count=hosted_result.retry_count if hosted_result else 0,
            # Retain a controlled hosted failure code without serialising the
            # provider response or any potential sensitive error body.
            failure=hosted_result.failure if hosted_result else result.failure,
            failure_stage=hosted_result.failure_stage if hosted_result else result.failure_stage,
        )

    def _score_voyage_order_with_bge(
        self,
        query: str,
        voyage_result: RerankResult,
        all_chunks: list[RetrievedChunk],
        *,
        top_k: int,
        candidate_top_k: int,
    ) -> RerankResult:
        """Attach BGE confidence scores without letting BGE reorder Voyage evidence."""
        scored = self._local_reranker().score_fixed_order_with_diagnostics(query, voyage_result.chunks)
        if not scored.reranker_used:
            # A hosted order without BGE-compatible confidence values must not
            # reach the evidence pipeline. Re-run the established local path;
            # it will itself degrade safely to RRF if the local model is gone.
            fallback = self._local_result(
                query,
                all_chunks,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                requested_provider="voyage",
                fallback_used=True,
                hosted_result=voyage_result,
            )
            return replace(
                fallback,
                failure=scored.failure or voyage_result.failure,
                failure_stage=scored.failure_stage or voyage_result.failure_stage,
                bge_scoring_latency_ms=scored.bge_scoring_latency_ms,
                bge_scoring_cpu_time_ms=scored.bge_scoring_cpu_time_ms,
            )

        # `scored.chunks` has the exact Voyage order. BGE logits are used only
        # for the existing confidence/evidence thresholds, never for sorting.
        return RerankResult(
            query=query,
            chunks=scored.chunks,
            reranker_used=True,
            model_name=scored.model_name,
            candidate_count=voyage_result.candidate_count,
            fallback_model_used=scored.fallback_model_used,
            queue_wait_ms=scored.queue_wait_ms,
            inference_latency_ms=scored.inference_latency_ms,
            bge_scoring_latency_ms=scored.bge_scoring_latency_ms,
            bge_scoring_cpu_time_ms=scored.bge_scoring_cpu_time_ms,
            requested_provider="voyage",
            actual_provider="voyage",
            hosted_latency_ms=voyage_result.hosted_latency_ms,
            retry_count=voyage_result.retry_count,
            # Voyage scores stay isolated for evaluation; the chunk fields came
            # only from the BGE fixed-order score pass above.
            provider_raw_scores=voyage_result.provider_raw_scores,
            provider_order=voyage_result.provider_order,
            confidence_score_source="bge_sigmoid_fixed_voyage_order",
        )

    def rerank_with_diagnostics(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        candidate_top_k: int | None = None,
    ) -> RerankResult:
        top_k = top_k or self.settings.rerank_top_k
        candidate_top_k = candidate_top_k or self.settings.rerank_candidate_top_k
        if not self.settings.reranker_enabled:
            return RerankResult(
                query=query,
                chunks=chunks[: min(top_k, candidate_top_k)],
                candidate_count=min(len(chunks), candidate_top_k),
                failure="reranker disabled by configuration",
                failure_stage="disabled",
                requested_provider=self.settings.reranker_provider,
                actual_provider="rrf",
                confidence_score_source="none",
            )

        if self.settings.reranker_provider == "local":
            return self._local_result(
                query,
                chunks,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                requested_provider="local",
            )

        reranker_started = time.perf_counter()
        hosted_top_k = min(top_k, self.settings.hosted_rerank_top_k)
        hosted_candidates = min(candidate_top_k, self.settings.hosted_rerank_max_candidates)
        if not self._voyage_configured:
            blocked = RerankResult(
                query=query,
                chunks=chunks[:hosted_top_k],
                candidate_count=min(len(chunks), hosted_candidates),
                failure="voyage_remote_not_explicitly_enabled",
                failure_stage="remote_permission",
                requested_provider="voyage",
                actual_provider="voyage",
                confidence_score_source="none",
            )
        else:
            blocked = self._voyage().rerank_with_diagnostics(
                query, chunks, top_k=hosted_top_k, candidate_top_k=hosted_candidates
            )
            if blocked.reranker_used:
                hybrid = self._score_voyage_order_with_bge(
                    query, blocked, chunks, top_k=top_k, candidate_top_k=candidate_top_k
                )
                return replace(
                    hybrid,
                    total_reranker_latency_ms=(time.perf_counter() - reranker_started) * 1000.0,
                )

        if self.settings.reranker_fallback_provider == "local":
            fallback = self._local_result(
                query,
                chunks,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                requested_provider="voyage",
                fallback_used=True,
                hosted_result=blocked,
            )
            return replace(
                fallback,
                total_reranker_latency_ms=(time.perf_counter() - reranker_started) * 1000.0,
            )
        return replace(
            blocked,
            actual_provider="rrf",
            fallback_used=True,
            total_reranker_latency_ms=(time.perf_counter() - reranker_started) * 1000.0,
        )

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        return self.rerank_with_diagnostics(query, chunks, top_k=top_k).chunks

    def config(self) -> dict[str, Any]:
        return {
            "requested_provider": self.settings.reranker_provider,
            "remote_allowed": self.settings.reranker_remote_allowed,
            "fallback_provider": self.settings.reranker_fallback_provider,
            "hosted_model": self.settings.voyage_rerank_model,
            "hosted_max_candidates": self.settings.hosted_rerank_max_candidates,
            "hosted_top_k": self.settings.hosted_rerank_top_k,
            "confidence_profile": self.settings.reranker_confidence_profile,
            "local": self._local.config() if self._local else None,
        }
