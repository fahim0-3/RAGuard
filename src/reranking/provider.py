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

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import httpx

from src.config import get_settings
from src.reranking.cross_encoder import CrossEncoderReranker, RerankResult
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

__all__ = [
    "ConfiguredReranker",
    "VoyageReranker",
    "VoyageRerankerError",
]


class HostedRerankerError(RuntimeError):
    """Controlled operational failure. Its message is always safe to expose.

    The code is prefixed with the provider name, so a trace says which hosted
    reranker refused and why without carrying its response body.
    """

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


#: The original name, kept because tests and callers still raise and catch it.
VoyageRerankerError = HostedRerankerError


class HostedReranker:
    """Shared HTTP adapter for a hosted rerank endpoint.

    Voyage and Cohere differ only in the URL, the name of the top-k field, and
    the key their results arrive under; retries, failure classification and
    result parsing are identical and live here. A subclass supplies those three
    differences and its own name.

    The client is constructed without making a network request and contains no
    logging hooks, so the Authorization header and document contents cannot be
    written to application logs by this module. Tests inject a mock client and
    never contact a provider.
    """

    #: Set by each subclass.
    provider: str = ""
    endpoint: str = ""

    def _payload(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _results(body: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _error(self, reason: str, *, retryable: bool = False) -> HostedRerankerError:
        return HostedRerankerError(f"{self.provider}_{reason}", retryable=retryable)

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

    def close(self) -> None:
        """Release the underlying persistent HTTP client at service shutdown."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _backoff_seconds(retry_number: int) -> float:
        """Short bounded backoff; a query never waits unboundedly on retries."""
        return min(0.25 * (2**retry_number), 1.0)

    def _request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
        retries = 0
        started = time.perf_counter()
        while True:
            try:
                response = self._client.post(self.endpoint, json=payload)
            except httpx.TimeoutException:
                failure = self._error("timeout", retryable=True)
            except httpx.HTTPError:
                failure = self._error("unavailable", retryable=True)
            else:
                status_code = int(getattr(response, "status_code", 0))
                if status_code == 429:
                    failure = self._error("rate_limited", retryable=True)
                elif status_code >= 500:
                    failure = self._error("unavailable", retryable=True)
                elif status_code < 200 or status_code >= 300:
                    failure = self._error("request_rejected")
                else:
                    try:
                        body = response.json()
                    except (TypeError, ValueError):
                        failure = self._error("malformed_response")
                    else:
                        if not isinstance(body, dict):
                            failure = self._error("malformed_response")
                        else:
                            return body, retries, (time.perf_counter() - started) * 1000.0

            if not failure.retryable or retries >= self.max_retries:
                failure.args = (failure.code,)
                failure.retry_count = retries  # type: ignore[attr-defined]
                failure.latency_ms = (time.perf_counter() - started) * 1000.0  # type: ignore[attr-defined]
                raise failure
            self._sleep(self._backoff_seconds(retries))
            retries += 1

    def _parse_order(
        self, body: dict[str, Any], candidates: list[RetrievedChunk], top_k: int
    ) -> tuple[list[RetrievedChunk], dict[int, float], list[int]]:
        data = self._results(body)
        if not isinstance(data, list) or len(data) < top_k:
            raise self._error("malformed_response")

        indexed: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in data:
            if not isinstance(item, dict):
                raise self._error("malformed_response")
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
                raise self._error("malformed_response")
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
                requested_provider=self.provider,
                actual_provider=self.provider,
                confidence_score_source="unverified_hosted_order_only",
            )

        payload = self._payload(query, [chunk.content for chunk in candidates], effective_top_k)
        try:
            body, retries, latency_ms = self._request(payload)
            ordered, raw_scores, provider_order = self._parse_order(
                body, candidates, effective_top_k
            )
        except HostedRerankerError as exc:
            return RerankResult(
                query=query,
                chunks=candidates[:effective_top_k],
                model_name=self.model_name,
                candidate_count=len(candidates),
                failure=exc.code,
                failure_stage="hosted",
                requested_provider=self.provider,
                actual_provider=self.provider,
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
            requested_provider=self.provider,
            actual_provider=self.provider,
            confidence_score_source="unverified_hosted_order_only",
        )


class VoyageReranker(HostedReranker):
    """Voyage's rerank endpoint: results under `data`, top-k named `top_k`."""

    provider = "voyage"
    endpoint = VOYAGE_RERANK_URL

    def _payload(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_k": top_k,
        }

    @staticmethod
    def _results(body: dict[str, Any]) -> Any:
        return body.get("data")


class CohereReranker(HostedReranker):
    """Cohere's v2 rerank endpoint: results under `results`, top-k named `top_n`."""

    provider = "cohere"
    endpoint = COHERE_RERANK_URL

    def _payload(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

    @staticmethod
    def _results(body: dict[str, Any]) -> Any:
        return body.get("results")


def _await_prescore(future: Future[RerankResult] | None) -> RerankResult | None:
    """Collect an overlapped pre-score, or None so the caller scores afresh."""
    if future is None:
        return None
    try:
        return future.result()
    except Exception:  # noqa: BLE001 - the sequential pass is the fallback
        logger.warning("Overlapped local scoring failed; scoring sequentially")
        return None


def _project_prescored(
    prescored: RerankResult | None, voyage_chunks: list[RetrievedChunk]
) -> RerankResult | None:
    """Carry pool scores onto Voyage's picks, in Voyage's order.

    Returns None unless every pick was scored, so a partial or degraded
    pre-score can never leave a chunk without its confidence value; the
    caller then runs the original sequential pass.
    """
    if prescored is None or not prescored.reranker_used:
        return None
    by_id = {chunk.chunk_id: chunk for chunk in prescored.chunks}
    if any(chunk.chunk_id not in by_id for chunk in voyage_chunks):
        return None
    projected = [
        replace(
            chunk,
            rerank_score=by_id[chunk.chunk_id].rerank_score,
            normalised_rerank_score=by_id[chunk.chunk_id].normalised_rerank_score,
        )
        for chunk in voyage_chunks
    ]
    return replace(
        prescored,
        chunks=projected,
        candidate_count=len(projected),
        provider_raw_scores={
            chunk.chunk_id: prescored.provider_raw_scores.get(chunk.chunk_id)
            for chunk in voyage_chunks
        },
        provider_order=[chunk.chunk_id for chunk in voyage_chunks],
    )


#: Seconds Voyage is skipped after each transient failure. A rejected
#: request or malformed body is not here: those are not outages.
_HOSTED_COOLDOWN_S: dict[str, float] = {
    "timeout": 30.0,
    "unavailable": 60.0,
    "rate_limited": 60.0,
}


class ConfiguredReranker:
    """Select the configured provider once, with an explicit local fallback."""

    def __init__(
        self,
        *,
        settings: Any | None = None,
        local_factory: Callable[[], CrossEncoderReranker] = CrossEncoderReranker,
        voyage_factory: Callable[..., HostedReranker] = VoyageReranker,
        cohere_factory: Callable[..., HostedReranker] = CohereReranker,
    ) -> None:
        self.settings = settings or get_settings()
        self._local_factory = local_factory
        self._voyage_factory = voyage_factory
        self._cohere_factory = cohere_factory
        self._local: CrossEncoderReranker | None = None
        self._local_lock = threading.Lock()
        self._hosted_instances: dict[str, HostedReranker] = {}
        self._voyage_lock = threading.Lock()
        self._cooling_until: dict[str, float] = {}
        self._prescore_executor: ThreadPoolExecutor | None = None
        self._prescore_lock = threading.Lock()
        self._voyage_cooling_until = 0.0
        self._clock: Callable[[], float] = time.monotonic

    def _local_reranker(self) -> CrossEncoderReranker:
        if self._local is None:
            with self._local_lock:
                if self._local is None:
                    self._local = self._local_factory()
        return self._local

    def _hosted_configured(self, provider: str) -> bool:
        """Remote use is explicit: a key alone never sends policy text out."""
        if not self.settings.reranker_remote_allowed:
            return False
        if provider == "voyage":
            return bool(self.settings.voyage_api_key)
        if provider == "cohere":
            return bool(getattr(self.settings, "cohere_api_key", None))
        return False

    @property
    def _voyage_configured(self) -> bool:
        return self._hosted_configured("voyage")

    def _hosted_chain(self) -> list[str]:
        """Hosted providers to try, in order, without repeats."""
        chain = [self.settings.reranker_provider]
        second = getattr(self.settings, "reranker_hosted_fallback", "none")
        if second not in ("none", "", None) and second not in chain:
            chain.append(second)
        return [p for p in chain if p != "local"]

    @property
    def is_model_loaded(self) -> bool:
        if self.settings.reranker_provider == "voyage":
            # Voyage chooses the order, but the graph scores that order with
            # the local cross-encoder before evidence grading. Readiness must
            # therefore wait for the local scorer; otherwise a first user
            # request pays the full model-load cost.
            return bool(self._local and self._local.is_model_loaded)
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
            # The local scorer is part of the normal Voyage path, not only its
            # failure path. Load it before readiness opens so a request never
            # blocks on Hugging Face metadata and model construction.
            return self._voyage_configured and self._local_reranker().warmup()
        return self._local_reranker().warmup()

    def _hosted(self, provider: str) -> HostedReranker:
        """Build and cache one adapter per hosted provider."""
        existing = self._hosted_instances.get(provider)
        if existing is not None:
            return existing
        with self._voyage_lock:
            existing = self._hosted_instances.get(provider)
            if existing is None:
                if provider == "cohere":
                    existing = self._cohere_factory(
                        api_key=getattr(self.settings, "cohere_api_key", "") or "",
                        model_name=self.settings.cohere_rerank_model,
                        timeout_seconds=self.settings.hosted_rerank_timeout_seconds,
                        max_retries=self.settings.hosted_rerank_max_retries,
                    )
                else:
                    existing = self._voyage_factory(
                        api_key=self.settings.voyage_api_key or "",
                        model_name=self.settings.voyage_rerank_model,
                        timeout_seconds=self.settings.hosted_rerank_timeout_seconds,
                        max_retries=self.settings.hosted_rerank_max_retries,
                    )
                self._hosted_instances[provider] = existing
        return existing

    def _voyage(self) -> HostedReranker:
        return self._hosted("voyage")

    def close(self) -> None:
        """Close resources that are owned by this configured adapter."""
        with self._voyage_lock:
            adapters = list(self._hosted_instances.values())
            self._hosted_instances.clear()
        for adapter in adapters:
            adapter.close()
        with self._prescore_lock:
            executor, self._prescore_executor = self._prescore_executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _hosted_cooling(self, provider: str) -> bool:
        if not getattr(self.settings, "hosted_rerank_cooldown_enabled", False):
            return False
        return self._clock() < self._cooling_until.get(provider, 0.0)

    def _record_hosted_outcome(self, provider: str, result: RerankResult) -> None:
        if result.reranker_used:
            self._cooling_until.pop(provider, None)
            return
        # Codes are provider-prefixed; the cooldown table is not.
        reason = (result.failure or "").removeprefix(f"{provider}_")
        cooldown = _HOSTED_COOLDOWN_S.get(reason)
        if cooldown is not None:
            self._cooling_until[provider] = self._clock() + cooldown

    def _voyage_cooling(self) -> bool:
        return self._hosted_cooling("voyage")

    def _start_prescore(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> Future[RerankResult] | None:
        """Begin local scoring of the whole Voyage candidate pool.

        Runs while the hosted request is on the network, when the CPU would
        otherwise sit idle. The request's context is copied so the scorer
        still sees the request deadline.
        """
        if not getattr(self.settings, "reranker_overlap_local_scoring", False):
            return None
        if not candidates:
            return None
        with self._prescore_lock:
            if self._prescore_executor is None:
                self._prescore_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="rerank-prescore"
                )
            executor = self._prescore_executor
        context = contextvars.copy_context()
        return executor.submit(
            context.run,
            self._local_reranker().score_fixed_order_with_diagnostics,
            query,
            list(candidates),
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
        prescored: RerankResult | None = None,
        provider: str = "voyage",
    ) -> RerankResult:
        """Attach BGE confidence scores without letting BGE reorder hosted evidence."""
        scored = _project_prescored(prescored, voyage_result.chunks)
        if scored is None:
            scored = self._local_reranker().score_fixed_order_with_diagnostics(
                query, voyage_result.chunks
            )
        if not scored.reranker_used:
            # A hosted order without BGE-compatible confidence values must not
            # reach the evidence pipeline. Re-run the established local path;
            # it will itself degrade safely to RRF if the local model is gone.
            fallback = self._local_result(
                query,
                all_chunks,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                requested_provider=self.settings.reranker_provider,
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
            requested_provider=self.settings.reranker_provider,
            actual_provider=provider,
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

        # The local pool scoring is started once and reused whichever hosted
        # provider answers: it scores the candidate pool, not one provider's
        # picks, so it is valid for any ordering that comes back.
        prescore = None
        blocked: RerankResult | None = None
        for provider in self._hosted_chain():
            if not self._hosted_configured(provider):
                blocked = RerankResult(
                    query=query,
                    chunks=chunks[:hosted_top_k],
                    candidate_count=min(len(chunks), hosted_candidates),
                    failure=f"{provider}_remote_not_explicitly_enabled",
                    failure_stage="remote_permission",
                    requested_provider=provider,
                    actual_provider=provider,
                    confidence_score_source="none",
                )
                continue
            if self._hosted_cooling(provider):
                # It failed moments ago; trying it again would repeat the
                # timeout before reaching the same fallback.
                blocked = RerankResult(
                    query=query,
                    chunks=chunks[:hosted_top_k],
                    candidate_count=min(len(chunks), hosted_candidates),
                    failure=f"{provider}_cooling_down",
                    failure_stage="circuit_open",
                    requested_provider=provider,
                    actual_provider=provider,
                    confidence_score_source="none",
                )
                continue
            if prescore is None:
                prescore = self._start_prescore(query, chunks[:hosted_candidates])
            blocked = self._hosted(provider).rerank_with_diagnostics(
                query, chunks, top_k=hosted_top_k, candidate_top_k=hosted_candidates
            )
            self._record_hosted_outcome(provider, blocked)
            if blocked.reranker_used:
                hybrid = self._score_voyage_order_with_bge(
                    query,
                    blocked,
                    chunks,
                    top_k=top_k,
                    candidate_top_k=candidate_top_k,
                    prescored=_await_prescore(prescore),
                    provider=provider,
                )
                # `hybrid` may itself be a local result, when confidence scoring
                # failed; only mark a hosted fallback when hosted order was used.
                used_hosted = hybrid.actual_provider == provider
                return replace(
                    hybrid,
                    fallback_used=hybrid.fallback_used
                    or (used_hosted and provider != self.settings.reranker_provider),
                    total_reranker_latency_ms=(time.perf_counter() - reranker_started) * 1000.0,
                )
        _await_prescore(prescore)
        assert blocked is not None  # the chain always has one entry here

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
