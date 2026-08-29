"""Cross-encoder reranking (Phase C).

Retrieval decides *what is available*; the cross-encoder decides *what is
relevant*. RRF fuses two rankings that never see the query and the document
together, so it cannot tell a document that shares vocabulary with the query
from one that answers it. A cross-encoder reads the pair jointly and scores it,
which is exactly the failure mode the Phase B baseline exposed: the correct
document was inside the candidate set for every golden case, but ranked second
(GC-002) or fourth (GC-010).

The reranker does two jobs in RAGuard:

1. It reorders the fusion candidates by true query-document relevance.
2. Its top score is the *retrieval confidence signal* that drives self-healing.

Scores
------
Raw cross-encoder outputs are unbounded logits. `rerank_score` holds the raw
value and drives ordering; `normalised_rerank_score` squashes it through a
sigmoid into [0, 1] and is the only value compared against thresholds.

sentence-transformers may apply its own sigmoid inside `predict`, which would
make a second sigmoid here silently wrong for the confidence thresholds. The
identity activation is therefore requested explicitly, so what comes back is
always a logit and the normalisation is always ours.

Failure policy
--------------
A reranker is an *enhancement*. Retrieval already produced a usable ranking, so
a model that cannot load, or inference that throws, must never take down the
system. Every failure path returns the RRF ordering untouched with
`reranker_used=False`, and the RRF scores are preserved exactly. A failed load
is remembered so a 2 GB download is not retried once per query.

Cost warning: the primary model is ~568 M parameters. On CPU expect roughly 0.3
to 1.0 seconds per 20 candidates. Keep `RERANK_CANDIDATE_TOP_K` modest unless a
GPU is available.
"""

from __future__ import annotations

import inspect
import logging
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

__all__ = [
    "CrossEncoderReranker",
    "RerankResult",
    "get_reranker",
    "is_reranker_model_loaded",
    "loaded_reranker_model_name",
    "reranker_model_load_error",
    "reset_reranker",
    "sigmoid",
    "warmup_reranker_model",
]


def sigmoid(x: float) -> float:
    # Guard against overflow on large negative logits.
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(slots=True)
class RerankResult:
    """One reranking, including the reason it did not happen."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    reranker_used: bool = False
    model_name: str | None = None
    candidate_count: int = 0
    fallback_model_used: bool = False
    failure: str | None = None
    failure_stage: str | None = None
    queue_wait_ms: float = 0.0
    inference_latency_ms: float = 0.0
    # BGE scoring is distinct from hosted ordering in the hybrid Voyage path.
    # For local-only reranking these mirror the full inference measurements.
    bge_scoring_latency_ms: float = 0.0
    bge_scoring_cpu_time_ms: float = 0.0
    total_reranker_latency_ms: float = 0.0
    # Provider observability is deliberately metadata-only: it contains no
    # query or passage text and never includes credentials.
    requested_provider: str = "local"
    actual_provider: str = "local"
    fallback_used: bool = False
    hosted_latency_ms: float = 0.0
    retry_count: int = 0
    # Hosted relevance scores are retained separately for offline calibration.
    # They are never written into RetrievedChunk's BGE-specific score fields.
    provider_raw_scores: dict[int, float] = field(default_factory=dict)
    provider_order: list[int] = field(default_factory=list)
    confidence_score_source: str = "bge_sigmoid"

    def observability_dict(self) -> dict[str, Any]:
        """Safe operational metadata: never query text, chunks, or credentials."""
        return {
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            "fallback_used": self.fallback_used,
            "candidate_count": self.candidate_count,
            "returned_count": len(self.chunks),
            "reranker_used": self.reranker_used,
            "model_name": self.model_name,
            "hosted_latency_ms": round(self.hosted_latency_ms, 1),
            "reranking_latency_ms": round(
                self.total_reranker_latency_ms
                or (self.hosted_latency_ms + self.queue_wait_ms + self.inference_latency_ms),
                1,
            ),
            "inference_latency_ms": round(self.inference_latency_ms, 1),
            "bge_scoring_latency_ms": round(self.bge_scoring_latency_ms, 1),
            "bge_scoring_cpu_time_ms": round(self.bge_scoring_cpu_time_ms, 1),
            "queue_wait_ms": round(self.queue_wait_ms, 1),
            "retry_count": self.retry_count,
            "failure": self.failure,
            "failure_stage": self.failure_stage,
            "confidence_score_source": self.confidence_score_source,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            **self.observability_dict(),
            "fallback_model_used": self.fallback_model_used,
            "provider_score_count": len(self.provider_raw_scores),
        }


class CrossEncoderReranker:
    """Scores (query, chunk) pairs and reorders candidates by relevance."""

    def __init__(
        self,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        enabled: bool | None = None,
        model: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.resolved_reranker_model
        self.fallback_model_name = fallback_model_name or settings.reranker_fallback_model
        requested_device = device or settings.reranker_device.strip() or settings.model_device
        self.requested_device = requested_device.lower()
        self.device = (
            settings.resolved_reranker_device
            if self.requested_device == "auto"
            else requested_device
        )
        self._preferred_device = self.device
        self.enabled = settings.reranker_enabled if enabled is None else enabled
        configured_batch_size = batch_size or settings.reranker_batch_size
        self.batch_size = configured_batch_size or (32 if self.device.startswith("cuda") else 16)
        self.max_length = settings.reranker_max_length
        self.max_concurrency = settings.reranker_max_concurrency
        self.cpu_threads = settings.reranker_cpu_threads
        self.warmup_inference = settings.reranker_warmup_inference

        self._model: Any | None = model
        self._loaded_model_name: str | None = self.model_name if model is not None else None
        self._load_attempted = model is not None
        self._load_error: str | None = None
        self._activation_parameter: str | None = None
        self._lock = threading.Lock()
        self._inference_limiter = threading.BoundedSemaphore(self.max_concurrency)

    # -- model loading -----------------------------------------------------

    def _load(self, model_name: str) -> CrossEncoder:
        """Construct the underlying model. Overridden in tests."""
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s on %s", model_name, self.device)
        return CrossEncoder(model_name, device=self.device, max_length=self.max_length)

    def _configure_cpu_threads(self) -> None:
        """Apply an explicit CPU-thread ceiling only when an operator chose one."""
        if not self.device.startswith("cpu") or not self.cpu_threads:
            return
        try:
            import torch

            torch.set_num_threads(self.cpu_threads)
        except ImportError:  # pragma: no cover - CrossEncoder requires torch in production
            logger.warning("RERANKER_CPU_THREADS ignored because PyTorch is unavailable")

    def _load_on_device(self, model_name: str, device: str) -> Any:
        self.device = device
        self._configure_cpu_threads()
        return self._load(model_name)

    def _candidate_devices(self) -> tuple[str, ...]:
        """Use CPU only as the bounded fallback for an automatic CUDA choice."""
        if self.requested_device == "auto" and self._preferred_device.startswith("cuda"):
            return (self._preferred_device, "cpu")
        return (self._preferred_device,)

    def _get_model(self) -> Any | None:
        """Return the model, or None if it cannot be loaded.

        Never raises. The failure is recorded once; subsequent calls return None
        immediately rather than retrying the download.
        """
        if self._model is not None or self._load_attempted:
            return self._model

        with self._lock:
            if self._model is not None or self._load_attempted:
                return self._model
            self._load_attempted = True

            for candidate, is_fallback in (
                (self.model_name, False),
                (self.fallback_model_name, True),
            ):
                if not candidate or (is_fallback and candidate == self.model_name):
                    continue
                candidate_devices = self._candidate_devices()
                for device in candidate_devices:
                    try:
                        self._model = self._load_on_device(candidate, device)
                        self._loaded_model_name = candidate
                        self._load_error = None
                        if device != candidate_devices[0]:
                            logger.warning(
                                "Cross-encoder CUDA initialization failed; using CPU fallback for %s",
                                candidate,
                            )
                        if is_fallback:
                            logger.warning(
                                "Primary reranker %s unavailable; using fallback %s",
                                self.model_name,
                                candidate,
                            )
                        return self._model
                    except Exception as exc:  # noqa: BLE001 - any load failure degrades to RRF
                        self._load_error = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "Cross-encoder %s failed to load on %s: %s", candidate, device, exc
                        )

            logger.error("No cross-encoder could be loaded; retrieval falls back to RRF order")
            return None

    def _activation_parameter_name(self, model: Any) -> str | None:
        """Name of `predict`'s activation argument, which was renamed upstream.

        sentence-transformers used `activation_fct` through v4 and `activation_fn`
        from v5. Passing the identity under whichever name exists guarantees raw
        logits, so `normalised_rerank_score` is always our own sigmoid.
        """
        if self._activation_parameter is not None:
            return self._activation_parameter or None
        try:
            params = inspect.signature(model.predict).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            self._activation_parameter = ""
            return None
        for name in ("activation_fn", "activation_fct"):
            if name in params:
                self._activation_parameter = name
                return name
        self._activation_parameter = ""
        return None

    def _predict(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        kwargs: dict[str, Any] = {"batch_size": self.batch_size}
        activation_param = self._activation_parameter_name(model)
        if activation_param:
            import torch

            kwargs[activation_param] = torch.nn.Identity()
        return [float(score) for score in model.predict(pairs, **kwargs)]

    # -- public API --------------------------------------------------------

    @property
    def is_model_loaded(self) -> bool:
        """Whether weights are resident; this probe never initiates a load."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded_model_name(self) -> str | None:
        return self._loaded_model_name

    def warmup(self) -> bool:
        """Load weights and initialize one synthetic inference off the request path."""
        if not self.enabled:
            return True
        model = self._get_model()
        if model is None:
            return False
        if self.warmup_inference and hasattr(model, "predict"):
            try:
                self._predict(model, [("warmup", "warmup")])
            except Exception as exc:  # noqa: BLE001 - loaded weights remain usable
                logger.warning("Reranker inference warm-up failed: %s", type(exc).__name__)
        return True

    def score_fixed_order_with_diagnostics(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> RerankResult:
        """Score an already-ranked evidence set without changing its order.

        The Voyage hybrid path uses this method after hosted ordering.  It
        supplies the existing BGE-compatible confidence values for every final
        evidence chunk while deliberately preserving Voyage's top-five order.
        Exactly one `predict` call is made for the complete supplied list.
        """

        candidates = list(chunks)

        def degraded(failure: str | None, stage: str | None) -> RerankResult:
            return RerankResult(
                query=query,
                chunks=candidates,
                reranker_used=False,
                model_name=self._loaded_model_name,
                candidate_count=len(candidates),
                failure=failure,
                failure_stage=stage,
                confidence_score_source="none",
            )

        if not candidates:
            return degraded(None, None)
        if not self.enabled:
            return degraded("reranker disabled by configuration", "disabled")

        model = self._get_model()
        if model is None:
            return degraded(self._load_error or "model unavailable", "load")

        queued_at = time.perf_counter()
        inference_started = queued_at
        cpu_started = time.process_time()
        try:
            with self._inference_limiter:
                inference_started = time.perf_counter()
                cpu_started = time.process_time()
                scores = self._predict(model, [(query, chunk.content) for chunk in candidates])
                inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                cpu_time_ms = (time.process_time() - cpu_started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - scoring must fail closed
            logger.error("Cross-encoder fixed-order scoring failed: %s", exc)
            return degraded(f"{type(exc).__name__}: {exc}", "inference")
        queue_wait_ms = (inference_started - queued_at) * 1000.0

        if len(scores) != len(candidates):
            message = f"model returned {len(scores)} scores for {len(candidates)} candidates"
            logger.error("Cross-encoder contract violated: %s", message)
            return degraded(message, "inference")

        # Do not sort.  BGE is a confidence scorer here, not a second reranker.
        scored = [
            replace(chunk, rerank_score=score, normalised_rerank_score=sigmoid(score))
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        return RerankResult(
            query=query,
            chunks=scored,
            reranker_used=True,
            model_name=self._loaded_model_name,
            candidate_count=len(candidates),
            fallback_model_used=self._loaded_model_name != self.model_name,
            queue_wait_ms=queue_wait_ms,
            inference_latency_ms=inference_latency_ms,
            bge_scoring_latency_ms=inference_latency_ms,
            bge_scoring_cpu_time_ms=cpu_time_ms,
            total_reranker_latency_ms=queue_wait_ms + inference_latency_ms,
            provider_raw_scores={
                chunk.chunk_id: score for chunk, score in zip(candidates, scores, strict=True)
            },
            provider_order=[chunk.chunk_id for chunk in candidates],
        )

    def rerank_with_diagnostics(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        candidate_top_k: int | None = None,
    ) -> RerankResult:
        """Rerank candidates, reporting whether the cross-encoder was actually used."""
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k
        candidate_top_k = candidate_top_k or settings.rerank_candidate_top_k
        candidates = chunks[:candidate_top_k]

        def degraded(failure: str | None, stage: str | None) -> RerankResult:
            """RRF order, untouched scores, reranker_used=False."""
            return RerankResult(
                query=query,
                chunks=candidates[:top_k],
                reranker_used=False,
                model_name=self._loaded_model_name,
                candidate_count=len(candidates),
                failure=failure,
                failure_stage=stage,
            )

        if not candidates:
            return degraded(None, None)
        if not self.enabled:
            return degraded("reranker disabled by configuration", "disabled")

        model = self._get_model()
        if model is None:
            return degraded(self._load_error or "model unavailable", "load")

        queued_at = time.perf_counter()
        inference_started = queued_at
        cpu_started = time.process_time()
        try:
            with self._inference_limiter:
                inference_started = time.perf_counter()
                cpu_started = time.process_time()
                scores = self._predict(model, [(query, chunk.content) for chunk in candidates])
                inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                cpu_time_ms = (time.process_time() - cpu_started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - inference failure degrades to RRF
            logger.error("Cross-encoder inference failed: %s", exc)
            return degraded(f"{type(exc).__name__}: {exc}", "inference")
        queue_wait_ms = (inference_started - queued_at) * 1000.0

        if len(scores) != len(candidates):
            message = f"model returned {len(scores)} scores for {len(candidates)} candidates"
            logger.error("Cross-encoder contract violated: %s", message)
            return degraded(message, "inference")

        # `replace` copies every other field, so the BM25, vector, and RRF
        # scores and the retriever ranks survive reranking intact.
        scored = [
            replace(chunk, rerank_score=score, normalised_rerank_score=sigmoid(score))
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        # Ties break on chunk_id, matching RRF, so repeated runs are identical.
        scored.sort(key=lambda c: (-(c.rerank_score or float("-inf")), c.chunk_id))

        return RerankResult(
            query=query,
            chunks=scored[:top_k],
            reranker_used=True,
            model_name=self._loaded_model_name,
            candidate_count=len(candidates),
            fallback_model_used=self._loaded_model_name != self.model_name,
            queue_wait_ms=queue_wait_ms,
            inference_latency_ms=inference_latency_ms,
            bge_scoring_latency_ms=inference_latency_ms,
            bge_scoring_cpu_time_ms=cpu_time_ms,
            total_reranker_latency_ms=queue_wait_ms + inference_latency_ms,
            provider_raw_scores={
                chunk.chunk_id: score for chunk, score in zip(candidates, scores, strict=True)
            },
            provider_order=[chunk.chunk_id for chunk in scored[:top_k]],
        )

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Reranked top-k, or the RRF top-k if the cross-encoder is unavailable."""
        return self.rerank_with_diagnostics(query, chunks, top_k=top_k).chunks

    def config(self) -> dict[str, Any]:
        """Configuration block recorded in evaluation reports."""
        settings = get_settings()
        return {
            "enabled": self.enabled,
            "configured_model": self.model_name,
            "fallback_model": self.fallback_model_name,
            "loaded_model": self._loaded_model_name,
            "load_error": self._load_error,
            "device": self.device,
            "requested_device": self.requested_device,
            "candidate_top_k": settings.rerank_candidate_top_k,
            "final_top_k": settings.rerank_top_k,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "max_concurrency": self.max_concurrency,
            "cpu_threads": self.cpu_threads or None,
            "warmup_inference": self.warmup_inference,
            "input_pair": "(original query, candidate chunk content)",
            "score_normalisation": "sigmoid over raw logits, computed in RAGuard",
            "activation_parameter": self._activation_parameter or None,
            "tie_break": "ascending chunk_id",
            "scores_preserved": [
                "sparse_score",
                "dense_score",
                "fusion_score",
                "rerank_score",
                "normalised_rerank_score",
                "retriever_ranks",
            ],
        }


_reranker: Any | None = None
_lock = threading.Lock()


def get_reranker() -> Any:
    """Return the explicit configured provider without auto-routing.

    The import is intentionally local to avoid a module cycle: the provider
    adapter reuses ``CrossEncoderReranker`` as its lazy local implementation.
    """
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from src.reranking.provider import ConfiguredReranker

                _reranker = ConfiguredReranker()
    return _reranker


def reset_reranker() -> None:
    """Drop the cached reranker so new settings take effect."""
    global _reranker
    with _lock:
        _reranker = None


def warmup_reranker_model() -> bool:
    """Load the configured reranker once and record failures for readiness."""
    loaded = get_reranker().warmup()
    if loaded:
        logger.info("Reranker model ready")
    return loaded


def is_reranker_model_loaded() -> bool:
    """Readiness probe that does not itself load model weights."""
    return get_reranker().is_model_loaded


def reranker_model_load_error() -> str | None:
    """Return the recorded load error; callers must sanitise it before output."""
    return get_reranker().load_error


def loaded_reranker_model_name() -> str | None:
    """Return the actual resident model, if warm-up has completed."""
    return get_reranker().loaded_model_name
