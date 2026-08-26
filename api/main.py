"""FastAPI service for RAGuard.

A thin transport layer. Retrieval, reranking, evidence grading, rewriting,
generation, and citation verification all live in the LangGraph workflow; this
module turns HTTP into a graph invocation and the resulting state into a
response. No endpoint reimplements a stage, because two implementations of the
retry policy is one too many.

The response exposes the guard rails, not just the answer: the retry count, the
queries the healing loop tried, the verification verdict, and the sequence of
nodes that ran. That decision trail is the contribution of the project, and it
is what the Streamlit UI renders.

Two boundaries are enforced here rather than trusted:

- **Liveness is not readiness.** `/health` answers from the process alone and
  never touches PostgreSQL or a model, so a service with a broken dependency
  stays diagnosable instead of being restarted in a loop. `/ready` is where
  dependencies are checked.
- **Errors are translated, not forwarded.** An exception's text can carry a
  connection string or a prompt, so clients receive a category and a request ID
  while the detail goes to the log.

The embedding and reranker models are warmed in background threads at start-up,
so the first user query does not sit inside a model download. `/ready` stays
503 until the enabled model stack is resident; `/health` never waits on it.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.admission import query_admission
from api.observability import log_event, runtime_metrics
from api.schemas import (
    REQUEST_ID_PATTERN,
    CitationOut,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    TraceStep,
)
from api.tracing import (
    annotate_query_span,
    configure_tracing,
    inject_trace_context,
    shutdown_tracing,
    start_request_span,
)
from src.config import (
    Settings,
    enforce_production_configuration,
    enforce_production_runtime_storage,
    get_settings,
)
from src.generation.llm_factory import LLMProviderError
from src.reranking import (
    is_reranker_model_loaded,
    loaded_reranker_model_name,
    reranker_model_load_error,
    warmup_reranker_model,
)
from src.retrieval.bm25 import get_bm25_index, refresh_bm25_index
from src.retrieval.embeddings import (
    is_model_loaded,
    model_load_error,
    warmup_embedding_model,
)
from src.retrieval.vector_store import close_pool, count_chunks, init_schema
from src.self_healing.graph import SelfHealingGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

API_VERSION = "0.2.0"

#: Excerpt length per citation. Enough to judge the citation, short enough that
#: the endpoint does not become a document-dumping service.
CITATION_EXCERPT_CHARS = 400


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    preflight = enforce_production_configuration(settings)
    if settings.runtime_environment == "production":
        enforce_production_runtime_storage(settings)
        for warning in preflight.warnings:
            logger.warning("Production preflight recommendation: %s", warning.code)
    configure_tracing(settings)
    try:
        init_schema()
        logger.info("Connected to pgvector, %d chunks indexed", count_chunks())
    except Exception:
        # Never crash on a dependency failure at start-up: /health must stay
        # reachable so the cause is diagnosable rather than a restart loop.
        logger.exception("Database initialisation failed at start-up")

    # Load the embedding model off the request path. On a cold cache this
    # downloads ~2.2 GB, which is far too long to sit inside the first user
    # query — that is what made the UI look hung.
    #
    # A background thread rather than a blocking await, for two reasons:
    # /health must answer immediately so the container is not killed by its own
    # health check while warming, and `SentenceTransformer(...)` is synchronous
    # so awaiting it would block the event loop entirely. `/ready` is the gate
    # that stays closed until this finishes.
    for target, name in (
        (warmup_embedding_model, "embedding-warmup"),
        (warmup_reranker_model, "reranker-warmup"),
    ):
        threading.Thread(target=target, name=name, daemon=True).start()

    yield
    close_pool()
    shutdown_tracing()


app = FastAPI(
    title="RAGuard",
    description="Self-healing hybrid RAG with citation verification and abstention",
    version=API_VERSION,
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    # Configurable, and not "*" by default: the browser sends cookies to
    # whatever this allows.
    allow_origins=_settings.cors_allow_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Key", "X-Request-ID"],
    expose_headers=["traceparent"],
)


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """Propagate W3C context while keeping traces free of customer content."""
    with start_request_span(request.method, request.url.path, request.headers) as span:
        try:
            response = await call_next(request)
        except Exception:
            span.set_attribute("http.response.status_code", 500)
            raise
        span.set_attribute("http.response.status_code", response.status_code)
        trace_context = inject_trace_context()
        if traceparent := trace_context.get("traceparent"):
            response.headers["traceparent"] = traceparent
        return response


def graph_dependency() -> SelfHealingGraph:
    return get_graph_service()


_service: SelfHealingGraph | None = None


def get_graph_service() -> SelfHealingGraph:
    """Process-wide graph service. Built once, on first use."""
    global _service
    if _service is None:
        _service = SelfHealingGraph()
    return _service


def reset_graph_service() -> None:
    """Drop the cached service. Used by tests that inject a fake."""
    global _service
    _service = None
    query_admission.reset()
    runtime_metrics.reset()


SettingsDep = Annotated[Settings, Depends(get_settings)]
GraphDep = Annotated[SelfHealingGraph, Depends(graph_dependency)]


def _safe_header_request_id(request: Request) -> str | None:
    """Return only an allow-listed correlation ID from an HTTP header."""
    candidate = (request.headers.get("X-Request-ID") or "").strip()
    return candidate if REQUEST_ID_PATTERN.fullmatch(candidate) else None


def require_admin(request: Request, settings: SettingsDep) -> None:
    """Protect diagnostic and state-changing operational endpoints.

    An empty server-side key disables these endpoints instead of accidentally
    publishing document contents or an expensive reindex/warmup operation.
    """
    supplied = request.headers.get("X-Admin-Key") or ""
    expected = settings.admin_api_key
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        logger.warning("Operational endpoint denied [path=%s]", request.url.path)
        log_event("operational_access_denied", path=request.url.path)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def admit_query(request: Request, settings: SettingsDep) -> Iterator[None]:
    """Bound local model work before the graph starts executing.

    This is intentionally process-local. A distributed deployment must enforce
    the same limit before traffic reaches individual workers.
    """
    peer = request.client.host if request.client else "unknown"
    lease = query_admission.acquire(
        peer,
        max_concurrency=settings.query_max_concurrency,
        requests_per_minute=settings.query_rate_limit_per_minute,
    )
    if lease.reason == "rate_limited":
        runtime_metrics.record_admission_rejected(lease.reason)
        log_event("query_admission_rejected", reason=lease.reason)
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS)
    if lease.reason in {"busy", "admission_backend_unavailable"}:
        runtime_metrics.record_admission_rejected(lease.reason)
        log_event("query_admission_rejected", reason=lease.reason)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    runtime_metrics.record_admitted()
    try:
        yield
    finally:
        query_admission.release(lease)


AdminDep = Annotated[None, Depends(require_admin)]
AdmissionDep = Annotated[None, Depends(admit_query)]


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


def _error(
    request_id: str | None, code: int, error: str, detail: str
) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(error=error, detail=detail, request_id=request_id).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with the offending field, but never the raw exception text."""
    fields = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        fields.append(f"{location or 'body'}: {err.get('msg', 'invalid')}")
    logger.info("Rejected malformed request: %s", fields)
    return _error(
        _safe_header_request_id(request),
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_request",
        "; ".join(fields)[:500],
    )


@app.exception_handler(LLMProviderError)
async def provider_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    """A misconfigured or unreachable provider is a 503, not a 500."""
    logger.error("Provider unavailable: %s", exc)
    return _error(
        _safe_header_request_id(request),
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "provider_unavailable",
        "The configured language model provider is unavailable. Check service configuration.",
    )


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. The cause goes to the log; the client gets a reference."""
    request_id = _safe_header_request_id(request) or str(uuid.uuid4())
    logger.exception("Unhandled error [request_id=%s]", request_id)
    return _error(
        request_id,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred. Quote the request_id when reporting this.",
    )


@app.exception_handler(HTTPException)
async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep expected HTTP failures on the same non-leaking response contract."""
    request_id = _safe_header_request_id(request)
    if exc.status_code == status.HTTP_403_FORBIDDEN:
        return _error(request_id, exc.status_code, "forbidden", "This endpoint requires access.")
    if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return _error(
            request_id,
            exc.status_code,
            "rate_limited",
            "Too many requests. Retry shortly.",
        )
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return _error(
            request_id,
            exc.status_code,
            "service_busy",
            "The service is temporarily busy. Retry shortly.",
        )
    return _error(request_id, exc.status_code, "request_failed", "The request could not be completed.")


# --------------------------------------------------------------------------
# Health and readiness
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only.

    Deliberately touches no dependency: an orchestrator restarting the API
    because PostgreSQL is down fixes nothing and destroys the evidence.
    """
    return HealthResponse(status="ok", service="raguard", version=API_VERSION)


@app.get("/ready", response_model=ReadyResponse)
def ready(settings: SettingsDep, response: JSONResponse = None) -> Any:  # noqa: ARG001
    """Readiness: can this process actually serve a query?

    Checks the database, the ingested corpus, and provider configuration. It
    does not load BGE-M3 or the cross-encoder — a readiness probe that pulls
    2 GB of weights is a readiness probe that times out.
    """
    checks: dict[str, Any] = {}
    problems: list[str] = []

    try:
        chunk_count = count_chunks()
        checks["database"] = {"status": "ok", "chunks_indexed": chunk_count}
        if chunk_count == 0:
            problems.append("no chunks indexed; run: python -m src.ingestion.ingest")
            checks["database"]["status"] = "empty"
    except Exception as exc:
        logger.warning("Readiness: database check failed: %s", type(exc).__name__)
        checks["database"] = {"status": "unavailable", "error": type(exc).__name__}
        problems.append("database unavailable")

    if settings.llm_provider == "gemini" and not settings.google_api_key:
        checks["llm_provider"] = {"status": "unconfigured", "provider": "gemini"}
        problems.append("GOOGLE_API_KEY is not set")
    else:
        checks["llm_provider"] = {"status": "configured", "provider": settings.llm_provider}

    # The embedding model is a hard requirement for /query: without it there is
    # no dense arm and no query vector. It used to be absent from this probe
    # entirely, so /ready answered 200 while the first query blocked for the
    # whole model download.
    if is_model_loaded():
        checks["embedding_model"] = {
            "status": "loaded",
            "model": settings.embedding_model,
            "device": settings.model_device,
        }
    else:
        error = model_load_error()
        checks["embedding_model"] = {
            "status": "failed" if error else "loading",
            "model": settings.embedding_model,
            "device": settings.model_device,
            # Type name only. The message can carry a cache path.
            "error": error.split(":")[0] if error else None,
        }
        problems.append(
            "embedding model failed to load"
            if error
            else "embedding model still loading (first start downloads ~2.2 GB)"
        )

    if not settings.reranker_enabled:
        checks["reranker_model"] = {
            "status": "disabled",
            "model": settings.resolved_reranker_model,
            "device": settings.resolved_reranker_device,
        }
    elif is_reranker_model_loaded():
        checks["reranker_model"] = {
            "status": "loaded",
            "model": loaded_reranker_model_name() or settings.resolved_reranker_model,
            "device": settings.resolved_reranker_device,
        }
    else:
        error = reranker_model_load_error()
        checks["reranker_model"] = {
            "status": "failed" if error else "loading",
            "model": settings.resolved_reranker_model,
            "device": settings.resolved_reranker_device,
            "error": error.split(":")[0] if error else None,
        }
        problems.append(
            "reranker model failed to load" if error else "reranker model still loading"
        )

    checks["retrieval"] = {
        "strategy": "hybrid_bm25_dense_rrf",
        "reranker": settings.resolved_reranker_model,
    }

    if problems:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadyResponse(
                status="not_ready", checks=checks, detail="; ".join(problems)
            ).model_dump(),
        )
    return ReadyResponse(status="ready", checks=checks)


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------


def _citations_from(state: dict[str, Any]) -> list[CitationOut]:
    """Build citations from the chunks the graph actually retrieved.

    Metadata comes from the retrieved chunk, never from generated text, which
    is the same rule Phase E enforces one layer down.
    """
    labels = list(state.get("citations") or [])
    by_label = {c.citation_label: c for c in (state.get("retrieved_chunks") or [])}

    out: list[CitationOut] = []
    for label in labels:
        chunk = by_label.get(label)
        if chunk is None:
            # Verification rejects these upstream; skip rather than invent.
            continue
        out.append(
            CitationOut(
                citation_label=label,
                policy_id=chunk.policy_id,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                chunk_id=chunk.chunk_id,
                excerpt=chunk.content[:CITATION_EXCERPT_CHARS],
            )
        )
    return out


def _verification_status(verification: dict[str, Any]) -> str:
    if not verification or not verification.get("checked"):
        return "not_checked"
    return "supported" if verification.get("supported") else "unsupported"


def _public_failure_reason(summary: dict[str, Any]) -> str | None:
    """Never project provider, database, or exception text into the API."""
    outcome = summary.get("final_outcome")
    if outcome == "clarify":
        return "ambiguous_request"
    if outcome == "escalate":
        return "risk_escalation"

    reason = str(summary.get("failure_reason") or "")
    allowed = {
        "retrieval_failed",
        "provider_error",
        "invalid_output",
        "rejected_insufficient_context",
        "rejected_invalid_citation",
        "rejected_empty_answer",
        "answer_not_grounded",
    }
    return reason if reason in allowed else ("request_not_completed" if reason else None)


def to_response(
    state: dict[str, Any], summary: dict[str, Any], latency_ms: float
) -> QueryResponse:
    """Project graph state onto the public contract, field by field."""
    verification = summary.get("verification_result") or {}
    grade = summary.get("evidence_grade") or {}

    return QueryResponse(
        request_id=str(summary.get("request_id") or ""),
        outcome=summary.get("final_outcome") or "error",
        answer=summary.get("final_answer") or "",
        citations=_citations_from(state),
        confidence=float(summary.get("answer_confidence") or 0.0),
        more_info_required=summary.get("final_outcome") != "answer",
        retry_count=int(summary.get("retry_count") or 0),
        max_retries=int(summary.get("max_retries") or 0),
        rewritten_queries=list(summary.get("rewritten_queries") or []),
        risk_level=str(summary.get("risk_level") or "none"),
        verification_status=_verification_status(verification),  # type: ignore[arg-type]
        verified_claim_count=int(verification.get("supported_claim_count") or 0),
        unsupported_claim_count=int(verification.get("unsupported_claim_count") or 0),
        evidence_sufficient=grade.get("sufficient"),
        retrieved_chunk_count=int(summary.get("retrieved_chunk_count") or 0),
        reranker_used=summary.get("reranker_used"),
        failure_reason=_public_failure_reason(summary),
        trace=[
            TraceStep(step=i, node=node)
            for i, node in enumerate(summary.get("node_sequence") or [], start=1)
        ],
        latency_ms=round(latency_ms, 1),
        prompt_version=str(state.get("prompt_version") or ""),
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, service: GraphDep, _admission: AdmissionDep
) -> QueryResponse | JSONResponse:
    """Run the self-healing workflow over one question.

    Every stage happens inside the graph. This function supplies a request ID,
    invokes it, and projects the result.
    """
    request_id = request.request_id or str(uuid.uuid4())
    started = time.perf_counter()

    # A caller can bypass the Streamlit readiness check and call this endpoint
    # directly. Never put that request on the multi-gigabyte model download;
    # the startup warmup owns initialization and the caller can retry once
    # `/ready` reports success.
    if not is_model_loaded():
        failed = model_load_error() is not None
        logger.warning(
            "Query rejected before model readiness [request_id=%s, status=%s]",
            request_id,
            "failed" if failed else "loading",
        )
        failure_code = "model_unavailable" if failed else "service_not_ready"
        runtime_metrics.record_failure(failure_code)
        annotate_query_span(request_id=request_id, failure_reason=failure_code)
        log_event("query_rejected", failure_reason=failure_code)
        return _error(
            request_id,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            failure_code,
            (
                "The embedding model failed to initialize. Check the API logs."
                if failed
                else "The service is still initializing the embedding model. Retry shortly."
            ),
        )

    logger.info("Query received [request_id=%s, chars=%d]", request_id, len(request.query))

    try:
        state = service.invoke(request.query, request_id=request_id)
    except LLMProviderError:
        runtime_metrics.record_failure("provider_unavailable")
        annotate_query_span(request_id=request_id, failure_reason="provider_unavailable")
        log_event("query_failed", failure_reason="provider_unavailable")
        raise
    except Exception as exc:
        logger.exception("Graph execution failed [request_id=%s]", request_id)
        runtime_metrics.record_failure("workflow_unavailable")
        annotate_query_span(request_id=request_id, failure_reason="workflow_unavailable")
        log_event("query_failed", failure_reason="workflow_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answering workflow is temporarily unavailable.",
        ) from exc

    from src.self_healing.graph import summarise

    summary = summarise(state)
    latency_ms = (time.perf_counter() - started) * 1000.0

    response = to_response(state, summary, latency_ms)
    runtime_metrics.record_completed(
        outcome=response.outcome,
        latency_ms=latency_ms,
        retrieved_chunk_count=response.retrieved_chunk_count,
        evidence_sufficient=response.evidence_sufficient,
        verification_status=response.verification_status,
        reranker_used=response.reranker_used,
    )
    annotate_query_span(request_id=request_id, outcome=response.outcome)
    log_event(
        "query_completed",
        outcome=response.outcome,
        retry_count=response.retry_count,
        retrieved_chunk_count=response.retrieved_chunk_count,
        evidence_sufficient=response.evidence_sufficient,
        verification_status=response.verification_status,
        reranker_used=response.reranker_used,
        latency_ms=round(latency_ms, 1),
    )
    return response


# --------------------------------------------------------------------------
# Operational endpoints (retained from the earlier API)
# --------------------------------------------------------------------------


@app.post("/retrieve")
def retrieve(
    request: QueryRequest, settings: SettingsDep, _admin: AdminDep
) -> dict[str, Any]:
    """Retrieval and reranking only, with no LLM call.

    The endpoint to use when deciding whether a failure is retrieval or
    generation.
    """
    from src.reranking import get_reranker
    from src.retrieval.hybrid import get_hybrid_retriever

    candidates = get_hybrid_retriever().retrieve(request.query)
    result = get_reranker().rerank_with_diagnostics(
        request.query, candidates, top_k=settings.rerank_top_k
    )
    return {
        "query": request.query,
        "reranker_used": result.reranker_used,
        "chunks": [c.to_dict() for c in result.chunks],
    }


@app.post("/admin/reindex")
def reindex(_admin: AdminDep) -> dict[str, int]:
    """Rebuild the in-memory BM25 index after ingestion."""
    index = refresh_bm25_index()
    return {"bm25_documents": index.size, "chunks_indexed": count_chunks()}


@app.post("/admin/warmup")
def warmup(_admin: AdminDep, settings: SettingsDep) -> dict[str, str]:
    """Load the embedding and reranker models ahead of the first user request."""
    embedding_ready = warmup_embedding_model()
    reranker_ready = not settings.reranker_enabled or warmup_reranker_model()
    get_bm25_index()
    if not embedding_ready or not reranker_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model warm-up failed; inspect server logs and /ready",
        )
    return {"status": "warm"}


@app.get("/admin/metrics")
def metrics(_admin: AdminDep) -> dict[str, Any]:
    """Process-local operational aggregates with no customer or corpus data."""
    return runtime_metrics.snapshot()


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(_admin: AdminDep) -> Response:
    """Prometheus scrape output, protected by the operational API key.

    In production also restrict this path at the ingress/network layer; the
    key is defense in depth and keeps the endpoint closed by default.
    """
    return Response(
        content=runtime_metrics.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/config")
def config(settings: SettingsDep) -> dict[str, Any]:
    """Non-secret configuration, for reproducing a run. Never includes a key."""
    return {
        "api_version": API_VERSION,
        "runtime_environment": settings.runtime_environment,
        "runtime_profile": settings.runtime_profile,
        "llm_provider": settings.llm_provider,
        "generation_model": (
            settings.llm_model
            or (settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model)
        ),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.resolved_reranker_model,
        "configured_full_reranker_model": settings.reranker_model,
        "reranker_device": settings.resolved_reranker_device,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "rrf_k": settings.rrf_k,
        "evidence_top_score_threshold": settings.evidence_top_score_threshold,
        "evidence_min_relevant_chunks": settings.evidence_min_relevant_chunks,
        "evidence_confidence_threshold": settings.evidence_confidence_threshold,
        "graph_max_retries": settings.graph_max_retries,
        "graph_max_regenerations": settings.graph_max_regenerations,
        "verifier_backend": settings.verifier_backend,
    }
