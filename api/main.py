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

Models load lazily on first use. Call `/admin/warmup` after start-up to move
that cost off the first user request.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    CitationOut,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    TraceStep,
)
from src.config import Settings, get_settings
from src.generation.llm_factory import LLMProviderError
from src.retrieval.bm25 import get_bm25_index, refresh_bm25_index
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
    try:
        init_schema()
        logger.info("Connected to pgvector, %d chunks indexed", count_chunks())
    except Exception:
        # Never crash on a dependency failure at start-up: /health must stay
        # reachable so the cause is diagnosable rather than a restart loop.
        logger.exception("Database initialisation failed at start-up")
    yield
    close_pool()


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
    allow_headers=["Content-Type", "X-Request-ID"],
)


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


SettingsDep = Annotated[Settings, Depends(get_settings)]
GraphDep = Annotated[SelfHealingGraph, Depends(graph_dependency)]


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
        request.headers.get("X-Request-ID"),
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_request",
        "; ".join(fields)[:500],
    )


@app.exception_handler(LLMProviderError)
async def provider_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    """A misconfigured or unreachable provider is a 503, not a 500."""
    logger.error("Provider unavailable: %s", exc)
    return _error(
        request.headers.get("X-Request-ID"),
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "provider_unavailable",
        "The configured language model provider is unavailable. Check service configuration.",
    )


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. The cause goes to the log; the client gets a reference."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    logger.exception("Unhandled error [request_id=%s]", request_id)
    return _error(
        request_id,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "An unexpected error occurred. Quote the request_id when reporting this.",
    )


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

    checks["retrieval"] = {"strategy": "hybrid_bm25_dense_rrf", "reranker": settings.reranker_model}

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
        failure_reason=summary.get("failure_reason") or None,
        trace=[
            TraceStep(step=i, node=node)
            for i, node in enumerate(summary.get("node_sequence") or [], start=1)
        ],
        latency_ms=round(latency_ms, 1),
        prompt_version=str(state.get("prompt_version") or ""),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, service: GraphDep) -> QueryResponse:
    """Run the self-healing workflow over one question.

    Every stage happens inside the graph. This function supplies a request ID,
    invokes it, and projects the result.
    """
    request_id = request.request_id or str(uuid.uuid4())
    started = time.perf_counter()

    logger.info("Query received [request_id=%s, chars=%d]", request_id, len(request.query))

    try:
        state = service.invoke(request.query, request_id=request_id)
    except LLMProviderError:
        raise
    except Exception as exc:
        logger.exception("Graph execution failed [request_id=%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answering workflow is temporarily unavailable.",
        ) from exc

    from src.self_healing.graph import summarise

    summary = summarise(state)
    latency_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "Query completed [request_id=%s, outcome=%s, retries=%s, latency_ms=%.0f]",
        request_id,
        summary.get("final_outcome"),
        summary.get("retry_count"),
        latency_ms,
    )
    return to_response(state, summary, latency_ms)


# --------------------------------------------------------------------------
# Operational endpoints (retained from the earlier API)
# --------------------------------------------------------------------------


@app.post("/retrieve")
def retrieve(request: QueryRequest, settings: SettingsDep) -> dict[str, Any]:
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
def reindex() -> dict[str, int]:
    """Rebuild the in-memory BM25 index after ingestion."""
    index = refresh_bm25_index()
    return {"bm25_documents": index.size, "chunks_indexed": count_chunks()}


@app.post("/admin/warmup")
def warmup() -> dict[str, str]:
    """Load the embedding and reranker models ahead of the first user request."""
    from src.reranking import get_reranker
    from src.retrieval.embeddings import get_embedding_model
    from src.retrieval.types import RetrievedChunk

    get_embedding_model()
    # A non-empty list is required, otherwise rerank short-circuits and the
    # cross-encoder is never actually loaded.
    dummy = RetrievedChunk(chunk_id=-1, content="warmup", source="warmup", chunk_index=0)
    get_reranker().rerank("warmup", [dummy])
    get_bm25_index()
    return {"status": "warm"}


@app.get("/config")
def config(settings: SettingsDep) -> dict[str, Any]:
    """Non-secret configuration, for reproducing a run. Never includes a key."""
    return {
        "api_version": API_VERSION,
        "llm_provider": settings.llm_provider,
        "generation_model": (
            settings.llm_model
            or (settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model)
        ),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
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
