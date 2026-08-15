"""FastAPI service for RAGuard.

The API exposes the guard rails, not just the answer. `/query` returns the
confidence scores, the rewritten queries, the citation report, and the full
decision trace, because the contribution of this project is the *decision*, not
the text. The Streamlit UI renders that trace directly.

Models are loaded lazily on first request. A cold first call takes tens of
seconds while BGE-M3 and the reranker load; call `/admin/warmup` after start-up
to move that cost off the first user request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.retrieval.bm25 import get_bm25_index, refresh_bm25_index
from src.retrieval.vector_store import close_pool, count_chunks, init_schema
from src.self_healing.pipeline import SelfHealingRAG, get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_schema()
        logger.info("Connected to pgvector, %d chunks indexed", count_chunks())
    except Exception:
        # Do not crash the service: /health must stay reachable so the failure
        # is diagnosable rather than a container restart loop.
        logger.exception("Database initialisation failed at start-up")
    yield
    close_pool()


app = FastAPI(
    title="RAGuard",
    description="Self-healing hybrid RAG with citation verification and abstention",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    use_llm_rewrite: bool = True
    use_llm_verification: bool = Field(
        default=False,
        description="Adds an LLM entailment check per claim. Slower and non-deterministic.",
    )


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int | None = None
    bm25_documents: int | None = None
    llm_provider: str | None = None
    detail: str | None = None


def pipeline_dependency() -> SelfHealingRAG:
    return get_pipeline()


# Annotated dependencies rather than default arguments: the default-argument
# form trips B008 and hides the dependency from type checkers.
SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[SelfHealingRAG, Depends(pipeline_dependency)]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    """Liveness. Never raises, so an unhealthy service is still inspectable."""
    try:
        return HealthResponse(
            status="ok",
            chunks_indexed=count_chunks(),
            llm_provider=settings.llm_provider,
        )
    except Exception as exc:
        return HealthResponse(status="degraded", detail=str(exc))


@app.get("/ready", response_model=HealthResponse)
def ready(settings: SettingsDep) -> HealthResponse:
    """Readiness. Fails when the corpus has not been ingested."""
    chunks = count_chunks()
    if chunks == 0:
        raise HTTPException(
            status_code=503,
            detail="No chunks indexed. Run: python -m src.ingestion.ingest",
        )
    return HealthResponse(
        status="ready",
        chunks_indexed=chunks,
        bm25_documents=get_bm25_index().size,
        llm_provider=settings.llm_provider,
    )


@app.post("/query")
def query(request: QueryRequest, pipeline: PipelineDep) -> dict[str, Any]:
    """Full self-healing pipeline: retrieve, heal, generate, verify, or abstain."""
    try:
        response = pipeline.answer(
            request.question,
            use_llm_rewrite=request.use_llm_rewrite,
            use_llm_verification=request.use_llm_verification,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return response.to_dict()


@app.post("/retrieve")
def retrieve(request: RetrieveRequest, pipeline: PipelineDep) -> dict[str, Any]:
    """Retrieval and reranking only, with no LLM call.

    This is the endpoint the deterministic evaluation tier exercises, and the
    one to use when debugging whether a failure is retrieval or generation.
    """
    chunks, confidence = pipeline.retrieve_only(request.question)
    return {
        "question": request.question,
        "confidence": confidence.to_dict(),
        "chunks": [c.to_dict() for c in chunks[: request.top_k]],
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
    """Non-secret configuration, for reproducing an evaluation run."""
    return {
        "llm_provider": settings.llm_provider,
        "generation_model": (
            settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model
        ),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "rrf_k": settings.rrf_k,
        "retrieval_confidence_threshold": settings.retrieval_confidence_threshold,
        "abstain_threshold": settings.abstain_threshold,
        "max_healing_attempts": settings.max_healing_attempts,
    }
