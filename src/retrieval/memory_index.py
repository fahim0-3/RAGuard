"""In-memory dense retrieval over a small corpus, kept fresh against pgvector.

pgvector remains the source of truth: ingestion writes there and nothing else.
For a corpus this size (22 chunks, 1024 dimensions, under 100 KB of vectors) a
query against the remote database is almost entirely network: measured at a
0.26 ms execution inside a 230-830 ms round trip. Holding the vectors in the
API process removes that round trip from every query.

Semantics match the pgvector query exactly. pgvector returns
``1 - (embedding <=> query)``, which for the cosine operator is cosine
similarity, ordered by distance and limited to ``top_k``. The same value is
computed here from unit-normalised vectors, with ties broken by the database's
fetch order so the result is deterministic.

Freshness. Ingestion runs in its own process, so the API cannot be told when
the table changes. A background thread compares a cheap fingerprint of the
table (row count, largest id, and digests of content, metadata and vectors)
every ``corpus_refresh_interval_s`` and rebuilds when it differs. BM25 is rebuilt
from the same snapshot at the same moment, so the sparse and dense halves of a
hybrid query never describe different corpora. ``POST /admin/reindex`` rebuilds
immediately.

Above ``MAX_IN_MEMORY_CHUNKS`` the index is not built and every query uses
pgvector, which is what an index server is for.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_IN_MEMORY_CHUNKS",
    "InMemoryDenseIndex",
    "dense_search",
    "is_memory_index_built",
    "memory_index_status",
    "refresh_corpus_indexes",
    "start_corpus_refresher",
    "stop_corpus_refresher",
    "warmup_corpus_indexes",
]

#: Beyond this, a full copy of the vectors per API process stops being cheap.
MAX_IN_MEMORY_CHUNKS = 50_000


class InMemoryDenseIndex:
    """Exact cosine search over unit-normalised vectors."""

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk],
        embeddings: np.ndarray,
        fingerprint: str = "",
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError(f"{len(chunks)} chunks but embeddings of shape {tuple(matrix.shape)}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # A zero vector has no direction; it scores 0 against everything,
        # instead of dividing by zero.
        self._matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self._chunks = list(chunks)
        self.fingerprint = fingerprint

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if not self._chunks or top_k <= 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        similarity = self._matrix @ (query / norm)
        # Stable, so equal scores keep the database's fetch order.
        order = np.argsort(-similarity, kind="stable")[:top_k]
        return [replace(self._chunks[i], dense_score=float(similarity[i])) for i in order]


_lock = threading.Lock()
_index: InMemoryDenseIndex | None = None
_refresher: threading.Thread | None = None
_stop = threading.Event()


def _memory_backend_enabled() -> bool:
    return getattr(get_settings(), "dense_index_backend", "memory") == "memory"


def is_memory_index_built() -> bool:
    return _index is not None


def memory_index_status() -> dict[str, Any]:
    """Operational facts for readiness; no corpus content."""
    index = _index
    return {
        "backend": "memory" if _memory_backend_enabled() else "pgvector",
        "built": index is not None,
        "chunks": index.size if index is not None else 0,
    }


def dense_search(query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
    """Dense retrieval: in memory when the index is built, otherwise pgvector.

    The pgvector path is kept as the fallback, not removed: a corpus above the
    in-memory limit, a disabled memory backend, or a query that arrives before
    the first build all still get a correct answer from the database.
    """
    index = _index
    if index is not None and _memory_backend_enabled():
        return index.search(query_embedding, top_k)
    from src.retrieval.vector_store import dense_search as pgvector_dense_search

    return pgvector_dense_search(query_embedding, top_k)


def refresh_corpus_indexes() -> dict[str, int]:
    """Rebuild BM25 and the dense index from one consistent database snapshot.

    The fingerprint is taken before the fetch. If ingestion lands between the
    two, the index holds the newer rows under the older fingerprint, and the
    next refresher tick sees a difference and rebuilds again: at worst one
    extra rebuild, never a stale index that believes itself current.
    """
    global _index
    from src.retrieval.bm25 import install_bm25_index
    from src.retrieval.vector_store import corpus_fingerprint, fetch_corpus_with_embeddings

    fingerprint = corpus_fingerprint()
    chunks, embeddings = fetch_corpus_with_embeddings()
    install_bm25_index(chunks)

    if not _memory_backend_enabled():
        with _lock:
            _index = None
        return {"bm25_documents": len(chunks), "dense_in_memory": 0}
    if len(chunks) > MAX_IN_MEMORY_CHUNKS:
        logger.info(
            "Corpus of %d chunks exceeds the in-memory limit; dense search stays on pgvector",
            len(chunks),
        )
        with _lock:
            _index = None
        return {"bm25_documents": len(chunks), "dense_in_memory": 0}

    index = InMemoryDenseIndex(chunks, embeddings, fingerprint)
    with _lock:
        _index = index
    logger.info("In-memory dense index built over %d chunks", index.size)
    return {"bm25_documents": len(chunks), "dense_in_memory": index.size}


def warmup_corpus_indexes() -> bool:
    """Start-up build, returning success instead of raising."""
    try:
        refresh_corpus_indexes()
    except Exception:
        logger.exception("Corpus index warmup failed")
        return False
    return True


def _refresh_if_changed() -> bool:
    from src.retrieval.vector_store import corpus_fingerprint

    current = _index.fingerprint if _index is not None else None
    if current is not None and corpus_fingerprint() == current:
        return False
    refresh_corpus_indexes()
    return True


def _refresher_loop(interval_s: float) -> None:
    while not _stop.wait(interval_s):
        try:
            if _refresh_if_changed():
                logger.info("Corpus changed; retrieval indexes rebuilt")
        except Exception:
            # A database blip must not kill the thread; the next tick retries,
            # and queries keep using the last good index meanwhile.
            logger.warning("Corpus freshness check failed; keeping the current index")


def start_corpus_refresher() -> None:
    """Begin polling for corpus changes. No-op when disabled or already running."""
    global _refresher
    interval = float(getattr(get_settings(), "corpus_refresh_interval_s", 0.0))
    if interval <= 0 or not _memory_backend_enabled():
        return
    with _lock:
        if _refresher is not None and _refresher.is_alive():
            return
        _stop.clear()
        _refresher = threading.Thread(
            target=_refresher_loop, args=(interval,), name="corpus-refresher", daemon=True
        )
        _refresher.start()


def stop_corpus_refresher() -> None:
    _stop.set()


def reset_memory_index() -> None:
    """Drop the index. Used by tests."""
    global _index
    with _lock:
        _index = None
