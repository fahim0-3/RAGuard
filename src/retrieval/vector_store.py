"""pgvector-backed chunk store.

Owns the schema, the connection pool, and dense similarity search. Ingestion
and retrieval both go through this module so the table definition exists in
exactly one place.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                settings = get_settings()
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1,
                    max_size=8,
                    timeout=settings.db_pool_timeout_s,
                    reconnect_timeout=settings.db_reconnect_timeout_s,
                    kwargs={"connect_timeout": settings.db_connect_timeout_s},
                    configure=_configure,
                    # Managed databases such as Neon can close an idle
                    # connection while a local embedding model is loading.
                    # Validate on checkout so the pool replaces a dead socket
                    # before application SQL sees it.
                    check=ConnectionPool.check_connection,
                    open=True,
                )
    return _pool


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema() -> None:
    """Create the extension, table, and indexes. Safe to run repeatedly."""
    settings = get_settings()
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
        id           BIGSERIAL PRIMARY KEY,
        source       TEXT        NOT NULL,
        doc_id       TEXT        NOT NULL,
        chunk_index  INTEGER     NOT NULL,
        content      TEXT        NOT NULL,
        metadata     JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
        embedding    VECTOR({settings.vector_dimension}) NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (source, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
        ON {CHUNKS_TABLE} USING hnsw (embedding vector_cosine_ops);

    CREATE INDEX IF NOT EXISTS chunks_source_idx ON {CHUNKS_TABLE} (source);
    """
    with get_connection() as conn:
        conn.execute(ddl)
        conn.commit()
    logger.info("Schema ready (vector dimension=%s)", settings.vector_dimension)


def clear_source(source: str) -> int:
    """Delete every chunk for one document. Used for idempotent re-ingestion."""
    with get_connection() as conn:
        cur = conn.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE source = %s", (source,))
        conn.commit()
        return cur.rowcount


def insert_chunks(records: Sequence[dict[str, Any]]) -> int:
    """Insert chunk records. Each record needs source, doc_id, chunk_index,
    content, metadata, embedding."""
    if not records:
        return 0
    rows = [
        (
            r["source"],
            r["doc_id"],
            r["chunk_index"],
            r["content"],
            json.dumps(r.get("metadata", {})),
            r["embedding"],
        )
        for r in records
    ]
    sql = f"""
        INSERT INTO {CHUNKS_TABLE} (source, doc_id, chunk_index, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (source, chunk_index) DO UPDATE
        SET content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def count_chunks() -> int:
    with get_connection() as conn:
        row = conn.execute(f"SELECT count(*) FROM {CHUNKS_TABLE}").fetchone()
    return int(row[0]) if row else 0


def fetch_all_chunks() -> list[RetrievedChunk]:
    """Load the whole corpus. Used to build the in-memory BM25 index."""
    sql = f"""
        SELECT id, content, source, doc_id, chunk_index, metadata
        FROM {CHUNKS_TABLE}
        ORDER BY source, chunk_index
    """
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql).fetchall()
    return [
        RetrievedChunk(
            chunk_id=row["id"],
            content=row["content"],
            source=row["source"],
            doc_id=row["doc_id"] or "",
            chunk_index=row["chunk_index"],
            metadata=row["metadata"] or {},
        )
        for row in rows
    ]


def source_policy_ids() -> dict[str, str]:
    """Map each source filename to its document identifier, for example REF-001."""
    sql = f"SELECT DISTINCT source, doc_id FROM {CHUNKS_TABLE} ORDER BY source"
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql).fetchall()
    return {row["source"]: row["doc_id"] or "" for row in rows}


def dense_search(query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
    """Cosine nearest neighbours. Returns similarity in [0, 1] as `dense_score`."""
    import numpy as np

    sql = f"""
        SELECT id, content, source, doc_id, chunk_index, metadata,
               1 - (embedding <=> %s) AS similarity
        FROM {CHUNKS_TABLE}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    vector = np.asarray(query_embedding, dtype=np.float32)
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql, (vector, vector, top_k)).fetchall()
    return [
        RetrievedChunk(
            chunk_id=row["id"],
            content=row["content"],
            source=row["source"],
            doc_id=row["doc_id"] or "",
            chunk_index=row["chunk_index"],
            metadata=row["metadata"] or {},
            dense_score=float(row["similarity"]),
        )
        for row in rows
    ]
