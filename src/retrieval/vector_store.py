"""pgvector-backed chunk store.

Owns the schema, the connection pool, and dense similarity search. Ingestion
and retrieval both go through this module so the table definition exists in
exactly one place.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

CHUNKS_TABLE = "chunks"

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()

#: When each pooled connection was last handed back in a clean state. Weak
#: keys, so a connection the pool discards takes its entry with it.
_returned_at: weakref.WeakKeyDictionary[psycopg.Connection, float] = weakref.WeakKeyDictionary()


def _mark_returned(conn: psycopg.Connection) -> None:
    """Pool `reset` hook: record that this connection just worked."""
    _returned_at[conn] = time.monotonic()


def _check_if_idle(conn: psycopg.Connection) -> None:
    """Validate a connection on checkout only if it may have gone stale.

    A connection that completed a statement and came back seconds ago is
    known good, and validating it costs a full network round trip. One
    that has sat idle, or has never been returned, is checked as before.
    Raising here makes the pool discard the connection and open another.
    """
    window = float(get_settings().db_checkout_validation_idle_s)
    returned = _returned_at.get(conn)
    if window > 0 and returned is not None and time.monotonic() - returned < window:
        return
    ConnectionPool.check_connection(conn)


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    # This applies to every pooled runtime connection. Schema work uses the
    # separate admin connection, so application credentials need no DDL rights.
    statement_timeout_ms = int(get_settings().db_statement_timeout_s * 1_000)
    conn.execute(f"SET statement_timeout TO {statement_timeout_ms}")


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                settings = get_settings()
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=getattr(settings, "db_pool_min_size", 1),
                    max_size=getattr(settings, "db_pool_max_size", 8),
                    timeout=settings.db_pool_timeout_s,
                    reconnect_timeout=settings.db_reconnect_timeout_s,
                    # Read paths dominate runtime traffic. Autocommit avoids a
                    # separate COMMIT round trip when returning a read-only
                    # connection to a remote managed database. Mutating paths
                    # below use explicit ``conn.transaction()`` blocks.
                    kwargs={
                        "connect_timeout": settings.db_connect_timeout_s,
                        "autocommit": True,
                        # A half-open socket (the server vanished without a
                        # FIN) otherwise blocks a read until the OS gives up,
                        # which can be hours; the server-side statement
                        # timeout cannot fire on a server that is gone.
                        # Keepalive probes turn that into an error in about a
                        # minute, well inside the request budget.
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 3,
                    },
                    configure=_configure,
                    # Managed databases such as Neon can close an idle
                    # connection while a local embedding model is loading.
                    # Validate on checkout so the pool replaces a dead socket
                    # before application SQL sees it — but only when the
                    # connection has been idle long enough to be at risk.
                    check=_check_if_idle,
                    reset=_mark_returned,
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


def enable_vector_extension() -> None:
    """Create pgvector before opening connections that register its types.

    ``ConnectionPool.configure`` calls :func:`register_vector`, which queries
    PostgreSQL's type catalog. On a fresh database that callback cannot succeed
    until the extension exists, so bootstrap must use an unconfigured direct
    connection. ``DATABASE_ADMIN_URL`` can provide that connection separately
    from the runtime pool; the normal direct ``DATABASE_URL`` is the fallback.
    """
    settings = get_settings()
    with psycopg.connect(
        settings.schema_database_url,
        connect_timeout=settings.db_connect_timeout_s,
    ) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()


def init_schema() -> None:
    """Create the extension, table, and indexes. Safe to run repeatedly."""
    settings = get_settings()
    enable_vector_extension()
    # A managed database may accept a direct bootstrap connection before the
    # asynchronous runtime pool has opened its first connection.  Startup can
    # wait through its reconnect window; normal requests still use the shorter
    # `db_pool_timeout_s` checkout bound in `get_connection()`.
    get_pool().wait(timeout=settings.db_reconnect_timeout_s)
    ddl = f"""
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
    with get_connection() as conn, conn.transaction():
        conn.execute(ddl)
    logger.info("Schema ready (vector dimension=%s)", settings.vector_dimension)


def clear_source(source: str) -> int:
    """Delete every chunk for one document. Used for idempotent re-ingestion."""
    with get_connection() as conn:
        cur = conn.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE source = %s", (source,))
        return cur.rowcount


def _chunk_rows(records: Sequence[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
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


def _upsert_chunks(cur: psycopg.Cursor, records: Sequence[dict[str, Any]]) -> int:
    rows = _chunk_rows(records)
    sql = f"""
        INSERT INTO {CHUNKS_TABLE} (source, doc_id, chunk_index, content, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (source, chunk_index) DO UPDATE
        SET doc_id = EXCLUDED.doc_id,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding
    """
    cur.executemany(sql, rows)
    return len(rows)


def insert_chunks(records: Sequence[dict[str, Any]]) -> int:
    """Insert chunk records in one transaction."""
    if not records:
        return 0
    with get_connection() as conn, conn.transaction(), conn.cursor() as cur:
        written = _upsert_chunks(cur, records)
    return written


def replace_source_chunks(source: str, records: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Atomically replace every stored chunk for one source.

    The delete and upsert share a transaction. If serialization, constraint,
    or database execution fails, the pooled connection context rolls back and
    preserves the previously valid source instead of leaving it empty.
    """
    if not records:
        raise ValueError("Source replacement requires at least one chunk")
    if any(record.get("source") != source for record in records):
        raise ValueError("Every replacement chunk must match the source")

    with get_connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE source = %s", (source,))
        removed = cur.rowcount
        written = _upsert_chunks(cur, records)
    return removed, written


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


def fetch_corpus_with_embeddings() -> tuple[list[RetrievedChunk], np.ndarray]:
    """The whole corpus with its vectors, in `fetch_all_chunks` order.

    One snapshot feeds both in-memory indexes, so BM25 and dense search
    always describe the same rows and share the same chunk objects.
    """
    import numpy as np

    sql = f"""
        SELECT id, content, source, doc_id, chunk_index, metadata, embedding
        FROM {CHUNKS_TABLE}
        ORDER BY source, chunk_index
    """
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql).fetchall()
    chunks = [
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
    dimension = get_settings().vector_dimension
    embeddings = (
        np.vstack([_as_float32(row["embedding"]) for row in rows])
        if rows
        else np.zeros((0, dimension), dtype=np.float32)
    )
    return chunks, embeddings


def _as_float32(value: Any) -> np.ndarray:
    """pgvector's adapter yields `Vector` objects, not arrays; accept either."""
    import numpy as np

    to_numpy = getattr(value, "to_numpy", None)
    array = to_numpy() if callable(to_numpy) else value
    return np.asarray(array, dtype=np.float32)


def corpus_fingerprint() -> str:
    """A cheap signature that changes whenever any retrievable row changes.

    The table has no update timestamp, and an upsert rewrites content and
    vectors in place, so the signature digests the data itself: content,
    metadata, document id and vector of every row, plus count and max id.
    One small result row, so the freshness poll stays off the query path.
    """
    sql = f"""
        SELECT count(*) AS n,
               coalesce(max(id), 0) AS max_id,
               md5(coalesce(string_agg(
                   id::text || ':' || doc_id || ':' || md5(content) || ':'
                   || md5(metadata::text) || ':' || md5(embedding::text),
                   ',' ORDER BY id), '')) AS digest
        FROM {CHUNKS_TABLE}
    """
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(sql).fetchone()
    return f"{row['n']}:{row['max_id']}:{row['digest']}"


def source_policy_ids() -> dict[str, str]:
    """Map each source filename to its document identifier, for example REF-001."""
    sql = f"SELECT DISTINCT source, doc_id FROM {CHUNKS_TABLE} ORDER BY source"
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql).fetchall()
    return {row["source"]: row["doc_id"] or "" for row in rows}


def _read_retrying_stale_connection(statement: Callable[[psycopg.Connection], Any]) -> Any:
    """Run a read-only statement, retrying once if the connection died idle.

    Checkout validation is skipped for recently used connections, so a
    socket the database closed in between reaches the statement instead.
    The pool discards a broken connection when it comes back, and the read
    is idempotent, so it is retried once on another connection.

    Two failures are deliberately not retried. A statement timeout is a slow
    query, and repeating it only doubles the wait. A pool timeout means no
    connection could be had at all, typically a database outage; retrying
    it, or validating the pool against an unreachable server, is how a
    request once spent over half an hour in retrieval.
    """
    try:
        with get_connection() as conn:
            return statement(conn)
    except (psycopg.errors.QueryCanceled, PoolTimeout):
        raise
    except psycopg.OperationalError:
        logger.warning("Pooled connection failed; retrying the read once")
        with get_connection() as conn:
            return statement(conn)


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

    def statement(conn: psycopg.Connection) -> list[dict[str, Any]]:
        with conn.cursor(row_factory=dict_row) as cur:
            return cur.execute(sql, (vector, vector, top_k)).fetchall()

    rows = _read_retrying_stale_connection(statement)
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
