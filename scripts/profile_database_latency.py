"""Profile managed pgvector latency without exposing data or credentials.

The probe is deliberately read-only.  It separates connection-pool checkout
(including the configured health check), a trivial network round trip, and the
actual nearest-neighbour statement.  Only aggregate timings and query-plan
metadata are printed; connection strings, vectors, and chunk contents are not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.vector_store import (  # noqa: E402
    CHUNKS_TABLE,
    close_pool,
    dense_search,
    get_connection,
    get_pool,
)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000


def _print_summary(name: str, samples: list[float]) -> None:
    print(
        f"{name}: mean={mean(samples):.1f} ms "
        f"min={min(samples):.1f} ms max={max(samples):.1f} ms n={len(samples)}"
    )


def _plan_document(raw: Any) -> dict[str, Any]:
    """Normalise psycopg's JSON/JSONB result into the EXPLAIN document."""
    if isinstance(raw, str):
        import json

        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise RuntimeError("Unexpected EXPLAIN JSON response")
    return raw[0]


def profile(samples: int) -> None:
    pool_started = perf_counter()
    pool = get_pool()
    pool.wait()
    print(f"pool_warmup: {_elapsed_ms(pool_started):.1f} ms")

    checkout_samples: list[float] = []
    select_samples: list[float] = []
    complete_select_samples: list[float] = []
    for _ in range(samples):
        started = perf_counter()
        with get_connection() as conn:
            checkout_samples.append(_elapsed_ms(started))
            query_started = perf_counter()
            conn.execute("SELECT 1").fetchone()
            select_samples.append(_elapsed_ms(query_started))
        complete_select_samples.append(_elapsed_ms(started))

    vector_sql = f"""
        SELECT 1 - (embedding <=> %s) AS similarity
        FROM {CHUNKS_TABLE}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    held_select_samples: list[float] = []
    held_vector_samples: list[float] = []
    pooled_vector_samples: list[float] = []
    with get_connection() as conn:
        row = conn.execute(f"SELECT embedding FROM {CHUNKS_TABLE} LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Cannot profile vector search because the chunks table is empty")
        vector = row[0]
        application_vector = (
            vector.to_numpy().tolist() if hasattr(vector, "to_numpy") else list(vector)
        )

        for _ in range(samples):
            started = perf_counter()
            conn.execute("SELECT 1").fetchone()
            held_select_samples.append(_elapsed_ms(started))

            started = perf_counter()
            conn.execute(vector_sql, (vector, vector, 5)).fetchall()
            held_vector_samples.append(_elapsed_ms(started))

        explain_raw = conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + vector_sql,
            (vector, vector, 5),
        ).fetchone()
        if explain_raw is None:
            raise RuntimeError("EXPLAIN returned no result")
        explain = _plan_document(explain_raw[0])
        chunk_count = int(conn.execute(f"SELECT count(*) FROM {CHUNKS_TABLE}").fetchone()[0])

    for _ in range(samples):
        started = perf_counter()
        dense_search(application_vector, 5)
        pooled_vector_samples.append(_elapsed_ms(started))

    _print_summary("pool_checkout_with_validation", checkout_samples)
    _print_summary("select_1_after_checkout", select_samples)
    _print_summary("complete_pooled_select_1", complete_select_samples)
    _print_summary("held_connection_select_1", held_select_samples)
    _print_summary("held_connection_vector_query", held_vector_samples)
    _print_summary("complete_pooled_vector_search", pooled_vector_samples)
    print(f"server_planning_time: {float(explain['Planning Time']):.3f} ms")
    print(f"server_execution_time: {float(explain['Execution Time']):.3f} ms")
    print(f"server_plan_node: {explain['Plan']['Node Type']}")
    print(f"chunk_count: {chunk_count}")
    print(f"pool_stats: {pool.get_stats()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 100:
        parser.error("--samples must be between 1 and 100")

    try:
        profile(args.samples)
    finally:
        close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
