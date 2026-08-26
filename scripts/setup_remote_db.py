"""Enable the database extension required by RAGuard.

This is intentionally separate from the read-only connection check so the
database mutation is explicit and repeatable. RAGuard's application tables and
indexes remain owned by ``src.retrieval.vector_store.init_schema``.
"""

from __future__ import annotations

import sys

import psycopg

from src.config import get_settings


def main() -> int:
    """Enable pgvector without printing connection metadata or credentials."""
    settings = get_settings()
    try:
        with psycopg.connect(settings.database_url, connect_timeout=10) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    except psycopg.Error as exc:
        print(f"Database setup failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print("pgvector extension: enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
