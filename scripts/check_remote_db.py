"""Verify a configured remote PostgreSQL database without changing it.

This check deliberately does not call ``init_schema`` or issue DDL. It proves
that DATABASE_URL is reachable and reports whether the pgvector extension was
enabled by the database provider before RAGuard is allowed to ingest data.
"""

from __future__ import annotations

import sys

import psycopg

from src.config import get_settings


def main() -> int:
    """Return a user-actionable status without ever printing credentials."""
    settings = get_settings()
    try:
        with psycopg.connect(settings.database_url, connect_timeout=10) as conn:
            row = conn.execute(
                """
                SELECT current_database(),
                       EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
                """
            ).fetchone()
    except psycopg.Error as exc:
        # Driver messages can include hostnames and other connection metadata;
        # do not risk echoing any part of a URL from a diagnostic command.
        print(f"Database connection failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    database_name, vector_enabled = row
    print(f"Connected to database: {database_name}")
    if not vector_enabled:
        print(
            "pgvector is not enabled. Enable the 'vector' extension in the managed database, "
            "then run this check again.",
            file=sys.stderr,
        )
        return 2

    print("pgvector extension: enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
