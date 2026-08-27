"""Verify a configured remote PostgreSQL database without changing it.

This check deliberately does not call ``init_schema`` or issue DDL. It proves
that DATABASE_URL is reachable and reports whether the pgvector extension was
enabled by the database provider before RAGuard is allowed to ingest data.
"""

from __future__ import annotations

import sys

import psycopg

from src.config import get_settings


def connection_failure_message(error_text: str) -> str:
    """Classify a connection error without reflecting DSN or endpoint details.

    Driver error strings may include the full host list resolved from a managed
    endpoint. They are useful for internal classification but must never be
    echoed by a command users are encouraged to run with production secrets.
    """
    normalized = error_text.lower()
    if "permission denied" in normalized and (
        "10013" in normalized or "connection to server" in normalized
    ):
        return (
            "Database connection is blocked by local network policy. Allow outbound TCP 5432 "
            "or run this check from a network-enabled environment."
        )
    if "timed out" in normalized or "timeout" in normalized:
        return (
            "Database connection timed out. Verify outbound TCP 5432 access and the managed "
            "database's network policy."
        )
    if any(token in normalized for token in ("could not translate host", "getaddrinfo")):
        return "Database hostname could not be resolved. Verify DNS and the DATABASE_URL host."
    return "Database connection failed. Run the read-only check from an environment with database access."


def main() -> int:
    """Return a user-actionable status without ever printing credentials."""
    settings = get_settings()
    try:
        with psycopg.connect(
            settings.database_url,
            connect_timeout=settings.db_connect_timeout_s,
        ) as conn:
            row = conn.execute(
                """
                SELECT current_database(),
                       EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
                """
            ).fetchone()
    except psycopg.Error as exc:
        # Driver messages can include hostnames and other connection metadata;
        # classify them internally but never reflect them to stdout/stderr.
        print(connection_failure_message(str(exc)), file=sys.stderr)
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
