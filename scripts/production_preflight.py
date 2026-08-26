"""Fail-fast production checks with optional read-only dependency probes."""

from __future__ import annotations

import argparse

import psycopg

from src.config import get_settings, validate_production_settings


def _check_database() -> tuple[bool, str]:
    settings = get_settings()
    try:
        with psycopg.connect(
            settings.database_url,
            connect_timeout=settings.db_connect_timeout_s,
        ) as conn:
            row = conn.execute(
                """
                SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector'),
                       to_regclass('public.chunks') IS NOT NULL
                """
            ).fetchone()
            vector_enabled, chunks_exists = row or (False, False)
            if not vector_enabled:
                return False, "pgvector extension is not enabled"
            if not chunks_exists:
                return False, "chunks table does not exist"
            chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if chunk_count < 1:
                return False, "chunks table is empty"
    except psycopg.Error as exc:
        return False, f"database unavailable ({type(exc).__name__})"
    return True, f"database ready ({chunk_count} chunks)"


def _check_redis() -> tuple[bool, str]:
    from redis import Redis
    from redis.exceptions import RedisError

    settings = get_settings()
    client = None
    try:
        client = Redis.from_url(
            settings.admission_redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()
    except (RedisError, ValueError) as exc:
        return False, f"Redis unavailable ({type(exc).__name__})"
    finally:
        if client is not None:
            client.close()
    return True, "Redis ready"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-database", action="store_true")
    parser.add_argument("--check-redis", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.runtime_environment != "production":
        print("ERROR environment_not_production: set RAGUARD_ENVIRONMENT=production")
        return 2

    report = validate_production_settings(settings)
    for issue in report.errors:
        print(f"ERROR {issue.code}: {issue.message}")
    for issue in report.warnings:
        print(f"WARNING {issue.code}: {issue.message}")
    if not report.ok:
        return 2

    dependency_checks = []
    if args.check_database:
        dependency_checks.append(_check_database())
    if args.check_redis:
        dependency_checks.append(_check_redis())
    for ok, message in dependency_checks:
        print(("OK " if ok else "ERROR ") + message)
    if any(not ok for ok, _ in dependency_checks):
        return 3

    print("Production preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
