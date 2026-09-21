"""Run schema setup with the configured administrative database connection.

Use this command as a deployment release step before starting production API
workers.  It is intentionally separate from FastAPI lifespan so a runtime
database role does not need DDL privileges.
"""

from __future__ import annotations

from src.retrieval.vector_store import close_pool, init_schema


def main() -> int:
    try:
        init_schema()
    finally:
        close_pool()
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
