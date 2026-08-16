"""Retry accounting for the self-healing graph.

The graph contains a genuine cycle: `query_rewriter -> hybrid_retrieve ->
rerank -> evidence_grader -> query_rewriter`. A cycle with a routing function
that reads mutable state is exactly where a workflow acquires an infinite loop,
so the bound lives here as pure functions over the state rather than inline in
the routing lambda where it would be untestable.

Two independent counters, because they bound different loops:

- `retry_count` bounds retrieval retries after weak evidence.
- `regeneration_count` bounds regeneration after a failed citation check.

Both are explicit state fields. Neither is derived from list lengths, because
a rewrite that produces no new query still consumes an attempt, and deriving
the count from `rewritten_queries` would silently grant an extra pass.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RetryDecision",
    "may_regenerate",
    "may_retry",
    "record_regeneration",
    "record_retry",
    "retry_snapshot",
]


class RetryDecision:
    """Namespaced constants for the routing functions to return."""

    RETRY = "retry"
    STOP = "stop"


def may_retry(state: dict[str, Any]) -> bool:
    """True when another retrieval retry is permitted."""
    return int(state.get("retry_count", 0)) < int(state.get("max_retries", 0))


def may_regenerate(state: dict[str, Any]) -> bool:
    """True when another generation attempt is permitted."""
    return int(state.get("regeneration_count", 0)) < int(state.get("max_regenerations", 0))


def record_retry(state: dict[str, Any]) -> int:
    """Next retry count. Callers persist it; this never mutates in place."""
    return int(state.get("retry_count", 0)) + 1


def record_regeneration(state: dict[str, Any]) -> int:
    return int(state.get("regeneration_count", 0)) + 1


def retry_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Observable retry accounting, for traces and reports."""
    return {
        "retry_count": int(state.get("retry_count", 0)),
        "max_retries": int(state.get("max_retries", 0)),
        "retries_remaining": max(
            0, int(state.get("max_retries", 0)) - int(state.get("retry_count", 0))
        ),
        "regeneration_count": int(state.get("regeneration_count", 0)),
        "max_regenerations": int(state.get("max_regenerations", 0)),
        "rewritten_queries": list(state.get("rewritten_queries", [])),
    }
