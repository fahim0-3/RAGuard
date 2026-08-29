"""Shared, bounded timing primitives for graph, API, and evaluation layers.

Only fixed operational stage names and finite non-negative numbers are allowed
to leave the workflow. Query text, request IDs, exception messages, and source
metadata never enter this module.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from typing import Any

MAX_TIMING_MS = 3_600_000.0

GRAPH_STAGE_NAMES: tuple[str, ...] = (
    "sanitize_and_classify",
    "risk_router",
    "ambiguity_detector",
    "hybrid_retrieve",
    "rerank",
    "evidence_grader",
    "query_rewriter",
    "generate_answer",
    "verify_citations",
    "finalize_answer",
    "abstain",
    "clarify",
    "escalate",
)

RETRIEVAL_STAGE_NAMES: tuple[str, ...] = (
    "query_embedding",
    "vector_search",
    "bm25_search",
    "rrf_fusion",
    "deduplication",
)

TimingSamples = dict[str, list[float]]


def elapsed_ms(started: float) -> float:
    """Return a monotonic, bounded duration suitable for operational state."""
    return round(min(max((time.perf_counter() - started) * 1000.0, 0.0), MAX_TIMING_MS), 3)


def merge_timing_samples(
    existing: Mapping[str, Iterable[float]] | None,
    incoming: Mapping[str, Iterable[float]] | None,
) -> TimingSamples:
    """LangGraph reducer that appends attempts instead of overwriting retries."""
    merged = {name: list(values) for name, values in (existing or {}).items()}
    for name, values in (incoming or {}).items():
        merged.setdefault(name, []).extend(values)
    return merged


def sanitise_timing_samples(
    raw: Mapping[str, Any] | None,
    allowed_names: Iterable[str],
) -> TimingSamples:
    """Filter untrusted state to fixed stage names and safe numeric samples."""
    allowed = frozenset(allowed_names)
    clean: TimingSamples = {}
    if not isinstance(raw, Mapping):
        return clean

    for name, values in raw.items():
        if name not in allowed:
            continue
        candidates = values if isinstance(values, (list, tuple)) else [values]
        safe_values: list[float] = []
        for value in candidates:
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number) or number < 0.0:
                continue
            safe_values.append(round(min(number, MAX_TIMING_MS), 3))
        if safe_values:
            clean[str(name)] = safe_values
    return clean


def aggregate_timing_samples(samples: Mapping[str, Iterable[float]]) -> dict[str, dict[str, Any]]:
    """Convert per-attempt samples into a compact per-request summary."""
    aggregated: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        numbers = [float(value) for value in values]
        if not numbers:
            continue
        total = sum(numbers)
        aggregated[name] = {
            "count": len(numbers),
            "total_ms": round(total, 3),
            "average_ms": round(total / len(numbers), 3),
            "max_ms": round(max(numbers), 3),
        }
    return aggregated
