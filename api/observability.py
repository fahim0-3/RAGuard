"""Privacy-safe, process-local service telemetry.

This module intentionally records only bounded operational labels and numeric
aggregates. It never receives a customer query, IP address, policy excerpt,
prompt, credential, or provider exception message. Deployments with multiple
API workers should export these snapshots to a shared metrics backend.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_LATENCY_BUCKETS_MS = (100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0)


class RuntimeMetrics:
    """Thread-safe counters and latency aggregates for one API process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._latency_count = 0
        self._latency_sum_ms = 0.0
        self._latency_max_ms = 0.0
        self._latency_buckets: Counter[float] = Counter()

    def reset(self) -> None:
        """Clear process-local state for an explicit service/test reset."""
        with self._lock:
            self._counters.clear()
            self._latency_count = 0
            self._latency_sum_ms = 0.0
            self._latency_max_ms = 0.0
            self._latency_buckets.clear()

    def record_admitted(self) -> None:
        with self._lock:
            self._counters["query_admitted_total"] += 1

    def record_admission_rejected(self, reason: str) -> None:
        with self._lock:
            self._counters["query_admission_rejections_total"] += 1
            self._counters[f"query_admission_rejected_{reason}_total"] += 1

    def record_failure(self, reason: str) -> None:
        with self._lock:
            self._counters["query_failed_total"] += 1
            self._counters[f"query_failure_{reason}_total"] += 1

    def record_completed(
        self,
        *,
        outcome: str,
        latency_ms: float,
        retrieved_chunk_count: int = 0,
        evidence_sufficient: bool | None = None,
        verification_status: str = "not_checked",
        reranker_used: bool | None = None,
        llm_calls_used: int = 0,
        budget_exhausted: bool = False,
        budget_exhaustion_reason: str = "",
    ) -> None:
        """Record a completed admitted query using allow-listed categorical fields."""
        with self._lock:
            self._counters["query_completed_total"] += 1
            self._counters[f"query_outcome_{outcome}_total"] += 1
            self._counters["retrieved_chunks_total"] += max(0, retrieved_chunk_count)
            self._counters["llm_calls_total"] += max(0, llm_calls_used)
            if budget_exhausted:
                self._counters["request_budget_exhausted_total"] += 1
                safe_reason = (
                    budget_exhaustion_reason
                    if budget_exhaustion_reason in {"deadline", "llm_call_limit"}
                    else "unknown"
                )
                self._counters[f"request_budget_reason_{safe_reason}_total"] += 1
            self._counters[f"verification_{verification_status}_total"] += 1
            if evidence_sufficient is not None:
                self._counters[
                    f"evidence_sufficient_{str(evidence_sufficient).lower()}_total"
                ] += 1
            if reranker_used is not None:
                self._counters[f"reranker_used_{str(reranker_used).lower()}_total"] += 1

            safe_latency = max(0.0, float(latency_ms))
            self._latency_count += 1
            self._latency_sum_ms += safe_latency
            self._latency_max_ms = max(self._latency_max_ms, safe_latency)
            for bucket in _LATENCY_BUCKETS_MS:
                if safe_latency <= bucket:
                    self._latency_buckets[bucket] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-ready aggregates without per-request data."""
        with self._lock:
            counters = dict(self._counters)
            average = self._latency_sum_ms / self._latency_count if self._latency_count else 0.0
            return {
                "schema_version": 1,
                "queries": {
                    "admitted": counters.get("query_admitted_total", 0),
                    "completed": counters.get("query_completed_total", 0),
                    "failed": counters.get("query_failed_total", 0),
                    "admission_rejected": counters.get("query_admission_rejections_total", 0),
                    "outcomes": _group(counters, "query_outcome_", "_total"),
                    "failures": _group(counters, "query_failure_", "_total"),
                    "admission_rejections": _group(
                        counters, "query_admission_rejected_", "_total"
                    ),
                },
                "grounding": {
                    "retrieved_chunks_total": counters.get("retrieved_chunks_total", 0),
                    "evidence_sufficient": _group(
                        counters, "evidence_sufficient_", "_total"
                    ),
                    "verification": _group(counters, "verification_", "_total"),
                    "reranker_used": _group(counters, "reranker_used_", "_total"),
                },
                "budget": {
                    "llm_calls_total": counters.get("llm_calls_total", 0),
                    "exhausted_total": counters.get("request_budget_exhausted_total", 0),
                    "exhaustion_reasons": _group(
                        counters, "request_budget_reason_", "_total"
                    ),
                },
                "latency_ms": {
                    "count": self._latency_count,
                    "average": round(average, 2),
                    "max": round(self._latency_max_ms, 2),
                },
            }

    def prometheus_text(self) -> str:
        """Return Prometheus text exposition with only bounded metric labels.

        This exporter deliberately has no query, customer identifier, source
        document, or request ID label. Those values create high-cardinality,
        privacy-sensitive time series and belong in traces, not metrics.
        """
        with self._lock:
            counters = dict(self._counters)
            latency_count = self._latency_count
            latency_sum_seconds = self._latency_sum_ms / 1_000.0
            latency_max_seconds = self._latency_max_ms / 1_000.0
            latency_buckets = dict(self._latency_buckets)

        lines: list[str] = []
        _emit_counter(lines, "raguard_queries_admitted", "Queries admitted to the workflow.", counters.get("query_admitted_total", 0))
        _emit_counter(lines, "raguard_queries_completed", "Queries completed by the workflow.", counters.get("query_completed_total", 0))
        _emit_counter(lines, "raguard_queries_failed", "Queries that failed after admission.", counters.get("query_failed_total", 0))
        _emit_counter(lines, "raguard_query_admission_rejections", "Queries rejected before workflow execution.", counters.get("query_admission_rejections_total", 0))
        _emit_counter(lines, "raguard_retrieved_chunks", "Retrieved chunks considered by completed queries.", counters.get("retrieved_chunks_total", 0))
        _emit_counter(lines, "raguard_llm_calls", "Budgeted model invocations used by completed queries.", counters.get("llm_calls_total", 0))
        _emit_counter(lines, "raguard_request_budget_exhausted", "Completed queries that exhausted a request budget.", counters.get("request_budget_exhausted_total", 0))

        _emit_labelled_counter(lines, "raguard_query_outcomes", "Completed queries by public outcome.", _group(counters, "query_outcome_", "_total"), "outcome")
        _emit_labelled_counter(lines, "raguard_query_failures", "Failed queries by safe failure category.", _group(counters, "query_failure_", "_total"), "reason")
        _emit_labelled_counter(lines, "raguard_query_admission_rejections_by_reason", "Admission rejections by safe reason.", _group(counters, "query_admission_rejected_", "_total"), "reason")
        _emit_labelled_counter(lines, "raguard_evidence_decisions", "Evidence sufficiency decisions.", _group(counters, "evidence_sufficient_", "_total"), "sufficient")
        _emit_labelled_counter(lines, "raguard_verification_results", "Citation verification results.", _group(counters, "verification_", "_total"), "status")
        _emit_labelled_counter(lines, "raguard_reranker_usage", "Completed queries by reranker usage.", _group(counters, "reranker_used_", "_total"), "used")
        _emit_labelled_counter(lines, "raguard_request_budget_exhaustions", "Request budget exhaustion by bounded reason.", _group(counters, "request_budget_reason_", "_total"), "reason")

        lines.extend((
            "# HELP raguard_query_latency_seconds Workflow latency for completed queries.",
            "# TYPE raguard_query_latency_seconds histogram",
        ))
        for bucket in _LATENCY_BUCKETS_MS:
            lines.append(
                f'raguard_query_latency_seconds_bucket{{le="{bucket / 1_000:g}"}} {latency_buckets.get(bucket, 0)}'
            )
        lines.append(f'raguard_query_latency_seconds_bucket{{le="+Inf"}} {latency_count}')
        lines.append(f"raguard_query_latency_seconds_sum {latency_sum_seconds:.6f}")
        lines.append(f"raguard_query_latency_seconds_count {latency_count}")
        lines.extend((
            "# HELP raguard_query_latency_seconds_max Maximum completed-query latency since process start.",
            "# TYPE raguard_query_latency_seconds_max gauge",
            f"raguard_query_latency_seconds_max {latency_max_seconds:.6f}",
        ))
        return "\n".join(lines) + "\n"


def _group(counters: dict[str, int], prefix: str, suffix: str) -> dict[str, int]:
    return {
        name.removeprefix(prefix).removesuffix(suffix): value
        for name, value in counters.items()
        if name.startswith(prefix) and name.endswith(suffix)
    }


def _emit_counter(lines: list[str], name: str, help_text: str, value: int) -> None:
    lines.extend((f"# HELP {name}_total {help_text}", f"# TYPE {name}_total counter", f"{name}_total {value}"))


def _emit_labelled_counter(
    lines: list[str], name: str, help_text: str, values: dict[str, int], label: str
) -> None:
    lines.extend((f"# HELP {name}_total {help_text}", f"# TYPE {name}_total counter"))
    for value, count in sorted(values.items()):
        lines.append(f'{name}_total{{{label}="{_prometheus_label(value)}"}} {count}')


def _prometheus_label(value: str) -> str:
    """Escape a bounded internal category for the Prometheus text format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def log_event(name: str, **fields: object) -> None:
    """Emit one structured event with an allow-listed caller-owned payload."""
    logger.info("event=%s fields=%s", name, json.dumps(fields, sort_keys=True, separators=(",", ":")))


runtime_metrics = RuntimeMetrics()
