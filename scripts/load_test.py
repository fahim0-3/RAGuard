"""Run a bounded, local-only concurrency test against a RAGuard API.

This tool is intentionally limited to loopback addresses. It is for a local
Docker Compose or native API process, never a public deployment. It records
only aggregate timings and public outcome/status counts; questions and answers
are neither printed nor written to disk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

DEFAULT_QUERY = "How long does a refund take to reach my credit card?"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True, slots=True)
class Sample:
    latency_ms: float
    status_code: int
    outcome: str
    error: str = ""


def _is_loopback(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in LOOPBACK_HOSTS


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarise(samples: list[Sample]) -> dict[str, object]:
    latencies = [sample.latency_ms for sample in samples]
    statuses = Counter(str(sample.status_code) for sample in samples)
    outcomes = Counter(sample.outcome or "request_failed" for sample in samples)
    errors = Counter(sample.error for sample in samples if sample.error)
    completed = sum(1 for sample in samples if 200 <= sample.status_code < 300)
    return {
        "requests": len(samples),
        "successful_http_responses": completed,
        "http_statuses": dict(sorted(statuses.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "errors": dict(sorted(errors.items())),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": round(_percentile(latencies, 0.95), 2),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def _request(base_url: str, query: str, timeout_s: float) -> Sample:
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(f"{base_url.rstrip('/')}/query", json={"query": query})
        latency_ms = (time.perf_counter() - started) * 1_000.0
        try:
            body = response.json()
        except ValueError:
            body = {}
        return Sample(latency_ms, response.status_code, str(body.get("outcome") or ""))
    except httpx.HTTPError as exc:
        return Sample(
            (time.perf_counter() - started) * 1_000.0,
            0,
            "request_failed",
            type(exc).__name__,
        )


def run(
    base_url: str, query: str, requests: int, concurrency: int, timeout_s: float
) -> dict[str, object]:
    if not _is_loopback(base_url):
        raise ValueError("load testing is restricted to localhost or a loopback address")
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="raguard-load") as executor:
        futures = [executor.submit(_request, base_url, query, timeout_s) for _ in range(requests)]
        samples = [future.result() for future in as_completed(futures)]
    return summarise(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    if not 1 <= args.requests <= 10_000:
        parser.error("--requests must be between 1 and 10000")
    if not 1 <= args.concurrency <= 128:
        parser.error("--concurrency must be between 1 and 128")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    try:
        result = run(args.base_url, args.query, args.requests, args.concurrency, args.timeout_s)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    sys.exit(main())
