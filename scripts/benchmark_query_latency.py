"""End-to-end warm-query latency benchmark against a running RAGuard API.

Sends a fixed question set to `/query`, one request at a time, and records the
operational fields of each response: latency, graph path, per-stage timings,
LLM call count, provider and fallbacks, retries, and outcome.

Answer and passage text are never written out. A benchmark report is an
operational artefact and is shared freely; policy content does not belong in it.

Usage:
    python scripts/benchmark_query_latency.py --base-url http://127.0.0.1:8000 \\
        --spacing 45 --out reports/latency_baseline.json

`--spacing` matters. Free-tier hosted providers enforce per-minute token
budgets, and back-to-back requests turn a latency benchmark into a rate-limit
benchmark. Use `--spacing 0` deliberately, to measure the cascade itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

#: Covers every category the latency work must preserve: direct and specific
#: policy facts, identifiers, an unsupported topic, an ambiguous request, and a
#: question whose only correct response is abstention.
QUESTIONS: list[tuple[str, str]] = [
    ("refund_overview", "What is the refund policy?"),
    ("refund_timing", "How long do card refunds take?"),
    ("delivery", "When is a domestic parcel declared lost?"),
    ("payment_code", "What does gateway code PAY-409 mean?"),
    ("return_window", "How long do I have to return electronics?"),
    ("damage_window", "How long do I have to report concealed damage?"),
    ("return_fee", "What is the return shipping fee for a change of mind?"),
    ("unsupported", "Do you sell gaming laptops?"),
    ("ambiguous", "I have a problem with my order"),
    ("abstain", "What is your price-match policy?"),
]


@dataclass
class Run:
    category: str
    wall_s: float
    status: int
    outcome: str = ""
    failure_reason: str | None = None
    latency_ms: float = 0.0
    node_sequence: list[str] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)
    retrieval_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    retries: int = 0
    regenerations: int = 0
    speculative: bool = False
    provider: str | None = None
    fallbacks: list[str] = field(default_factory=list)
    skipped_providers: list[str] = field(default_factory=list)
    verification: str = ""
    citations: int = 0
    error: str = ""


def _totals(block: dict | None) -> dict[str, float]:
    return {name: round(float(v.get("total_ms", 0.0)), 1) for name, v in (block or {}).items()}


def ask(client: httpx.Client, base_url: str, category: str, question: str) -> Run:
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url}/query", json={"query": question})
    except httpx.HTTPError as exc:
        return Run(category, round(time.perf_counter() - started, 2), 0, error=type(exc).__name__)
    wall = round(time.perf_counter() - started, 2)
    try:
        body = response.json()
    except ValueError:
        return Run(category, wall, response.status_code, error="non_json")

    return Run(
        category=category,
        wall_s=wall,
        status=response.status_code,
        outcome=str(body.get("outcome", "")),
        failure_reason=body.get("failure_reason"),
        latency_ms=float(body.get("latency_ms") or 0.0),
        node_sequence=[step.get("node", "") for step in body.get("trace") or []],
        stage_ms=_totals(body.get("stage_latency_ms")),
        retrieval_ms=_totals(body.get("retrieval_latency_ms")),
        llm_calls=int(body.get("llm_calls_used") or 0),
        retries=int(body.get("retry_count") or 0),
        regenerations=int(body.get("regeneration_count") or 0),
        speculative=bool(body.get("speculative_generation")),
        provider=body.get("llm_provider"),
        fallbacks=list(body.get("llm_fallbacks") or []),
        skipped_providers=list(body.get("llm_skipped_providers") or []),
        verification=str(body.get("verification_status", "")),
        citations=len(body.get("citations") or []),
    )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; honest for the small samples a benchmark uses."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(runs: list[Run]) -> dict:
    walls = [r.wall_s for r in runs if r.status == 200]
    stage_names = sorted({name for r in runs for name in r.stage_ms})
    per_stage = {
        name: {
            "p50_ms": round(statistics.median([r.stage_ms.get(name, 0.0) for r in runs]), 1),
            "p95_ms": round(percentile([r.stage_ms.get(name, 0.0) for r in runs], 95), 1),
        }
        for name in stage_names
    }
    return {
        "requests": len(runs),
        "p50_s": round(statistics.median(walls), 2) if walls else None,
        "p95_s": round(percentile(walls, 95), 2) if walls else None,
        "min_s": min(walls) if walls else None,
        "max_s": max(walls) if walls else None,
        "mean_llm_calls": round(statistics.mean(r.llm_calls for r in runs), 2) if runs else 0,
        "requests_with_fallback": sum(1 for r in runs if r.fallbacks),
        "outcomes": {o: sum(1 for r in runs if r.outcome == o) for o in {r.outcome for r in runs}},
        "per_stage": per_stage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--spacing", type=float, default=45.0, help="seconds between requests")
    parser.add_argument("--repeat", type=int, default=1, help="passes over the question set")
    parser.add_argument("--only", default="", help="comma-separated categories to run")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    selected = [q for q in QUESTIONS if not args.only or q[0] in args.only.split(",")]
    runs: list[Run] = []
    with httpx.Client(timeout=args.timeout) as client:
        total = len(selected) * args.repeat
        for index in range(total):
            category, question = selected[index % len(selected)]
            run = ask(client, args.base_url, category, question)
            runs.append(run)
            print(
                f"{index + 1:>2}/{total} {category:<16} {run.wall_s:>6.2f}s "
                f"{run.outcome or run.error:<10} calls={run.llm_calls} "
                f"retries={run.retries} regen={run.regenerations} spec={int(run.speculative)} "
                f"provider={run.provider} fallbacks={run.fallbacks or '-'} "
                f"skipped={run.skipped_providers or '-'}",
                flush=True,
            )
            if index + 1 < total and args.spacing > 0:
                time.sleep(args.spacing)

    summary = summarise(runs)
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "runs": [asdict(r) for r in runs]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
