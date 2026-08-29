"""Compare explicit Voyage ordering with local BGE on identical top-20 candidates.

This is a live, opt-in evaluation tool. It never runs as part of pytest and
requires both ``RERANKER_PROVIDER=voyage`` and ``RERANKER_REMOTE_ALLOWED=true``.
Voyage chooses the evidence order, then local BGE scores that fixed top-five
in one batch for the existing confidence pipeline. Voyage scores are reported
as raw distribution data only; this script does not create or infer a
confidence-threshold calibration.

Example:
    $env:RERANKER_PROVIDER='voyage'
    $env:RERANKER_REMOTE_ALLOWED='true'
    $env:VOYAGE_API_KEY='...'
    python scripts/benchmark_voyage_reranker.py --limit 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.evaluation.metrics import load_golden_dataset  # noqa: E402
from src.evaluation.reranking_eval import aggregate, score_case  # noqa: E402
from src.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from src.reranking.provider import VoyageReranker  # noqa: E402
from src.retrieval.hybrid import get_hybrid_retriever  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))]


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 1) if values else 0.0,
        "p50": round(statistics.median(values), 1) if values else 0.0,
        "p95": round(_percentile(values, 0.95), 1),
    }


def _require_explicit_voyage(settings: Any) -> None:
    if settings.reranker_provider != "voyage":
        raise RuntimeError("set RERANKER_PROVIDER=voyage before this live benchmark")
    if not settings.reranker_remote_allowed:
        raise RuntimeError("set RERANKER_REMOTE_ALLOWED=true before this live benchmark")
    if not settings.voyage_api_key:
        raise RuntimeError("set VOYAGE_API_KEY before this live benchmark")


def run_comparison(*, limit: int, local_device: str | None = None) -> dict[str, Any]:
    """Run both rerankers against the same retrieved candidate list per case."""
    settings = get_settings()
    _require_explicit_voyage(settings)
    cases = [case for case in load_golden_dataset() if case.get("expected_sources")]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("golden dataset has no answerable cases")

    retriever = get_hybrid_retriever()
    local = CrossEncoderReranker(device=local_device or settings.resolved_reranker_device)
    voyage = VoyageReranker(
        api_key=settings.voyage_api_key,
        model_name=settings.voyage_rerank_model,
        timeout_seconds=settings.hosted_rerank_timeout_seconds,
        max_retries=settings.hosted_rerank_max_retries,
    )
    cases_by_id = {case["id"]: case for case in cases}
    local_results = []
    hybrid_results = []
    samples: list[dict[str, Any]] = []
    local_latencies: list[float] = []
    voyage_latencies: list[float] = []
    bge_top_five_latencies: list[float] = []
    bge_top_five_cpu_times: list[float] = []
    hybrid_latencies: list[float] = []
    raw_scores: list[float] = []

    for case in cases:
        candidates = retriever.retrieve(case["question"])
        candidate_ids = [chunk.chunk_id for chunk in candidates[: settings.rerank_candidate_top_k]]

        local_started = time.perf_counter()
        local_result = local.rerank_with_diagnostics(case["question"], candidates)
        local_latency = (time.perf_counter() - local_started) * 1000.0
        hosted_result = voyage.rerank_with_diagnostics(
            case["question"],
            candidates,
            top_k=min(settings.rerank_top_k, settings.hosted_rerank_top_k),
            candidate_top_k=min(
                settings.rerank_candidate_top_k, settings.hosted_rerank_max_candidates
            ),
        )
        if hosted_result.reranker_used:
            # BGE scores confidence only. This method never sorts, so Voyage
            # remains solely responsible for the final evidence order.
            hybrid_result = local.score_fixed_order_with_diagnostics(
                case["question"], hosted_result.chunks
            )
        else:
            hybrid_result = hosted_result

        local_latencies.append(local_latency)
        voyage_latencies.append(hosted_result.hosted_latency_ms)
        bge_top_five_latencies.append(hybrid_result.bge_scoring_latency_ms)
        bge_top_five_cpu_times.append(hybrid_result.bge_scoring_cpu_time_ms)
        hybrid_latencies.append(
            hosted_result.hosted_latency_ms + hybrid_result.bge_scoring_latency_ms
        )
        raw_scores.extend(hosted_result.provider_raw_scores.values())
        local_results.append(
            score_case(
                "local_bge",
                case,
                local_result.chunks,
                reranker_used=local_result.reranker_used,
                failure=local_result.failure,
            )
        )
        hybrid_results.append(
            score_case(
                "voyage_bge_fixed_order",
                case,
                hybrid_result.chunks,
                reranker_used=hybrid_result.reranker_used,
                failure=hybrid_result.failure,
            )
        )
        samples.append(
            {
                "case_id": case["id"],
                "candidate_chunk_ids": candidate_ids,
                "local": {
                    "latency_ms": round(local_latency, 1),
                    "provider_order": local_result.provider_order,
                    "failure": local_result.failure,
                },
                "voyage": {
                    "latency_ms": round(hosted_result.hosted_latency_ms, 1),
                    "retry_count": hosted_result.retry_count,
                    "provider_order": hosted_result.provider_order,
                    "raw_score_count": len(hosted_result.provider_raw_scores),
                    "failure": hosted_result.failure,
                },
                "bge_top_five": {
                    "latency_ms": round(hybrid_result.bge_scoring_latency_ms, 1),
                    "cpu_time_ms": round(hybrid_result.bge_scoring_cpu_time_ms, 1),
                    "order_preserved": (
                        [chunk.chunk_id for chunk in hybrid_result.chunks]
                        == hosted_result.provider_order
                    ),
                    "failure": hybrid_result.failure,
                },
                "hybrid_total_latency_ms": round(
                    hosted_result.hosted_latency_ms + hybrid_result.bge_scoring_latency_ms, 1
                ),
            }
        )

    return {
        "report": "voyage_vs_local_reranker_evaluation",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {"path": "src/evaluation/golden_dataset.json", "cases": len(cases)},
        "contract": {
            "candidate_top_k": settings.rerank_candidate_top_k,
            "final_top_k": settings.rerank_top_k,
            "same_candidates_per_case": True,
            "voyage_ordering": "top-20 to top-5",
            "confidence_scoring": "BGE fixed-order top-5 batch",
        },
        "providers": {
            "local": {"model": local.model_name, "device": local.device},
            "voyage": {"model": settings.voyage_rerank_model},
        },
        "latency_ms": {
            "local_full_bge": _latency_summary(local_latencies),
            "voyage": _latency_summary(voyage_latencies),
            "bge_fixed_order_top_five": _latency_summary(bge_top_five_latencies),
            "bge_fixed_order_top_five_cpu": _latency_summary(bge_top_five_cpu_times),
            "voyage_plus_bge_top_five": _latency_summary(hybrid_latencies),
        },
        "quality": {
            "local": aggregate(local_results, cases_by_id),
            "voyage_bge_fixed_order": aggregate(hybrid_results, cases_by_id),
        },
        "voyage_raw_score_distribution": {
            "count": len(raw_scores),
            "min": min(raw_scores) if raw_scores else None,
            "max": max(raw_scores) if raw_scores else None,
            "mean": round(statistics.mean(raw_scores), 6) if raw_scores else None,
        },
        "calibration_next_step": (
            "Voyage raw values never enter BGE thresholds. BGE scores the fixed Voyage order; "
            "approve the hybrid only after quality, evidence-grade, citation, and abstention "
            "regression evaluation on a held-out set."
        ),
        "cases": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Voyage-versus-local reranker evaluation")
    parser.add_argument("--limit", type=int, default=0, help="0 runs every answerable golden case")
    parser.add_argument("--local-device", default=None, help="optional local BGE device override")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "voyage_vs_local_reranker.json",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    try:
        report = run_comparison(limit=args.limit, local_device=args.local_device)
    except RuntimeError as exc:
        print(f"BENCHMARK_BLOCKED: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"latency_ms": report["latency_ms"], "quality": report["quality"]}, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
