"""Measure cross-encoder latency on an explicit device.

This is deliberately a benchmark, not an evaluation replacement. It keeps the
existing retrieval configuration and corpus intact, reports model-load time
separately, and writes a transient JSON artifact so a CPU/CUDA comparison can
be reviewed before changing the service configuration.

Examples:

    python scripts/benchmark_reranker.py --device cpu --limit 12
    python scripts/benchmark_reranker.py --device cuda --limit 12
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
from src.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from src.retrieval.hybrid import get_hybrid_retriever  # noqa: E402
from src.retrieval.types import RetrievedChunk  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile for a compact, dependency-free report."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _require_device(device: str) -> None:
    if not device.lower().startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("CUDA benchmark requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to this process. "
            "For Docker, enable GPU access before running the CUDA benchmark."
        )


def _cases(limit: int) -> list[dict[str, Any]]:
    answerable = [case for case in load_golden_dataset() if case.get("expected_sources")]
    return answerable[:limit] if limit else answerable


def _warm_reranker(
    reranker: CrossEncoderReranker,
    query: str,
    candidates: list[RetrievedChunk],
) -> float:
    started = time.perf_counter()
    result = reranker.rerank_with_diagnostics(query, candidates)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not result.reranker_used:
        raise RuntimeError(f"reranker did not run: {result.failure or 'unknown failure'}")
    return elapsed_ms


def run_benchmark(device: str, limit: int) -> dict[str, Any]:
    """Benchmark only reranker inference; retrieval happens outside the timer."""
    _require_device(device)
    cases = _cases(limit)
    if not cases:
        raise RuntimeError("golden dataset has no answerable cases")

    settings = get_settings()
    retriever = get_hybrid_retriever()
    reranker = CrossEncoderReranker(device=device)

    first = cases[0]
    warmup_candidates = retriever.retrieve(first["question"])
    model_load_and_warmup_ms = _warm_reranker(reranker, first["question"], warmup_candidates)

    samples: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        candidates = retriever.retrieve(case["question"])
        started = time.perf_counter()
        result = reranker.rerank_with_diagnostics(case["question"], candidates)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not result.reranker_used:
            raise RuntimeError(
                f"{case['id']}: reranker did not run: {result.failure or 'unknown failure'}"
            )
        latencies.append(latency_ms)
        samples.append(
            {
                "case_id": case["id"],
                "candidate_count": result.candidate_count,
                "returned_count": len(result.chunks),
                "latency_ms": round(latency_ms, 1),
            }
        )

    return {
        "report": "reranker_device_benchmark",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": reranker._loaded_model_name,  # Loaded model is benchmark metadata.
        "device": device,
        "dataset": {
            "path": "src/evaluation/golden_dataset.json",
            "cases": len(cases),
        },
        "configuration": {
            "embedding_device": settings.model_device,
            "rerank_candidate_top_k": settings.rerank_candidate_top_k,
            "reranker_batch_size": settings.reranker_batch_size,
            "reranker_max_length": settings.reranker_max_length,
        },
        "model_load_and_warmup_ms": round(model_load_and_warmup_ms, 1),
        "inference_latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(statistics.median(latencies), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
        },
        "cases": samples,
    }


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Benchmark RAGuard reranker latency by device")
    parser.add_argument("--device", default=settings.resolved_reranker_device)
    parser.add_argument("--limit", type=int, default=12, help="0 runs all answerable cases")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "reranker_device_benchmark.json",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or greater")

    try:
        payload = run_benchmark(args.device, args.limit)
    except RuntimeError as exc:
        print(f"BENCHMARK_BLOCKED: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics = payload["inference_latency_ms"]
    print(f"device: {payload['device']}")
    print(f"model: {payload['model']}")
    print(f"cases: {payload['dataset']['cases']}")
    print(f"model load + warm-up: {payload['model_load_and_warmup_ms']:.1f} ms")
    print(f"inference: mean={metrics['mean']:.1f} ms p50={metrics['p50']:.1f} ms "
          f"p95={metrics['p95']:.1f} ms")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
