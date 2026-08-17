"""Evaluation entry point: the thing that answers "better or worse?".

    python -m src.evaluation.run_eval --retrieval
    python -m src.evaluation.run_eval --generation --safety
    python -m src.evaluation.run_eval --all --fail-on-regression

Layers stay separate so a run reports what it actually measured. `--retrieval`
needs PostgreSQL but no API key; `--generation` and `--safety` need a provider;
`--ragas` needs a working RAGAS install. A layer that cannot run is reported
`BLOCKED`, never silently skipped and never counted as a pass — a broken
dependency must not be able to produce a green build.

Exit code 1 means a gate failed or a requested layer was blocked. That is what
turns the CI job red.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.gates import (
    GENERATION_GATES,
    RETRIEVAL_GATES,
    SAFETY_GATES,
    GateSuite,
    evaluate_gates,
)

logger = logging.getLogger(__name__)

EVALUATION_VERSION = "phase_i_v1"
REPORTS_DIR = PROJECT_ROOT / "reports"

LAYERS = ("retrieval", "reranking", "generation", "ragas", "safety")


# --------------------------------------------------------------------------
# Preconditions, reported rather than raised
# --------------------------------------------------------------------------


def check_database() -> tuple[bool, str]:
    try:
        from src.retrieval.vector_store import count_chunks

        chunks = count_chunks()
    except Exception as exc:
        return False, f"DATABASE_UNAVAILABLE: {type(exc).__name__}"
    if chunks == 0:
        return False, "DATABASE_EMPTY: run python -m src.ingestion.ingest"
    return True, f"{chunks} chunks indexed"


def check_llm() -> tuple[bool, str]:
    settings = get_settings()
    if settings.llm_provider == "gemini" and not settings.google_api_key:
        return False, "LLM_NOT_CONFIGURED: GOOGLE_API_KEY is not set"
    return True, f"provider={settings.llm_provider}"


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


def run_retrieval_layer() -> dict[str, Any]:
    ok, detail = check_database()
    if not ok:
        return {"status": "BLOCKED", "reason": detail, "metrics": {}}

    from src.evaluation.golden_eval import run_golden_evaluation

    payload = run_golden_evaluation()
    return {
        "status": "MEASURED",
        "metrics": payload.get("measured_metrics", {}),
        "latency": payload.get("latency", {}),
        "dataset": payload.get("dataset", {}),
        "report": payload,
    }


def run_generation_layer(limit: int | None = None) -> dict[str, Any]:
    ok, detail = check_database()
    if not ok:
        return {"status": "BLOCKED", "reason": detail, "metrics": {}}
    ok, detail = check_llm()
    if not ok:
        return {"status": "BLOCKED", "reason": detail, "metrics": {}}

    from src.evaluation.generation_eval import evaluate_generation

    payload = evaluate_generation(limit=limit)
    return {
        "status": payload.get("status", "MEASURED"),
        "metrics": payload.get("metrics", {}),
        "latency": payload.get("latency", {}),
        "report": payload,
    }


def run_safety_layer(generation_report: dict[str, Any] | None) -> dict[str, Any]:
    if not generation_report:
        return {
            "status": "BLOCKED",
            "reason": "SAFETY_REQUIRES_GENERATION: run with --generation",
            "metrics": {},
        }

    from src.evaluation.safety_eval import evaluate_safety

    payload = evaluate_safety(generation_report)
    return {
        "status": payload.get("status", "MEASURED"),
        "metrics": payload.get("metrics", {}),
        "report": payload,
    }


def run_ragas_layer(generation_report: dict[str, Any] | None) -> dict[str, Any]:
    from src.evaluation.ragas_eval import run_ragas_evaluation

    records = []
    for case in (generation_report or {}).get("cases", []):
        records.append(
            {
                "case_id": case.get("case_id"),
                "question": case.get("question"),
                "answer": case.get("answer_preview", ""),
                "outcome": case.get("actual_outcome"),
                "contexts": case.get("cited_policy_ids") or [],
                "ground_truth": case.get("ground_truth", ""),
            }
        )

    payload = run_ragas_evaluation(records)
    return {
        "status": payload.get("status"),
        "metrics": payload.get("metrics", {}),
        "report": payload,
    }


def run_reranking_layer() -> dict[str, Any]:
    """Deliberately not executed here.

    The Phase C comparison reranks every golden case with a 568 M parameter
    cross-encoder: roughly forty minutes on CPU. Putting that behind a CLI flag
    that CI might call by default is how a merge gate becomes something people
    disable, so it stays an explicit, separate invocation.
    """
    return {
        "status": "NOT_RUN",
        "reason": (
            "Heavy benchmark. Run explicitly: "
            "python -m src.evaluation.reranking_eval"
        ),
        "metrics": {},
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def overall_status(layers: dict[str, dict[str, Any]], suite: GateSuite) -> str:
    """The single status a human or CI job should read.

    Deliberately not the gate status alone. A run whose layers were all blocked
    applies zero gates, and zero failing gates is not the same thing as a
    passing evaluation — reporting PASS there would turn a missing database
    into a green build, which is the exact failure this phase exists to stop.
    """
    blocked = [name for name, layer in layers.items() if layer.get("status") == "BLOCKED"]
    if blocked:
        return "BLOCKED"
    if suite.failures():
        return "FAIL"
    if not suite.results:
        return "NOT_MEASURED"
    return "PASS"


def build_report(layers: dict[str, dict[str, Any]], suite: GateSuite) -> dict[str, Any]:
    settings = get_settings()
    from src.evaluation.metrics import golden_dataset_version
    from src.generation.prompts import PROMPT_VERSION

    return {
        "report": "raguard_evaluation",
        "phase": "I",
        "evaluation_version": EVALUATION_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_status": overall_status(layers, suite),
        "blocked_layers": [
            name for name, layer in layers.items() if layer.get("status") == "BLOCKED"
        ],
        "reproducibility": {
            "dataset_version": golden_dataset_version(),
            "prompt_version": PROMPT_VERSION,
            "llm_provider": settings.llm_provider,
            "generation_model": settings.llm_model or settings.gemini_model,
            "judge_model": settings.gemini_judge_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "reranker_enabled": settings.reranker_enabled,
            "verifier_backend": settings.verifier_backend,
            "retrieval": {
                "dense_top_k": settings.dense_top_k,
                "sparse_top_k": settings.sparse_top_k,
                "fusion_top_k": settings.fusion_top_k,
                "rrf_k": settings.rrf_k,
                "rerank_top_k": settings.rerank_top_k,
            },
            "graph": {
                "max_retries": settings.graph_max_retries,
                "max_regenerations": settings.graph_max_regenerations,
                "evidence_top_score_threshold": settings.evidence_top_score_threshold,
                "evidence_min_relevant_chunks": settings.evidence_min_relevant_chunks,
                "evidence_confidence_threshold": settings.evidence_confidence_threshold,
            },
        },
        "layers": {
            name: {k: v for k, v in layer.items() if k != "report"}
            for name, layer in layers.items()
        },
        "gates": suite.to_dict(),
        "detail": {name: layer.get("report") for name, layer in layers.items() if layer.get("report")},
    }


def collect_gates(layers: dict[str, dict[str, Any]]) -> GateSuite:
    """Apply only the gates whose layer actually ran."""
    gates = []
    measured: dict[str, float | None] = {}

    if layers.get("retrieval", {}).get("status") == "MEASURED":
        gates.extend(RETRIEVAL_GATES)
        measured.update(layers["retrieval"]["metrics"])

    if layers.get("generation", {}).get("status") == "MEASURED":
        gates.extend(GENERATION_GATES)
        for key, value in layers["generation"]["metrics"].items():
            # Retrieval and generation both report citation_id_validity; the
            # generation number is the stricter one, so it wins.
            measured[key] = value

    if layers.get("safety", {}).get("status") == "MEASURED":
        gates.extend(SAFETY_GATES)
        measured.update(layers["safety"]["metrics"])

    return evaluate_gates(gates, measured)


def print_summary(payload: dict[str, Any], layers: dict[str, dict[str, Any]]) -> None:
    print("\n=== RAGuard evaluation ===")
    repro = payload["reproducibility"]
    print(f"evaluation : {payload['evaluation_version']}  dataset: {repro['dataset_version']}")
    print(f"provider   : {repro['llm_provider']}  generation: {repro['generation_model']}")
    print(f"judge      : {repro['judge_model']}  verifier: {repro['verifier_backend']}\n")

    for name, layer in layers.items():
        status = layer.get("status", "?")
        reason = layer.get("reason", "")
        print(f"  {name:<12} {status}" + (f"  ({reason})" if reason else ""))

    gates = payload["gates"]
    print(f"\n  gates: {gates['passed']} passed, {gates['failed']} failed, "
          f"{gates['not_measured']} not measured")
    for category, status in gates["by_category"].items():
        print(f"    {category:<12} {status}")

    if gates["failures"]:
        print("\n  FAILURES:")
        for line in gates["failures"]:
            print(f"    - {line}")

    overall = payload["overall_status"]
    print(f"\n  gate status: {gates['status']}")
    print(f"  overall    : {overall}")
    if overall == "BLOCKED":
        print("    (a blocked layer was not evaluated; this is not a pass)")
    elif overall == "NOT_MEASURED":
        print("    (no gate had a measured value to check)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAGuard evaluation")
    for layer in LAYERS:
        parser.add_argument(f"--{layer}", action="store_true", help=f"run the {layer} layer")
    parser.add_argument("--all", action="store_true", help="run every layer")
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument("--no-save", action="store_true", help="do not write a report file")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when a gate fails or a requested layer is blocked",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    selected = {layer for layer in LAYERS if getattr(args, layer)}
    if args.all:
        selected = set(LAYERS)
    if not selected:
        selected = {"retrieval"}

    layers: dict[str, dict[str, Any]] = {}

    if "retrieval" in selected:
        layers["retrieval"] = run_retrieval_layer()
    if "reranking" in selected:
        layers["reranking"] = run_reranking_layer()
    if "generation" in selected:
        layers["generation"] = run_generation_layer(limit=args.limit)
    if "safety" in selected:
        layers["safety"] = run_safety_layer(
            layers.get("generation", {}).get("report") if "generation" in layers else None
        )
    if "ragas" in selected:
        layers["ragas"] = run_ragas_layer(
            layers.get("generation", {}).get("report") if "generation" in layers else None
        )

    suite = collect_gates(layers)
    payload = build_report(layers, suite)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print_summary(payload, layers)

    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = payload["timestamp"].replace(":", "").replace("-", "")[:15]
        path = args.output or REPORTS_DIR / f"evaluation_{stamp}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nReport written to {path}")

    blocked = [n for n, layer in layers.items() if layer.get("status") == "BLOCKED"]
    if blocked:
        print(f"\nBLOCKED layers: {', '.join(blocked)}")

    if args.fail_on_regression and (suite.failures() or blocked):
        print("\nMerge blocked. Fix the regression, or update the gate in a "
              "separate commit that documents why the old floor was wrong.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
