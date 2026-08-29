"""Phase C: does the cross-encoder actually help?

Four retrieval configurations are measured on the same golden dataset, in the
same process, against the same corpus:

    A. bm25_only          BM25 top-20, no fusion, no dedup
    B. vector_only        pgvector top-20, no fusion, no dedup
    C. hybrid_rrf         the Phase B baseline: BM25 + dense, RRF, dedup
    D. hybrid_rrf_rerank  C's top-20 rescored by the cross-encoder, top-5 kept

A and B are deliberately raw. Their question is "what does this retriever alone
find", so fusion and deduplication are not applied to them; C and D carry the
full Phase B pipeline. The report records this so the comparison is not read as
something it is not.

Every metric is cut at k <= 5. Configuration D returns five results by design,
so any metric at k > 5 would measure truncation rather than ranking quality and
would make D look worse for a reason that has nothing to do with relevance.

The metric primitives are imported from the Phase B module rather than
reimplemented. A comparison computed with a second, subtly different definition
of MRR is not a comparison.

Running the cross-encoder is not evidence that it helped. The verdict block
requires MRR@5 to improve while HitRate@5 and Recall@5 hold, and reports any
case whose rank got worse.

Run:  python -m src.evaluation.reranking_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.metrics import (
    golden_dataset_version,
    load_golden_dataset,
    resolve_dataset_path,
)
from src.evaluation.retrieval_eval import (
    DEFAULT_REPORT_PATH as BASELINE_REPORT_PATH,
)
from src.evaluation.retrieval_eval import (
    HIT_RATE_CUTOFFS,
    MRR_CUTOFF,
    _mean,
    first_relevant_rank,
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "reranking_comparison.json"

# Cut at 5 so every configuration is judged on the same depth.
RECALL_CUTOFF = 5

# The two cases the Phase B baseline could not rank first.
WATCHED_CASES = ("GC-002", "GC-010")

CONFIG_A = "bm25_only"
CONFIG_B = "vector_only"
CONFIG_C = "hybrid_rrf"
CONFIG_D = "hybrid_rrf_rerank"
CONFIG_ORDER = (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D)


# --------------------------------------------------------------------------
# Per-case, per-configuration measurement
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ConfigCaseResult:
    """One configuration's result for one golden case."""

    config: str
    case_id: str
    retrieved_sources: list[str]
    retrieved_policy_ids: list[str]
    first_relevant_rank: int | None
    metrics: dict[str, float] = field(default_factory=dict)
    top_results: list[dict[str, Any]] = field(default_factory=list)
    reranker_used: bool | None = None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "first_relevant_rank": self.first_relevant_rank,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "retrieved_policy_ids": self.retrieved_policy_ids[:10],
            "top_results": self.top_results,
            "reranker_used": self.reranker_used,
            "failure": self.failure,
        }


def _summarise(chunk: RetrievedChunk, rank: int) -> dict[str, Any]:
    """Every stage score, so a ranking change can be attributed to a stage."""
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "policy_id": chunk.policy_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "bm25_score": round(chunk.sparse_score, 4) if chunk.sparse_score is not None else None,
        "vector_score": round(chunk.dense_score, 4) if chunk.dense_score is not None else None,
        "rrf_score": round(chunk.fusion_score, 6) if chunk.fusion_score is not None else None,
        "rerank_score": round(chunk.rerank_score, 4) if chunk.rerank_score is not None else None,
        "normalised_rerank_score": (
            round(chunk.normalised_rerank_score, 4)
            if chunk.normalised_rerank_score is not None
            else None
        ),
        "retriever_ranks": chunk.retriever_ranks,
    }


def score_case(
    config: str,
    case: dict[str, Any],
    chunks: list[RetrievedChunk],
    *,
    reranker_used: bool | None = None,
    failure: str | None = None,
    diagnostic_depth: int = 5,
) -> ConfigCaseResult:
    ranked_sources = [c.source for c in chunks]
    expected = set(case["expected_sources"])

    result = ConfigCaseResult(
        config=config,
        case_id=case["id"],
        retrieved_sources=ranked_sources,
        retrieved_policy_ids=[c.policy_id for c in chunks],
        first_relevant_rank=first_relevant_rank(ranked_sources, expected),
        top_results=[
            _summarise(chunk, rank) for rank, chunk in enumerate(chunks[:diagnostic_depth], start=1)
        ],
        reranker_used=reranker_used,
        failure=failure,
    )

    if case.get("should_abstain"):
        # No passage is correct; excluded from aggregates exactly as in Phase B.
        return result

    result.metrics = {
        **{
            f"hit_rate_at_{k}": hit_rate_at_k(ranked_sources, expected, k) for k in HIT_RATE_CUTOFFS
        },
        f"recall_at_{RECALL_CUTOFF}": recall_at_k(ranked_sources, expected, RECALL_CUTOFF),
        f"mrr_at_{MRR_CUTOFF}": reciprocal_rank_at_k(ranked_sources, expected, MRR_CUTOFF),
    }
    return result


def aggregate(results: list[ConfigCaseResult], cases_by_id: dict[str, dict]) -> dict[str, float]:
    scored = [r for r in results if not cases_by_id[r.case_id].get("should_abstain")]
    names = [
        *[f"hit_rate_at_{k}" for k in HIT_RATE_CUTOFFS],
        f"recall_at_{RECALL_CUTOFF}",
        f"mrr_at_{MRR_CUTOFF}",
    ]
    return {name: _mean([r.metrics.get(name, 0.0) for r in scored]) for name in names}


# --------------------------------------------------------------------------
# The comparison run
# --------------------------------------------------------------------------


def run_comparison(
    cases: list[dict[str, Any]] | None = None,
    retriever: Any | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    """Measure all four configurations and build the report payload."""
    from src.reranking import get_reranker
    from src.retrieval.bm25 import get_bm25_index
    from src.retrieval.embeddings import embed_query
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.vector_store import count_chunks, dense_search, source_policy_ids

    settings = get_settings()
    cases = cases if cases is not None else load_golden_dataset()
    retriever = retriever or get_hybrid_retriever()
    reranker = reranker or get_reranker()
    cases_by_id = {c["id"]: c for c in cases}
    policy_ids = source_policy_ids()

    results: dict[str, list[ConfigCaseResult]] = {name: [] for name in CONFIG_ORDER}
    errors: list[dict[str, str]] = []

    for case in cases:
        question = case["question"]
        logger.info("case %s", case["id"])

        # A. BM25 alone.
        bm25_hits = get_bm25_index().search(question, settings.sparse_top_k)
        results[CONFIG_A].append(score_case(CONFIG_A, case, bm25_hits))

        # B. pgvector alone.
        dense_hits = dense_search(embed_query(question), settings.dense_top_k)
        results[CONFIG_B].append(score_case(CONFIG_B, case, dense_hits))

        # C. The Phase B pipeline.
        fused = retriever.retrieve_with_diagnostics(question).results
        results[CONFIG_C].append(score_case(CONFIG_C, case, fused))

        # D. C's candidates, rescored. Never raises: a reranker failure degrades
        # to the RRF order and is recorded rather than aborting the run.
        reranked = reranker.rerank_with_diagnostics(question, fused, top_k=settings.rerank_top_k)
        results[CONFIG_D].append(
            score_case(
                CONFIG_D,
                case,
                reranked.chunks,
                reranker_used=reranked.reranker_used,
                failure=reranked.failure,
            )
        )
        if reranked.failure:
            errors.append(
                {
                    "case_id": case["id"],
                    "stage": reranked.failure_stage or "unknown",
                    "error": reranked.failure,
                }
            )

    metrics = {name: aggregate(results[name], cases_by_id) for name in CONFIG_ORDER}
    reranker_used = all(
        r.reranker_used for r in results[CONFIG_D] if r.reranker_used is not None
    ) and any(r.reranker_used for r in results[CONFIG_D])

    return {
        "report": "reranking_comparison",
        "phase": "C",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(resolve_dataset_path().relative_to(PROJECT_ROOT)),
            "version": golden_dataset_version(),
            "total_cases": len(cases),
            "scored_cases": sum(1 for c in cases if not c.get("should_abstain")),
            "abstention_cases_excluded": sum(1 for c in cases if c.get("should_abstain")),
        },
        "corpus": {
            "chunks_indexed": count_chunks(),
            "documents": len(policy_ids),
            "policy_ids": policy_ids,
        },
        "model": {
            "configured": settings.resolved_reranker_model,
            "loaded": getattr(reranker, "_loaded_model_name", None),
            "fallback_configured": settings.reranker_fallback_model,
            "fallback_used": getattr(reranker, "_loaded_model_name", None)
            not in (None, settings.resolved_reranker_model),
            "device": settings.model_device,
        },
        "model_configuration": reranker.config(),
        "retrieval_configurations": {
            CONFIG_A: {
                "description": "BM25 only, no fusion, no deduplication",
                "top_k": settings.sparse_top_k,
                "deduplication_applied": False,
            },
            CONFIG_B: {
                "description": "pgvector dense search only, no fusion, no deduplication",
                "top_k": settings.dense_top_k,
                "deduplication_applied": False,
            },
            CONFIG_C: {
                "description": "Phase B baseline: BM25 + dense, RRF, deduplication",
                "top_k": settings.fusion_top_k,
                "deduplication_applied": settings.dedup_enabled,
                **retriever.config(),
            },
            CONFIG_D: {
                "description": "Phase B candidates rescored by the cross-encoder",
                "candidate_top_k": settings.rerank_candidate_top_k,
                "top_k": settings.rerank_top_k,
                "deduplication_applied": settings.dedup_enabled,
            },
        },
        "metric_note": (
            "All metrics are cut at k <= 5. Configuration D returns 5 results by "
            "design, so a metric at k > 5 would measure truncation, not ranking."
        ),
        "reranker_used": reranker_used,
        "metrics": {
            name: {k: round(v, 4) for k, v in metrics[name].items()} for name in CONFIG_ORDER
        },
        "watched_cases": build_watched_cases(results, cases_by_id),
        "verdict": build_verdict(metrics, results, cases_by_id),
        "errors": errors,
        "cases": build_case_table(results, cases_by_id),
    }


def build_case_table(
    results: dict[str, list[ConfigCaseResult]], cases_by_id: dict[str, dict]
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for config in CONFIG_ORDER:
        for result in results[config]:
            case = cases_by_id[result.case_id]
            entry = by_case.setdefault(
                result.case_id,
                {
                    "case_id": result.case_id,
                    "question": case["question"],
                    "category": case.get("category", ""),
                    "difficulty": case.get("difficulty", ""),
                    "should_abstain": bool(case.get("should_abstain")),
                    "expected_sources": sorted(set(case["expected_sources"])),
                    "configs": {},
                },
            )
            entry["configs"][config] = result.to_dict()
    return [by_case[c] for c in cases_by_id if c in by_case]


def build_watched_cases(
    results: dict[str, list[ConfigCaseResult]], cases_by_id: dict[str, dict]
) -> dict[str, Any]:
    """Explicit before/after for the cases Phase B could not rank first."""
    by_config_case = {config: {r.case_id: r for r in results[config]} for config in CONFIG_ORDER}
    watched: dict[str, Any] = {}
    for case_id in WATCHED_CASES:
        if case_id not in cases_by_id:
            continue
        before = by_config_case[CONFIG_C].get(case_id)
        after = by_config_case[CONFIG_D].get(case_id)
        if before is None or after is None:
            continue
        watched[case_id] = {
            "question": cases_by_id[case_id]["question"],
            "expected_sources": sorted(set(cases_by_id[case_id]["expected_sources"])),
            "before": {
                "config": CONFIG_C,
                "first_relevant_rank": before.first_relevant_rank,
                "top_5_policy_ids": before.retrieved_policy_ids[:5],
                "metrics": {k: round(v, 4) for k, v in before.metrics.items()},
            },
            "after": {
                "config": CONFIG_D,
                "first_relevant_rank": after.first_relevant_rank,
                "top_5_policy_ids": after.retrieved_policy_ids[:5],
                "metrics": {k: round(v, 4) for k, v in after.metrics.items()},
                "reranker_used": after.reranker_used,
            },
            "rank_change": _rank_change(before.first_relevant_rank, after.first_relevant_rank),
        }
    return watched


def _rank_change(before: int | None, after: int | None) -> str:
    if before is None and after is None:
        return "absent in both"
    if before is None:
        return f"recovered: absent -> {after}"
    if after is None:
        return f"lost: {before} -> absent"
    if after < before:
        return f"improved: {before} -> {after}"
    if after > before:
        return f"regressed: {before} -> {after}"
    return f"unchanged: {before}"


def build_verdict(
    metrics: dict[str, dict[str, float]],
    results: dict[str, list[ConfigCaseResult]],
    cases_by_id: dict[str, dict],
) -> dict[str, Any]:
    """Did reranking earn its place? Running is not the same as helping."""
    mrr = f"mrr_at_{MRR_CUTOFF}"
    recall = f"recall_at_{RECALL_CUTOFF}"
    baseline, reranked = metrics[CONFIG_C], metrics[CONFIG_D]

    before = {r.case_id: r for r in results[CONFIG_C]}
    after = {r.case_id: r for r in results[CONFIG_D]}

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for case_id, before_result in before.items():
        if cases_by_id[case_id].get("should_abstain"):
            continue
        after_result = after[case_id]
        b, a = before_result.first_relevant_rank, after_result.first_relevant_rank
        entry = {"case_id": case_id, "before_rank": b, "after_rank": a}
        # Only ranks inside the reranked window are comparable: a case whose
        # correct source sat at rank 9 was never a candidate for the top 5.
        if b is not None and b <= 5 and (a is None or a > b):
            regressions.append(entry)
        elif b is not None and a is not None and a < b:
            improvements.append(entry)

    mrr_improved = reranked[mrr] > baseline[mrr]
    hit_maintained = reranked["hit_rate_at_5"] >= baseline["hit_rate_at_5"]
    recall_maintained = reranked[recall] >= baseline[recall]

    return {
        "mrr_at_5_improved": mrr_improved,
        "mrr_at_5_before": round(baseline[mrr], 4),
        "mrr_at_5_after": round(reranked[mrr], 4),
        "mrr_at_5_delta": round(reranked[mrr] - baseline[mrr], 4),
        "hit_rate_at_5_maintained": hit_maintained,
        "hit_rate_at_5_before": round(baseline["hit_rate_at_5"], 4),
        "hit_rate_at_5_after": round(reranked["hit_rate_at_5"], 4),
        "recall_at_5_maintained": recall_maintained,
        "recall_at_5_before": round(baseline[recall], 4),
        "recall_at_5_after": round(reranked[recall], 4),
        "hit_rate_at_1_before": round(baseline["hit_rate_at_1"], 4),
        "hit_rate_at_1_after": round(reranked["hit_rate_at_1"], 4),
        "improved_cases": improvements,
        "regressed_cases": regressions,
        "no_regressions": not regressions,
        "reranking_is_an_improvement": bool(
            mrr_improved and hit_maintained and recall_maintained and not regressions
        ),
    }


def compare_to_phase_b(payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Check configuration C against the committed Phase B baseline report.

    C is the Phase B pipeline, so its numbers should reproduce. A mismatch means
    something moved underneath the comparison and the run is not trustworthy.
    """
    path = path or BASELINE_REPORT_PATH
    if not path.exists():
        return {"available": False, "reason": f"{path} not found"}

    baseline = json.loads(path.read_text(encoding="utf-8"))
    measured = baseline.get("measured_metrics", {})
    current = payload["metrics"][CONFIG_C]
    shared = sorted(set(measured) & set(current))
    deltas = {k: round(current[k] - measured[k], 4) for k in shared}

    return {
        "available": True,
        "baseline_report": str(path.relative_to(PROJECT_ROOT)),
        "baseline_timestamp": baseline.get("timestamp"),
        "phase_b_metrics": {k: measured[k] for k in shared},
        "phase_c_hybrid_rrf_metrics": {k: current[k] for k in shared},
        "deltas": deltas,
        "hybrid_rrf_reproduces_phase_b": all(abs(d) < 1e-9 for d in deltas.values()),
    }


def save_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_table(payload: dict[str, Any]) -> None:
    names = [
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        f"recall_at_{RECALL_CUTOFF}",
        f"mrr_at_{MRR_CUTOFF}",
    ]
    header = f"{'metric':<16}" + "".join(f"{c:>20}" for c in CONFIG_ORDER)
    print(header)
    print("-" * len(header))
    for name in names:
        row = f"{name:<16}"
        for config in CONFIG_ORDER:
            row += f"{payload['metrics'][config][name]:>20.4f}"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval with and without reranking")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payload = run_comparison()
    payload["phase_b_comparison"] = compare_to_phase_b(payload)

    print("\n=== Retrieval configuration comparison (no LLM) ===")
    print(f"dataset : {payload['dataset']['version']}")
    print(
        f"cases   : {payload['dataset']['scored_cases']} scored, "
        f"{payload['dataset']['abstention_cases_excluded']} abstention case(s) excluded"
    )
    print(f"model   : {payload['model']['loaded']} (reranker_used={payload['reranker_used']})\n")
    _print_table(payload)

    print("\n--- watched cases ---")
    for case_id, watched in payload["watched_cases"].items():
        print(
            f"{case_id}: rank {watched['before']['first_relevant_rank']} -> "
            f"{watched['after']['first_relevant_rank']}  ({watched['rank_change']})"
        )

    verdict = payload["verdict"]
    print("\n--- verdict ---")
    print(
        f"MRR@5      {verdict['mrr_at_5_before']:.4f} -> {verdict['mrr_at_5_after']:.4f}  "
        f"(improved={verdict['mrr_at_5_improved']})"
    )
    print(
        f"HitRate@5  {verdict['hit_rate_at_5_before']:.4f} -> "
        f"{verdict['hit_rate_at_5_after']:.4f}  "
        f"(maintained={verdict['hit_rate_at_5_maintained']})"
    )
    print(
        f"Recall@5   {verdict['recall_at_5_before']:.4f} -> "
        f"{verdict['recall_at_5_after']:.4f}  "
        f"(maintained={verdict['recall_at_5_maintained']})"
    )
    print(f"regressions: {verdict['regressed_cases'] or 'none'}")
    print(f"improvement: {verdict['reranking_is_an_improvement']}")

    comparison = payload["phase_b_comparison"]
    if comparison.get("available"):
        print(
            f"\nhybrid_rrf reproduces Phase B baseline: "
            f"{comparison['hybrid_rrf_reproduces_phase_b']}"
        )
    if payload["errors"]:
        print(f"\nerrors: {payload['errors']}")

    if not args.no_save:
        print(f"\nReport written to {save_report(payload, args.output)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
