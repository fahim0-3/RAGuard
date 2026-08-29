"""Phase D: the deterministic golden evaluation.

Runs the 50-case golden dataset through retrieval and produces a **measured**
baseline. No LLM, no Ragas, no network beyond the local embedding model. Two
runs over unchanged code and an unchanged corpus produce identical metrics.

What is measured here, and what is not
--------------------------------------
Measured end-to-end: HitRate@1/3/5, Recall@5, Recall@10, MRR@5, keyword recall,
citation ID validity, and latency, over the 44 cases that expect retrieval.

Implemented and unit-tested but *not* measured here: structured output validity
and expected-outcome accuracy. Both describe what the system says, not what it
retrieves, so measuring them requires the LLM tier that Phase D deliberately
excludes. They are reported as `not_measured` with the reason attached rather
than filled with a plausible-looking number.

On Recall
---------
Recall@k only carries information beyond HitRate@k for cases that expect more
than one document. Eight of the 44 scored cases do. The report states this
explicitly, because a Recall figure computed over single-source cases is
arithmetically identical to HitRate and reporting it as independent evidence
would be misleading.

Run:  python -m src.evaluation.golden_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.deterministic_metrics import (
    citation_id_validity,
    find_regressions,
    latency_stats,
)
from src.evaluation.metrics import (
    golden_dataset_version,
    keyword_recall,
    load_baseline,
    load_golden_dataset,
    resolve_dataset_path,
)
from src.evaluation.retrieval_eval import (
    first_relevant_rank,
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.retrieval.types import RetrievedChunk
from src.timing import RETRIEVAL_STAGE_NAMES, sanitise_timing_samples

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "golden_baseline.json"

HIT_RATE_CUTOFFS = (1, 3, 5)
RECALL_CUTOFFS = (5, 10)
MRR_CUTOFF = 5

# Metrics that describe generated text rather than retrieved evidence.
NOT_MEASURED_IN_PHASE_D = {
    "structured_output_validity": (
        "Implemented in deterministic_metrics and unit-tested against fixtures. "
        "Measuring it end-to-end requires the LLM tier, which Phase D excludes."
    ),
    "expected_outcome_accuracy": (
        "Implemented in deterministic_metrics and unit-tested against fixtures. "
        "The retrieval tier does not decide answer/clarify/abstain/escalate, so "
        "measuring it end-to-end requires the LLM tier, which Phase D excludes."
    ),
}


@dataclass(slots=True)
class GoldenCaseResult:
    case_id: str
    question: str
    case_type: str
    category: str
    priority: str
    expected_outcome: str
    expected_sources: list[str]
    expected_policy_ids: list[str]
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_policy_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    first_relevant_rank: int | None = None
    keyword_recall: float | None = None
    missing_keywords: list[str] = field(default_factory=list)
    citation_id_validity: float | None = None
    invalid_policy_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_latency_ms: dict[str, float] = field(default_factory=dict)
    scored: bool = True
    failed: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    top_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "case_type": self.case_type,
            "category": self.category,
            "priority": self.priority,
            "expected_outcome": self.expected_outcome,
            "expected_sources": self.expected_sources,
            "expected_policy_ids": self.expected_policy_ids,
            "retrieved_policy_ids": self.retrieved_policy_ids[:10],
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "first_relevant_rank": self.first_relevant_rank,
            "keyword_recall": (
                round(self.keyword_recall, 4) if self.keyword_recall is not None else None
            ),
            "missing_keywords": self.missing_keywords,
            "citation_id_validity": self.citation_id_validity,
            "invalid_policy_ids": self.invalid_policy_ids,
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_latency_ms": {
                name: round(value, 3) for name, value in self.retrieval_latency_ms.items()
            },
            "scored": self.scored,
            "failed": self.failed,
            "failure_reasons": self.failure_reasons,
            "top_results": self.top_results,
        }


def _summarise(chunk: RetrievedChunk, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "policy_id": chunk.policy_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "bm25_score": round(chunk.sparse_score, 4) if chunk.sparse_score is not None else None,
        "vector_score": round(chunk.dense_score, 4) if chunk.dense_score is not None else None,
        "rrf_score": round(chunk.fusion_score, 6) if chunk.fusion_score is not None else None,
        "retriever_ranks": chunk.retriever_ranks,
    }


def evaluate_case(
    case: dict[str, Any],
    retriever: Any,
    known_policy_ids: set[str],
    diagnostic_depth: int = 5,
) -> GoldenCaseResult:
    started = time.perf_counter()
    if hasattr(retriever, "retrieve_with_diagnostics"):
        diagnostics = retriever.retrieve_with_diagnostics(case["question"])
        chunks = list(diagnostics.results)
        samples = sanitise_timing_samples(
            {name: [value] for name, value in diagnostics.timings_ms.items()},
            RETRIEVAL_STAGE_NAMES,
        )
        component_latency = {name: values[0] for name, values in samples.items()}
    else:
        chunks = retriever.retrieve(case["question"])
        component_latency = {}
    latency_ms = (time.perf_counter() - started) * 1000.0

    ranked_sources = [c.source for c in chunks]
    retrieved_policy_ids = [c.policy_id for c in chunks]
    expected = set(case["expected_sources"])

    result = GoldenCaseResult(
        case_id=case["id"],
        question=case["question"],
        case_type=case.get("case_type", ""),
        category=case.get("category", ""),
        priority=case.get("priority", ""),
        expected_outcome=case.get("expected_outcome", ""),
        expected_sources=sorted(expected),
        expected_policy_ids=sorted(case.get("expected_policy_ids", [])),
        retrieved_sources=ranked_sources,
        retrieved_policy_ids=retrieved_policy_ids,
        latency_ms=latency_ms,
        retrieval_latency_ms=component_latency,
        scored=not case.get("should_abstain", False),
        top_results=[
            _summarise(c, rank) for rank, c in enumerate(chunks[:diagnostic_depth], start=1)
        ],
    )

    # Citation validity applies to every case, including abstentions: a policy
    # ID that does not exist in the corpus is a defect regardless of outcome.
    cited = retrieved_policy_ids[:diagnostic_depth]
    result.citation_id_validity = citation_id_validity(cited, known_policy_ids)
    result.invalid_policy_ids = sorted({p for p in cited if p not in known_policy_ids})

    if not result.scored:
        return result

    result.metrics = {
        **{
            f"hit_rate_at_{k}": hit_rate_at_k(ranked_sources, expected, k) for k in HIT_RATE_CUTOFFS
        },
        **{f"recall_at_{k}": recall_at_k(ranked_sources, expected, k) for k in RECALL_CUTOFFS},
        f"mrr_at_{MRR_CUTOFF}": reciprocal_rank_at_k(ranked_sources, expected, MRR_CUTOFF),
    }
    result.first_relevant_rank = first_relevant_rank(ranked_sources, expected)

    context = "\n".join(c.content for c in chunks)
    expected_keywords = case.get("expected_keywords", [])
    if expected_keywords:
        result.keyword_recall = keyword_recall(context, expected_keywords)
        result.missing_keywords = [
            kw for kw in expected_keywords if kw.lower() not in context.lower()
        ]

    reasons: list[str] = []
    if result.metrics["hit_rate_at_5"] == 0.0:
        reasons.append("no expected source within top 5")
    if result.metrics["recall_at_10"] < 1.0:
        reasons.append("not all expected sources within top 10")
    if result.missing_keywords:
        reasons.append(f"keywords absent from retrieved context: {result.missing_keywords}")
    if result.invalid_policy_ids:
        reasons.append(f"retrieved unknown policy IDs: {result.invalid_policy_ids}")
    result.failure_reasons = reasons
    result.failed = bool(reasons)

    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(results: list[GoldenCaseResult]) -> dict[str, float]:
    scored = [r for r in results if r.scored]
    names = [
        *[f"hit_rate_at_{k}" for k in HIT_RATE_CUTOFFS],
        *[f"recall_at_{k}" for k in RECALL_CUTOFFS],
        f"mrr_at_{MRR_CUTOFF}",
    ]
    aggregated = {n: _mean([r.metrics.get(n, 0.0) for r in scored]) for n in names}

    keyword_scores = [r.keyword_recall for r in scored if r.keyword_recall is not None]
    aggregated["keyword_recall"] = _mean(keyword_scores)

    citation_scores = [
        r.citation_id_validity for r in results if r.citation_id_validity is not None
    ]
    aggregated["citation_id_validity"] = _mean(citation_scores)
    return aggregated


def run_golden_evaluation(
    cases: list[dict[str, Any]] | None = None, retriever: Any | None = None
) -> dict[str, Any]:
    """Measure retrieval over the golden dataset and build the report payload."""
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.vector_store import count_chunks, source_policy_ids

    settings = get_settings()
    cases = cases if cases is not None else load_golden_dataset()
    retriever = retriever or get_hybrid_retriever()
    policy_id_by_source = source_policy_ids()
    known_policy_ids = set(policy_id_by_source.values())

    # Warm the embedding model and the BM25 index before timing anything. The
    # first query otherwise absorbs a ~30 s one-time model load and reports it
    # as query latency, which is not a cost any user pays per question.
    retriever.retrieve("warm up the embedding model and the BM25 index")

    results = [evaluate_case(c, retriever, known_policy_ids) for c in cases]
    scored = [r for r in results if r.scored]
    failures = [r for r in scored if r.failed]
    measured = aggregate(results)

    baseline = load_baseline()
    targets = baseline.get("retrieval_v2", {})
    tolerance = float(baseline.get("tolerance", 0.0))
    regressions = find_regressions(measured, targets, tolerance)

    multi_source = [r for r in scored if len(r.expected_sources) > 1]
    latency_by_stage = {
        stage: latency_stats(
            [r.retrieval_latency_ms[stage] for r in results if stage in r.retrieval_latency_ms]
        ).to_dict()
        for stage in RETRIEVAL_STAGE_NAMES
        if any(stage in r.retrieval_latency_ms for r in results)
    }

    return {
        "report": "golden_baseline",
        "phase": "D",
        "measurement_status": "MEASURED",
        "_semantics": (
            "Every number under measured_metrics is an observation. Target "
            "thresholds live in src/evaluation/baseline.json and are never "
            "written by this script."
        ),
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(resolve_dataset_path().relative_to(PROJECT_ROOT)),
            "version": golden_dataset_version(),
            "total_cases": len(cases),
            "scored_cases": len(scored),
            "abstention_cases_excluded": len(cases) - len(scored),
            "case_type_distribution": dict(
                sorted(Counter(c.get("case_type", "") for c in cases).items())
            ),
            "expected_outcome_distribution": dict(
                sorted(Counter(c.get("expected_outcome", "") for c in cases).items())
            ),
            "category_distribution": dict(
                sorted(Counter(c.get("category", "") for c in cases).items())
            ),
            "priority_distribution": dict(
                sorted(Counter(c.get("priority", "") for c in cases).items())
            ),
        },
        "corpus": {
            "chunks_indexed": count_chunks(),
            "documents": len(policy_id_by_source),
            "policy_ids": policy_id_by_source,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "configuration": retriever.config(),
        "measured_metrics": {k: round(v, 4) for k, v in measured.items()},
        "recall_interpretation": {
            "scored_cases": len(scored),
            "cases_with_multiple_expected_sources": len(multi_source),
            "multi_source_case_ids": [r.case_id for r in multi_source],
            "note": (
                "Recall@k carries information beyond HitRate@k only for cases "
                "expecting more than one document. For the remaining "
                f"{len(scored) - len(multi_source)} single-source cases the two "
                "metrics are arithmetically identical."
            ),
        },
        "latency": {
            **latency_stats([r.latency_ms for r in results]).to_dict(),
            "stage": "retrieval only (BM25 + dense + RRF + deduplication)",
            "excludes": "one-time model loading, warmed before measurement",
            "reranker_applied": False,
            "by_stage": latency_by_stage,
        },
        "not_measured": NOT_MEASURED_IN_PHASE_D,
        "baseline_comparison": {
            "role": "diagnostic_only",
            "canonical_gate_source": "src/evaluation/gates.py :: RETRIEVAL_GATES",
            "note": (
                "The canonical merge verdict is produced by "
                "python -m src.evaluation.run_eval --retrieval --fail-on-regression. "
                "This historical comparison uses the global baseline tolerance and may be "
                "less strict than a structural canonical gate."
            ),
            "targets_source": "src/evaluation/baseline.json :: retrieval_v2",
            "targets": targets,
            "tolerance": tolerance,
            "regression_count": len(regressions),
            "regressions": [r.to_dict() for r in regressions],
        },
        "failure_summary": {
            "failed_case_count": len(failures),
            "failed_case_ids": [r.case_id for r in failures],
        },
        "cases": [r.to_dict() for r in results],
    }


def save_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic golden evaluation")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = run_golden_evaluation()

    print("\n=== Golden evaluation (MEASURED, deterministic, no LLM) ===")
    dataset = payload["dataset"]
    print(f"dataset : {dataset['version']}")
    print(
        f"cases   : {dataset['total_cases']} total, {dataset['scored_cases']} scored, "
        f"{dataset['abstention_cases_excluded']} abstention case(s) excluded"
    )
    print(
        f"corpus  : {payload['corpus']['chunks_indexed']} chunks, "
        f"{payload['corpus']['documents']} documents\n"
    )
    for name, value in payload["measured_metrics"].items():
        print(f"  {name:<22} {value:.4f}")

    latency = payload["latency"]
    print(
        f"\nlatency : mean {latency['mean_ms']:.0f} ms, p50 {latency['p50_ms']:.0f} ms, "
        f"p95 {latency['p95_ms']:.0f} ms, max {latency['max_ms']:.0f} ms"
    )

    recall = payload["recall_interpretation"]
    print(
        f"recall  : {recall['cases_with_multiple_expected_sources']} of "
        f"{recall['scored_cases']} scored cases expect multiple sources"
    )

    comparison = payload["baseline_comparison"]
    print(f"\ndiagnostic regressions vs targets: {comparison['regression_count']}")
    for regression in comparison["regressions"]:
        print(f"  {regression['metric']}: {regression['measured']} < {regression['target']}")

    failures = payload["failure_summary"]
    print(f"\nfailed cases: {failures['failed_case_count']} {failures['failed_case_ids']}")
    for case in payload["cases"]:
        if case["failed"]:
            print(f"  {case['case_id']} [{case['priority']}]: {'; '.join(case['failure_reasons'])}")

    if not args.no_save:
        print(f"\nReport written to {save_report(payload, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
