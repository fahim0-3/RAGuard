"""Phase B: measured retrieval baseline.

This module evaluates retrieval *alone*: BM25 + dense + RRF + deduplication,
with no cross-encoder and no LLM. That isolation is the point. A number that
depends on a reranker cannot tell you whether retrieval improved, and a number
that depends on an LLM cannot be reproduced.

Metrics, all computed over the golden cases that expect an answer (abstention
cases have no correct passage and are reported separately):

- **HitRate@k**   fraction of cases with at least one expected source in top k
- **Recall@k**    mean over cases of |retrieved ∩ expected| / |expected|
- **MRR@5**       mean of 1 / rank of the first expected source within top 5

Nothing here writes a threshold. Measured values go to the report artifact;
target thresholds live in baseline.json and are edited only by a human.

Run:  python -m src.evaluation.retrieval_eval
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
    keyword_recall,
    load_golden_dataset,
    resolve_dataset_path,
)
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_baseline.json"

HIT_RATE_CUTOFFS = (1, 3, 5)
RECALL_CUTOFFS = (5, 10)
MRR_CUTOFF = 5


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


def hit_rate_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    """1.0 when any expected source appears within the first k results."""
    return 1.0 if set(ranked_sources[:k]) & expected else 0.0


def recall_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    """Fraction of the expected sources present within the first k results."""
    if not expected:
        return 0.0
    return len(set(ranked_sources[:k]) & expected) / len(expected)


def first_relevant_rank(ranked_sources: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first expected source, or None if absent."""
    for rank, source in enumerate(ranked_sources, start=1):
        if source in expected:
            return rank
    return None


def reciprocal_rank_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    rank = first_relevant_rank(ranked_sources, expected)
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Per-case evaluation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CaseResult:
    case_id: str
    question: str
    category: str
    difficulty: str
    should_abstain: bool
    expected_sources: list[str]
    expected_policy_ids: list[str]
    retrieved_sources: list[str]
    retrieved_policy_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    first_relevant_rank: int | None = None
    expected_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    keyword_recall: float = 0.0
    top_results: list[dict[str, Any]] = field(default_factory=list)
    dropped_by_deduplication: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "should_abstain": self.should_abstain,
            "expected_sources": self.expected_sources,
            "expected_policy_ids": self.expected_policy_ids,
            "retrieved_sources": self.retrieved_sources,
            "retrieved_policy_ids": self.retrieved_policy_ids,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "first_relevant_rank": self.first_relevant_rank,
            "expected_keywords": self.expected_keywords,
            "missing_keywords": self.missing_keywords,
            "keyword_recall": round(self.keyword_recall, 4),
            "failed": self.failed,
            "failure_reasons": self.failure_reasons,
            "top_results": self.top_results,
            "dropped_by_deduplication": self.dropped_by_deduplication,
        }


def _summarise_chunk(chunk: RetrievedChunk, rank: int) -> dict[str, Any]:
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
        "content_preview": chunk.content[:250],
    }


def evaluate_case(
    case: dict[str, Any],
    retriever: Any,
    policy_ids: dict[str, str],
    diagnostic_depth: int = 5,
) -> CaseResult:
    diagnostics = retriever.retrieve_with_diagnostics(case["question"])
    chunks = diagnostics.results

    ranked_sources = [c.source for c in chunks]
    expected = set(case["expected_sources"])
    context_text = "\n".join(c.content for c in chunks)

    result = CaseResult(
        case_id=case["id"],
        question=case["question"],
        category=case.get("category", ""),
        difficulty=case.get("difficulty", ""),
        should_abstain=bool(case.get("should_abstain")),
        expected_sources=sorted(expected),
        expected_policy_ids=sorted({policy_ids.get(s, s) for s in expected}),
        retrieved_sources=ranked_sources,
        retrieved_policy_ids=[c.policy_id for c in chunks],
        expected_keywords=case.get("expected_keywords", []),
        top_results=[
            _summarise_chunk(chunk, rank)
            for rank, chunk in enumerate(chunks[:diagnostic_depth], start=1)
        ],
        dropped_by_deduplication=(
            [d.to_dict() for d in diagnostics.deduplication.dropped]
            if diagnostics.deduplication
            else []
        ),
    )

    if result.should_abstain:
        # No passage is correct here. Recorded for diagnostics only; these cases
        # are excluded from the aggregate metrics.
        return result

    result.metrics = {
        **{f"hit_rate_at_{k}": hit_rate_at_k(ranked_sources, expected, k) for k in HIT_RATE_CUTOFFS},
        **{f"recall_at_{k}": recall_at_k(ranked_sources, expected, k) for k in RECALL_CUTOFFS},
        f"mrr_at_{MRR_CUTOFF}": reciprocal_rank_at_k(ranked_sources, expected, MRR_CUTOFF),
    }
    result.first_relevant_rank = first_relevant_rank(ranked_sources, expected)
    result.keyword_recall = keyword_recall(context_text, result.expected_keywords)
    result.missing_keywords = [
        kw for kw in result.expected_keywords if kw.lower() not in context_text.lower()
    ]

    reasons: list[str] = []
    if result.metrics["hit_rate_at_5"] == 0.0:
        reasons.append("no expected source within top 5")
    if result.metrics["recall_at_10"] < 1.0:
        reasons.append("not all expected sources within top 10")
    if result.missing_keywords:
        reasons.append(f"keywords absent from retrieved context: {result.missing_keywords}")
    result.failure_reasons = reasons
    result.failed = bool(reasons)

    return result


# --------------------------------------------------------------------------
# Aggregate evaluation
# --------------------------------------------------------------------------


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    scored = [r for r in results if not r.should_abstain]
    metric_names = [
        *[f"hit_rate_at_{k}" for k in HIT_RATE_CUTOFFS],
        *[f"recall_at_{k}" for k in RECALL_CUTOFFS],
        f"mrr_at_{MRR_CUTOFF}",
    ]
    aggregated = {
        name: _mean([r.metrics.get(name, 0.0) for r in scored]) for name in metric_names
    }
    aggregated["keyword_recall"] = _mean([r.keyword_recall for r in scored])
    return aggregated


def evaluate_retrieval_baseline(
    cases: list[dict[str, Any]] | None = None,
    retriever: Any | None = None,
) -> dict[str, Any]:
    """Measure retrieval over the golden dataset and build the report payload."""
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.retrieval.vector_store import count_chunks, source_policy_ids

    settings = get_settings()
    cases = cases if cases is not None else load_golden_dataset()
    retriever = retriever or get_hybrid_retriever()
    policy_ids = source_policy_ids()

    results = [evaluate_case(case, retriever, policy_ids) for case in cases]
    scored = [r for r in results if not r.should_abstain]
    failures = [r for r in scored if r.failed]

    return {
        "report": "retrieval_baseline",
        "phase": "B",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(resolve_dataset_path().relative_to(PROJECT_ROOT)),
            "version": golden_dataset_version(),
            "total_cases": len(cases),
            "scored_cases": len(scored),
            "abstention_cases_excluded": len(cases) - len(scored),
        },
        "corpus": {
            "chunks_indexed": count_chunks(),
            "documents": len(policy_ids),
            "policy_ids": policy_ids,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "configuration": retriever.config(),
        "measured_metrics": {k: round(v, 4) for k, v in aggregate(results).items()},
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the retrieval baseline")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payload = evaluate_retrieval_baseline()

    print("\n=== Retrieval baseline (measured, no reranker, no LLM) ===")
    print(f"dataset  : {payload['dataset']['version']}")
    print(
        f"cases    : {payload['dataset']['scored_cases']} scored, "
        f"{payload['dataset']['abstention_cases_excluded']} abstention case(s) excluded"
    )
    print(f"corpus   : {payload['corpus']['chunks_indexed']} chunks, "
          f"{payload['corpus']['documents']} documents\n")
    for name, value in payload["measured_metrics"].items():
        print(f"  {name:<20} {value:.4f}")

    failures = payload["failure_summary"]
    print(f"\nfailed cases: {failures['failed_case_count']} {failures['failed_case_ids']}")
    for case in payload["cases"]:
        if case["failed"]:
            print(f"  {case['case_id']}: {'; '.join(case['failure_reasons'])}")

    if not args.no_save:
        path = save_report(payload, args.output)
        print(f"\nReport written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
