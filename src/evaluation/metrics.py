"""Evaluation metrics for the golden dataset.

Two tiers, deliberately separated.

**Tier 1, deterministic (blocks merges).** Retrieval-only metrics computed with
no LLM call: hit rate, MRR, exact-keyword recall, mean confidence. Identical
input produces an identical number on every run, which is the only property
that makes a merge gate trustworthy.

**Tier 2, LLM-judged (reports, does not block).** End-to-end behaviour and the
Ragas suite. These consume provider quota and vary between runs, so they run
nightly and are treated as evidence, not as a gate. Blocking a merge on a
metric that can differ on identical code trains the team to ignore the gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
BASELINE_PATH = Path(__file__).with_name("baseline.json")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def resolve_dataset_path(path: Path | None = None) -> Path:
    """Explicit argument wins, then GOLDEN_DATASET_PATH, then the packaged file."""
    if path is not None:
        return path
    from src.config import get_settings

    configured = get_settings().absolute_golden_dataset_path
    return configured if configured.exists() else GOLDEN_DATASET_PATH


def load_golden_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    data = json.loads(resolve_dataset_path(path).read_text(encoding="utf-8"))
    return data["cases"]


def golden_dataset_version(path: Path | None = None) -> str:
    data = json.loads(resolve_dataset_path(path).read_text(encoding="utf-8"))
    return data.get("version", "unknown")


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or BASELINE_PATH).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Primitive metrics
# --------------------------------------------------------------------------


def hit_at_k(chunks: list[RetrievedChunk], expected_sources: list[str], k: int) -> float:
    """1.0 when any expected source appears in the top k passages."""
    if not expected_sources:
        return 0.0
    retrieved = {c.source for c in chunks[:k]}
    return 1.0 if retrieved & set(expected_sources) else 0.0


def reciprocal_rank(chunks: list[RetrievedChunk], expected_sources: list[str]) -> float:
    """1 / rank of the first passage from an expected source."""
    if not expected_sources:
        return 0.0
    expected = set(expected_sources)
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.source in expected:
            return 1.0 / rank
    return 0.0


def keyword_recall(text: str, expected_keywords: list[str]) -> float:
    """Fraction of required literal strings present, case-insensitively.

    Exact-string recall is the metric that catches the failure the project set
    out to fix: an answer that is fluent but silently drops `PAY-402` or
    changes `14-day` to `30-day`.
    """
    if not expected_keywords:
        return 1.0
    haystack = text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in haystack)
    return found / len(expected_keywords)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationReport:
    tier: str
    metrics: dict[str, float]
    cases: list[dict[str, Any]] = field(default_factory=list)
    dataset_version: str = ""
    generated_at: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "generated_at": self.generated_at or datetime.now(UTC).isoformat(),
            "dataset_version": self.dataset_version or golden_dataset_version(),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "context": self.context,
            "cases": self.cases,
        }

    def save(self, directory: Path | None = None) -> Path:
        """Write the report using the dated naming convention."""
        directory = directory or (PROJECT_ROOT / "reports")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        path = directory / f"{stamp}_raguard-eval-{self.tier}_v1.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------
# Tier 1: deterministic retrieval evaluation (no LLM)
# --------------------------------------------------------------------------


def evaluate_retrieval(
    cases: list[dict[str, Any]] | None = None, k: int = 5
) -> EvaluationReport:
    from src.self_healing.pipeline import get_pipeline

    cases = cases if cases is not None else load_golden_dataset()
    pipeline = get_pipeline()

    per_case: list[dict[str, Any]] = []
    hits: list[float] = []
    rrs: list[float] = []
    kw_recalls: list[float] = []
    confidences: list[float] = []
    oos_confidences: list[float] = []

    for case in cases:
        chunks, confidence = pipeline.retrieve_only(case["question"])
        context_text = "\n".join(c.content for c in chunks)

        record = {
            "id": case["id"],
            "question": case["question"],
            "category": case.get("category"),
            "retrieved_sources": [c.source for c in chunks],
            "top_confidence": round(confidence.top_score, 4),
            "confidence_level": confidence.level,
        }

        if case.get("should_abstain"):
            # Out-of-scope cases have no correct passage. What is measured is
            # that confidence stays low enough for the pipeline to refuse.
            oos_confidences.append(confidence.top_score)
            record["expected"] = "abstain"
            per_case.append(record)
            continue

        hit = hit_at_k(chunks, case["expected_sources"], k)
        rr = reciprocal_rank(chunks, case["expected_sources"])
        kw = keyword_recall(context_text, case.get("expected_keywords", []))

        hits.append(hit)
        rrs.append(rr)
        kw_recalls.append(kw)
        confidences.append(confidence.top_score)

        record.update(
            {
                "expected_sources": case["expected_sources"],
                f"hit_at_{k}": hit,
                "reciprocal_rank": round(rr, 4),
                "keyword_recall": round(kw, 4),
                "missing_keywords": [
                    kw_str
                    for kw_str in case.get("expected_keywords", [])
                    if kw_str.lower() not in context_text.lower()
                ],
            }
        )
        per_case.append(record)

    return EvaluationReport(
        tier="retrieval",
        metrics={
            f"hit_rate_at_{k}": _mean(hits),
            f"mrr_at_{k}": _mean(rrs),
            "keyword_recall": _mean(kw_recalls),
            "mean_top_confidence": _mean(confidences),
            "oos_mean_top_confidence": _mean(oos_confidences),
        },
        cases=per_case,
        context={"k": k, "llm_used": False},
    )


# --------------------------------------------------------------------------
# Tier 2: end-to-end behaviour (consumes LLM quota)
# --------------------------------------------------------------------------


def evaluate_end_to_end(
    cases: list[dict[str, Any]] | None = None,
    use_llm_verification: bool = False,
) -> EvaluationReport:
    from src.self_healing.pipeline import get_pipeline

    cases = cases if cases is not None else load_golden_dataset()
    pipeline = get_pipeline()

    per_case: list[dict[str, Any]] = []
    abstention_correct: list[float] = []
    answer_rates: list[float] = []
    citation_validity: list[float] = []
    answer_kw: list[float] = []
    healing_used: list[float] = []

    for case in cases:
        response = pipeline.answer(
            case["question"], use_llm_verification=use_llm_verification
        )
        should_abstain = bool(case.get("should_abstain"))

        abstention_correct.append(1.0 if response.abstained == should_abstain else 0.0)
        healing_used.append(1.0 if response.healing_attempts else 0.0)

        record = {
            "id": case["id"],
            "question": case["question"],
            "abstained": response.abstained,
            "expected_abstain": should_abstain,
            "abstain_reason": response.abstain_reason,
            "healing_attempts": response.healing_attempts,
            "latency_ms": round(response.latency_ms, 1),
            "answer": response.answer,
        }

        if not should_abstain:
            answered = 0.0 if response.abstained else 1.0
            answer_rates.append(answered)
            if not response.abstained:
                cited_sources = {c["source"] for c in response.citations}
                valid = 1.0 if cited_sources & set(case["expected_sources"]) else 0.0
                citation_validity.append(valid)
                answer_kw.append(
                    keyword_recall(response.answer, case.get("expected_keywords", []))
                )
                record["cited_sources"] = sorted(cited_sources)
                record["citation_matches_expected"] = bool(valid)

        per_case.append(record)

    return EvaluationReport(
        tier="end-to-end",
        metrics={
            "abstention_accuracy": _mean(abstention_correct),
            "answer_rate": _mean(answer_rates),
            "citation_validity": _mean(citation_validity),
            "answer_keyword_recall": _mean(answer_kw),
            "healing_trigger_rate": _mean(healing_used),
        },
        cases=per_case,
        context={"llm_used": True, "llm_citation_verification": use_llm_verification},
    )


# --------------------------------------------------------------------------
# Tier 2: Ragas
# --------------------------------------------------------------------------


class _BGEEmbeddingsAdapter:
    """LangChain Embeddings interface over the already-loaded BGE-M3 model.

    Ragas needs an embedding model for answer relevancy and semantic similarity.
    Reusing the retrieval embedder keeps the judge consistent with the system
    under test and avoids loading a second multi-gigabyte model.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.retrieval.embeddings import embed_texts

        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        from src.retrieval.embeddings import embed_query

        return embed_query(text)


def run_ragas(cases: list[dict[str, Any]] | None = None) -> EvaluationReport:
    """Run the Ragas suite over end-to-end pipeline output.

    Kept out of the merge gate on purpose: these metrics are LLM-judged and
    therefore non-deterministic.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Ragas is not installed or its API has moved. Run: "
            "uv sync --locked --group evaluation"
        ) from exc

    from src.generation.llm_provider import get_chat_model
    from src.self_healing.pipeline import get_pipeline

    cases = cases if cases is not None else load_golden_dataset()
    pipeline = get_pipeline()

    samples = []
    for case in cases:
        if case.get("should_abstain"):
            # Ragas faithfulness is undefined for an intentional refusal.
            continue
        response = pipeline.answer(case["question"])
        samples.append(
            {
                "user_input": case["question"],
                "response": response.answer,
                "retrieved_contexts": response.contexts or [""],
                "reference": case["ground_truth"],
            }
        )

    judge = LangchainLLMWrapper(get_chat_model("judge"))
    embeddings = LangchainEmbeddingsWrapper(_BGEEmbeddingsAdapter())

    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=embeddings,
    )

    scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, int | float)}
    return EvaluationReport(
        tier="ragas",
        metrics=scores,
        cases=samples,
        context={"judge_llm": type(judge).__name__, "sample_count": len(samples)},
    )


# --------------------------------------------------------------------------
# Regression gate
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Regression:
    metric: str
    actual: float
    threshold: float

    def __str__(self) -> str:
        return (
            f"{self.metric}: {self.actual:.4f} < {self.threshold:.4f} "
            f"(shortfall {self.threshold - self.actual:.4f})"
        )


def compare_to_baseline(
    report: EvaluationReport,
    baseline_section: str,
    baseline: dict[str, Any] | None = None,
) -> list[Regression]:
    """Return every metric that fell below its baseline, tolerance included."""
    baseline = baseline or load_baseline()
    thresholds: dict[str, float] = baseline.get(baseline_section, {})
    tolerance = float(baseline.get("tolerance", 0.0))

    regressions: list[Regression] = []
    for metric, threshold in thresholds.items():
        if metric not in report.metrics:
            logger.warning("Baseline metric %r absent from report", metric)
            continue
        actual = report.metrics[metric]
        if actual < threshold - tolerance:
            regressions.append(Regression(metric, actual, threshold))
    return regressions
