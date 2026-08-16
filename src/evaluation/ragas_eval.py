"""RAGAS evaluation, and an honest answer when RAGAS cannot run.

RAGAS is optional and, in this environment, currently unusable: version 0.4.3
(the latest on PyPI) imports `langchain_community.chat_models.vertexai` at
module load, and `langchain-community` 0.4.x removed that module during its
sunset. There is no newer RAGAS to upgrade to, and downgrading
`langchain-community` would drag `langchain-core` back from the 1.5 line that
Phases E through H are built on.

So this module probes availability and reports `RAGAS_NOT_AVAILABLE` with the
exact import error. It does not approximate the metrics with something simpler
and call them RAGAS scores: a number labelled "faithfulness" that a different
implementation produced is worse than no number, because it looks comparable to
published figures and is not.

The dataset adapter lives in `ragas_adapter` and is fully testable without
RAGAS installed, so the work that Phase I can validate is validated.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.evaluation.ragas_adapter import AdapterReport, build_ragas_samples

logger = logging.getLogger(__name__)

__all__ = [
    "RAGAS_METRIC_PLAN",
    "RagasAvailability",
    "check_ragas_available",
    "run_ragas_evaluation",
]

#: Which metrics are meaningful for RAGuard's data, and why. Chosen rather than
#: run wholesale: two of the four common metrics need a reference answer, which
#: only the golden dataset's hand-written `ground_truth` provides.
RAGAS_METRIC_PLAN: list[dict[str, str]] = [
    {
        "metric": "faithfulness",
        "purpose": "Is every claim in the answer entailed by the retrieved contexts?",
        "requires": "answer + contexts",
        "meaningful_for_raguard": "yes — overlaps Phase G entailment, so it cross-checks it",
    },
    {
        "metric": "answer_relevancy",
        "purpose": "Does the answer address the question actually asked?",
        "requires": "answer + question + embeddings",
        "meaningful_for_raguard": "yes — nothing else in the system measures this",
    },
    {
        "metric": "context_precision",
        "purpose": "Are the retrieved contexts relevant, or padded?",
        "requires": "question + contexts + reference",
        "meaningful_for_raguard": "yes — complements HitRate, which ignores rank quality",
    },
    {
        "metric": "context_recall",
        "purpose": "Did retrieval find everything the reference answer needs?",
        "requires": "contexts + reference",
        "meaningful_for_raguard": (
            "partly — the 22-chunk corpus keeps recall near ceiling, so a high "
            "score is weak evidence"
        ),
    },
]


class RagasAvailability:
    """Result of probing for a usable RAGAS installation."""

    def __init__(self, available: bool, version: str | None, reason: str | None) -> None:
        self.available = available
        self.version = version
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }


def check_ragas_available() -> RagasAvailability:
    """Probe RAGAS. Import failure is a reportable state, not an exception."""
    try:
        import importlib.metadata as metadata

        version = metadata.version("ragas")
    except Exception:
        return RagasAvailability(False, None, "ragas is not installed")

    try:
        import ragas  # noqa: F401
        from ragas import evaluate  # noqa: F401
    except Exception as exc:
        return RagasAvailability(
            False,
            version,
            f"ragas {version} is installed but fails to import: {type(exc).__name__}: {exc}",
        )

    return RagasAvailability(True, version, None)


def run_ragas_evaluation(
    records: list[dict[str, Any]],
    include_non_answers: bool = False,
) -> dict[str, Any]:
    """Run RAGAS if it is usable; otherwise report why it is not.

    The adapter runs either way, so the dataset that *would* be evaluated is
    reported even when the library cannot execute. That is the difference
    between "we could not measure this" and "we did not look".
    """
    started = time.perf_counter()
    availability = check_ragas_available()
    adapter: AdapterReport = build_ragas_samples(records, include_non_answers)

    base: dict[str, Any] = {
        "report": "ragas_evaluation",
        "availability": availability.to_dict(),
        "metric_plan": RAGAS_METRIC_PLAN,
        "dataset": adapter.to_dict(),
    }

    if not availability.available:
        logger.warning("RAGAS unavailable: %s", availability.reason)
        return {
            **base,
            "status": "RAGAS_NOT_AVAILABLE",
            "metrics": {},
            "metric_status": {m["metric"]: "UNVALIDATED" for m in RAGAS_METRIC_PLAN},
            "remediation": (
                "ragas 0.4.3 imports langchain_community.chat_models.vertexai, removed "
                "in langchain-community 0.4.x. No newer ragas exists. Resolving this "
                "requires either an upstream ragas fix or pinning the langchain stack "
                "below 0.4/1.5, which would affect Phases E-H."
            ),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    if not adapter.samples:
        return {
            **base,
            "status": "NO_USABLE_SAMPLES",
            "metrics": {},
            "metric_status": {m["metric"]: "UNVALIDATED" for m in RAGAS_METRIC_PLAN},
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    # Reached only where RAGAS imports cleanly. Kept thin on purpose: this path
    # has never executed in this environment, so it must not pretend otherwise.
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy

        from src.generation.llm_factory import get_chat_model

        dataset = EvaluationDataset.from_list([s.to_dict() for s in adapter.samples])
        judge = LangchainLLMWrapper(get_chat_model("judge"))
        result = evaluate(dataset=dataset, metrics=[Faithfulness(), ResponseRelevancy()], llm=judge)
        scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, int | float)}
        return {
            **base,
            "status": "MEASURED",
            "metrics": scores,
            "metric_status": {k: "MEASURED" for k in scores},
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
    except Exception as exc:
        logger.exception("RAGAS execution failed")
        return {
            **base,
            "status": "RAGAS_EXECUTION_FAILED",
            "metrics": {},
            "metric_status": {m["metric"]: "UNVALIDATED" for m in RAGAS_METRIC_PLAN},
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
