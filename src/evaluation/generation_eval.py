"""End-to-end generation evaluation: the whole workflow, per golden case.

Runs the real graph — sanitize, risk routing, ambiguity, retrieval, reranking,
evidence grading, generation, citation verification — and compares the terminal
outcome against what the golden case says should happen.

The central rule is that a non-answer is not automatically a failure. A case
marked `abstain` that abstains is a *success*, and an evaluator that scores it
as a miss would push the system toward answering everything, which is the
failure mode this project exists to prevent. Outcomes are therefore compared
against expectations, not against the string "answer".

Infrastructure failures are kept distinct from behavioural ones. A provider
outage produces `provider_error` and is reported as `BLOCKED`, never folded
into an accuracy figure, because a metric that improves when the LLM is down is
not measuring quality.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.deterministic_metrics import latency_stats
from src.evaluation.metrics import golden_dataset_version, load_golden_dataset
from src.generation.llm_factory import model_name_for
from src.generation.llm_routing import select_route, workload_context
from src.timing import (
    GRAPH_STAGE_NAMES,
    RETRIEVAL_STAGE_NAMES,
    aggregate_timing_samples,
    sanitise_timing_samples,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationCaseResult",
    "evaluate_generation",
]

EVALUATION_VERSION = "phase_i_v1"

#: Outcomes that mean the workflow ran but could not produce an answer for an
#: infrastructure reason. Excluded from accuracy, reported separately.
INFRASTRUCTURE_OUTCOMES = frozenset({"error"})


@dataclass
class GenerationCaseResult:
    case_id: str
    question: str
    expected_outcome: str
    actual_outcome: str
    outcome_matched: bool
    priority: str = ""
    case_type: str = ""
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    expected_policy_ids: list[str] = field(default_factory=list)
    cited_policy_ids: list[str] = field(default_factory=list)
    invalid_citation_ids: list[str] = field(default_factory=list)
    verification_status: str = "not_checked"
    structured_output_valid: bool = True
    citation_complete: bool = True
    retry_count: int = 0
    latency_ms: float = 0.0
    failure_reason: str | None = None
    infrastructure_failure: bool = False
    stage_latency_samples_ms: dict[str, list[float]] = field(default_factory=dict)
    retrieval_latency_samples_ms: dict[str, list[float]] = field(default_factory=dict)
    #: Carried through for the RAGAS adapter, which needs the reference answer
    #: and the passages the model actually saw. Without these the adapter has
    #: to exclude every case, and RAGAS silently evaluates nothing.
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "outcome_matched": self.outcome_matched,
            "priority": self.priority,
            "case_type": self.case_type,
            "citations": self.citations,
            "expected_policy_ids": self.expected_policy_ids,
            "cited_policy_ids": self.cited_policy_ids,
            "invalid_citation_ids": self.invalid_citation_ids,
            "verification_status": self.verification_status,
            "structured_output_valid": self.structured_output_valid,
            "citation_complete": self.citation_complete,
            "retry_count": self.retry_count,
            "latency_ms": round(self.latency_ms, 1),
            "failure_reason": self.failure_reason,
            "infrastructure_failure": self.infrastructure_failure,
            "stage_latency_ms": aggregate_timing_samples(self.stage_latency_samples_ms),
            "retrieval_latency_ms": aggregate_timing_samples(self.retrieval_latency_samples_ms),
            "answer_preview": self.answer[:200],
            # Keys the RAGAS adapter reads. Added alongside the originals
            # rather than renaming them, so existing consumers keep working.
            "outcome": self.actual_outcome,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


def _expected_outcome(case: dict[str, Any]) -> str:
    """The golden dataset's expectation, tolerating the older field name."""
    return str(
        case.get("expected_outcome") or ("abstain" if case.get("should_abstain") else "answer")
    )


def evaluate_case(
    case: dict[str, Any], service: Any, known_policy_ids: set[str]
) -> GenerationCaseResult:
    started = time.perf_counter()
    expected = _expected_outcome(case)
    request_id = f"eval-{case.get('id', uuid.uuid4().hex[:8])}"

    try:
        state = service.invoke(case["question"], request_id=request_id)
    except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
        logger.exception("Case %s raised", case.get("id"))
        return GenerationCaseResult(
            case_id=str(case.get("id", "")),
            question=case["question"],
            expected_outcome=expected,
            actual_outcome="error",
            outcome_matched=False,
            priority=str(case.get("priority", "")),
            case_type=str(case.get("case_type", "")),
            # Reports become CI artifacts. The full exception stays in the
            # log, while the report keeps only a stable diagnostic category.
            failure_reason=f"evaluation_exception:{type(exc).__name__}",
            infrastructure_failure=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    from src.self_healing.graph import summarise

    summary = summarise(state)
    actual = str(summary.get("final_outcome") or "error")
    citations = list(summary.get("citations") or [])
    verification = summary.get("verification_result") or {}

    supplied = {c.citation_label: c for c in (state.get("retrieved_chunks") or [])}
    cited_policy_ids = [supplied[label].policy_id for label in citations if label in supplied]
    invalid = [label for label in citations if label not in supplied]
    # A cited policy outside the corpus is a fabrication regardless of label form.
    invalid += [pid for pid in cited_policy_ids if pid not in known_policy_ids]

    # A provider outage surfaces as an abstention with this reason; it is an
    # infrastructure failure, not the system choosing to refuse.
    abstain_reason = str(summary.get("abstain_reason") or "")
    infrastructure = abstain_reason == "provider_error" or actual in INFRASTRUCTURE_OUTCOMES

    return GenerationCaseResult(
        case_id=str(case.get("id", "")),
        question=case["question"],
        expected_outcome=expected,
        actual_outcome=actual,
        outcome_matched=(actual == expected),
        priority=str(case.get("priority", "")),
        case_type=str(case.get("case_type", "")),
        answer=str(summary.get("final_answer") or ""),
        citations=citations,
        expected_policy_ids=list(case.get("expected_policy_ids") or []),
        cited_policy_ids=cited_policy_ids,
        invalid_citation_ids=invalid,
        verification_status=(
            "supported"
            if verification.get("supported")
            else "unsupported"
            if verification.get("checked")
            else "not_checked"
        ),
        # Phase E guarantees a validated envelope; a missing outcome means the
        # contract itself broke.
        structured_output_valid=bool(summary.get("final_outcome")),
        citation_complete=(actual != "answer" or bool(citations)),
        retry_count=int(summary.get("retry_count") or 0),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        failure_reason=summary.get("failure_reason") or None,
        infrastructure_failure=infrastructure,
        stage_latency_samples_ms=sanitise_timing_samples(
            summary.get("stage_latency_samples_ms"), GRAPH_STAGE_NAMES
        ),
        retrieval_latency_samples_ms=sanitise_timing_samples(
            summary.get("retrieval_latency_samples_ms"), RETRIEVAL_STAGE_NAMES
        ),
        # The hand-written reference, never the model's own answer.
        ground_truth=str(case.get("ground_truth") or ""),
        # The passages the model actually saw, which is what faithfulness and
        # context recall have to be measured against.
        contexts=[c.content for c in (state.get("retrieved_chunks") or [])],
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_generation(
    cases: list[dict[str, Any]] | None = None,
    service: Any | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run every golden case through the workflow and score the outcomes."""
    from src.retrieval.vector_store import source_policy_ids
    from src.self_healing.graph import SelfHealingGraph

    settings = get_settings()
    evaluation_provider = select_route(settings, "evaluation").provider
    cases = cases if cases is not None else load_golden_dataset()
    if limit:
        cases = cases[:limit]
    service = service or SelfHealingGraph()
    known_policy_ids = set(source_policy_ids().values())

    started = datetime.now(UTC)
    # Evaluation is an explicit routing workload, not a property inferred from
    # a golden-case question. Dynamic routing therefore chooses Groq once per
    # graph run while static mode remains entirely manual.
    with workload_context("evaluation"):
        results = [evaluate_case(case, service, known_policy_ids) for case in cases]

    scored = [r for r in results if not r.infrastructure_failure]
    blocked = [r for r in results if r.infrastructure_failure]
    answered = [r for r in scored if r.actual_outcome == "answer"]
    latency_by_stage = {
        stage: latency_stats(
            [
                sample
                for result in results
                for sample in result.stage_latency_samples_ms.get(stage, [])
            ]
        ).to_dict()
        for stage in GRAPH_STAGE_NAMES
        if any(stage in result.stage_latency_samples_ms for result in results)
    }
    latency_by_retrieval_component = {
        stage: latency_stats(
            [
                sample
                for result in results
                for sample in result.retrieval_latency_samples_ms.get(stage, [])
            ]
        ).to_dict()
        for stage in RETRIEVAL_STAGE_NAMES
        if any(stage in result.retrieval_latency_samples_ms for result in results)
    }

    by_expected: dict[str, dict[str, int]] = {}
    for result in scored:
        bucket = by_expected.setdefault(result.expected_outcome, {"total": 0, "matched": 0})
        bucket["total"] += 1
        bucket["matched"] += int(result.outcome_matched)

    metrics = {
        "expected_outcome_accuracy": _rate(
            sum(1 for r in scored if r.outcome_matched), len(scored)
        ),
        "structured_output_validity": _rate(
            sum(1 for r in scored if r.structured_output_valid), len(scored)
        ),
        "citation_id_validity": _rate(
            sum(1 for r in answered if not r.invalid_citation_ids), len(answered)
        )
        if answered
        else None,
        "citation_completeness": _rate(sum(1 for r in scored if r.citation_complete), len(scored)),
        "supported_answer_rate": _rate(
            sum(1 for r in answered if r.verification_status == "supported"), len(answered)
        )
        if answered
        else None,
    }

    for outcome in ("answer", "clarify", "abstain", "escalate"):
        bucket = by_expected.get(outcome)
        metrics[f"{outcome}_accuracy"] = (
            _rate(bucket["matched"], bucket["total"]) if bucket else None
        )

    return {
        "report": "generation_evaluation",
        "phase": "I",
        "evaluation_version": EVALUATION_VERSION,
        "timestamp": started.isoformat(),
        "dataset": {
            "version": golden_dataset_version(),
            "total_cases": len(cases),
            "scored_cases": len(scored),
            "blocked_cases": len(blocked),
        },
        "configuration": {
            "llm_provider": settings.llm_provider,
            "llm_routing_mode": settings.llm_routing_mode,
            "generation_model": model_name_for("generator", evaluation_provider),
            "judge_model": model_name_for("judge", evaluation_provider),
            "routed_provider": evaluation_provider,
            "verifier_backend": settings.verifier_backend,
            "graph_max_retries": settings.graph_max_retries,
            "llm_execution_profile": settings.llm_execution_profile,
            "graph_llm_call_limit": settings.effective_graph_llm_call_limit,
            "reranker_enabled": settings.reranker_enabled,
        },
        "status": "BLOCKED" if blocked and not scored else "MEASURED",
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "outcome_breakdown": by_expected,
        "latency": {
            **latency_stats([r.latency_ms for r in results]).to_dict(),
            "stage": "end-to-end per case, model loading excluded (warmed before run)",
            "by_stage": latency_by_stage,
            "by_retrieval_component": latency_by_retrieval_component,
        },
        "blocked": [{"case_id": r.case_id, "reason": r.failure_reason} for r in blocked],
        "cases": [r.to_dict() for r in results],
    }


DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "generation_evaluation.json"
