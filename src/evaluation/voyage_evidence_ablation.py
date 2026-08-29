"""Evaluation-only ablation of BGE confidence scoring after Voyage ordering.

No serving module imports this file.  It compares the current Voyage->BGE-5
path with an experiment that leaves Voyage's top-five order untouched and lets
the structured evidence grader plus citation verifier provide semantic safety.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.config import get_settings
from src.generation.schemas import AnswerResponse
from src.retrieval.types import RetrievedChunk
from src.self_healing.evidence_grader import policy_ids_in
from src.self_healing.state import EvidenceGrade, VerificationResult

PathName = Literal["baseline", "candidate", "enhanced_candidate"]
FinalClassification = Literal[
    "correct_answer",
    "correct_abstention",
    "unsafe_incorrect_answer",
    "unnecessary_abstention",
]


@dataclass(frozen=True, slots=True)
class SanityGateResult:
    permitted: bool
    reason: str
    signals: dict[str, Any]


@dataclass(slots=True)
class PathEvaluation:
    case_id: str
    path: PathName
    expected_outcome: Literal["answer", "abstain"]
    final_outcome: Literal["answer", "abstain"]
    classification: FinalClassification
    reason: str
    evidence_grader_rejected: bool = False
    citation_verification_failed: bool = False
    citation_supported: bool = False
    generation_attempted: bool = False
    verification_attempted: bool = False
    latency_ms: float = 0.0
    # ``latency_ms`` is retained for report compatibility and is always the
    # end-to-end value below.  Shared work is measured once by the runner;
    # each path then records only its own sequential downstream work.
    shared_retrieval_voyage_ms: float = 0.0
    path_after_shared_ms: float = 0.0
    end_to_end_ms: float = 0.0
    component_latency_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: dict[str, int] = field(default_factory=dict)
    provider_failures: list[str] = field(default_factory=list)
    # Sanitized evaluation-only failure observations; never raw provider text.
    stage_failures: list[dict[str, str]] = field(default_factory=list)
    provider_retries: dict[str, int] = field(default_factory=dict)
    execution_context: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    evidence_grade: dict[str, Any] = field(default_factory=dict)
    # Present only for the evaluation-only enhanced answerability path.
    answerability: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    supplied_chunk_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_minimal_sanity_gate(
    query: str,
    chunks: list[RetrievedChunk],
    voyage_raw_scores: dict[int, float],
) -> SanityGateResult:
    """Reject only retrieval states unusable by the semantic grader.

    This deliberately contains no relevance threshold, support-count rule, or
    Voyage score calibration.  A missing exact policy identifier is retained:
    an identifier-specific question cannot be grounded by passages that do
    not contain the identifier anywhere in the selected evidence.
    """
    requested_ids = policy_ids_in(query)
    evidence_text = "\n".join(chunk.content for chunk in chunks).upper()
    retrieved_ids = {chunk.policy_id.upper() for chunk in chunks}
    evidence_ids = set(policy_ids_in(evidence_text))
    matched_ids = [
        policy_id
        for policy_id in requested_ids
        if policy_id in retrieved_ids or policy_id in evidence_ids
    ]
    signals: dict[str, Any] = {
        "chunk_count": len(chunks),
        "voyage_scored_chunk_count": sum(chunk.chunk_id in voyage_raw_scores for chunk in chunks),
        "requested_policy_ids": requested_ids,
        "matched_policy_ids": matched_ids,
    }
    if not chunks:
        return SanityGateResult(False, "no_passages_retrieved", signals)

    missing_or_invalid_scores = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.chunk_id not in voyage_raw_scores
        or not isinstance(voyage_raw_scores[chunk.chunk_id], (int, float))
        or isinstance(voyage_raw_scores[chunk.chunk_id], bool)
        or not math.isfinite(float(voyage_raw_scores[chunk.chunk_id]))
    ]
    if missing_or_invalid_scores:
        signals["invalid_score_chunk_ids"] = missing_or_invalid_scores
        return SanityGateResult(False, "voyage_scores_missing_or_invalid", signals)

    if requested_ids and not matched_ids:
        return SanityGateResult(False, "requested_policy_identifier_missing", signals)
    return SanityGateResult(True, "minimal_sanity_passed", signals)


def semantic_only_evidence_grade(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    chain: Any,
    signals: dict[str, Any],
) -> EvidenceGrade:
    """Run the production structured grader without a reranker-score gate.

    The evaluator intentionally reuses the production grader chain and passage
    formatter, so only the deterministic BGE-confidence conjunction is ablated.
    This function is evaluation-only and is never called from the graph.
    """
    from src.self_healing.evidence_grader import _format_passages  # noqa: PLC2701

    if not chunks:
        return EvidenceGrade(
            relevant=False,
            sufficient=False,
            confidence=0.0,
            missing_information=["any policy passage matching the question"],
            rationale="no passages retrieved",
            signals=signals,
            deterministic_only=True,
        )
    try:
        raw = chain.invoke({"question": query, "context": _format_passages(chunks)})
    except Exception:  # noqa: BLE001 - an unavailable grader must fail closed
        return EvidenceGrade(
            relevant=False,
            sufficient=False,
            confidence=0.0,
            missing_information=["semantic evidence grader unavailable"],
            rationale="semantic evidence grader unavailable",
            signals=signals,
            deterministic_only=True,
        )
    if not isinstance(raw, dict):
        return EvidenceGrade(
            relevant=False,
            sufficient=False,
            confidence=0.0,
            missing_information=["semantic evidence grader returned invalid output"],
            rationale="semantic evidence grader returned invalid output",
            signals=signals,
            deterministic_only=True,
        )

    grade = EvidenceGrade.model_validate({**raw, "signals": signals})
    grade.sufficient = bool(
        grade.sufficient
        and grade.confidence >= get_settings().evidence_confidence_threshold
    )
    if grade.sufficient:
        grade.missing_information = []
    elif not grade.missing_information:
        grade.missing_information = ["grader judged the passages incomplete"]
    return grade


def _classification(
    expected_outcome: Literal["answer", "abstain"], final_outcome: Literal["answer", "abstain"]
) -> FinalClassification:
    if expected_outcome == "answer":
        return "correct_answer" if final_outcome == "answer" else "unnecessary_abstention"
    return "unsafe_incorrect_answer" if final_outcome == "answer" else "correct_abstention"


def evaluate_path(
    *,
    case_id: str,
    question: str,
    expected_outcome: Literal["answer", "abstain"],
    path: PathName,
    chunks: list[RetrievedChunk],
    gate: SanityGateResult,
    grade_evidence: Callable[[], EvidenceGrade],
    generate_answer: Callable[[], AnswerResponse],
    verify: Callable[[AnswerResponse], VerificationResult],
    shared_latency_ms: float,
    component_latency_ms: dict[str, float],
    initial_path_after_shared_ms: float = 0.0,
    shared_provider_retries: int = 0,
) -> PathEvaluation:
    """Run one downstream path and classify its final safety outcome."""
    if path != "baseline" and "bge_fixed_order_confidence_ms" in component_latency_ms:
        raise ValueError("non-baseline evaluation must not invoke or time BGE confidence scoring")
    started = time.perf_counter()
    components = dict(component_latency_ms)

    def timing() -> tuple[float, float]:
        """Return non-overlapping downstream and end-to-end wall time."""
        path_after_shared_ms = initial_path_after_shared_ms + (
            time.perf_counter() - started
        ) * 1000.0
        return path_after_shared_ms, shared_latency_ms + path_after_shared_ms

    def result_timing() -> dict[str, float]:
        path_after_shared_ms, end_to_end_ms = timing()
        return {
            "latency_ms": end_to_end_ms,
            "shared_retrieval_voyage_ms": shared_latency_ms,
            "path_after_shared_ms": path_after_shared_ms,
            "end_to_end_ms": end_to_end_ms,
        }
    # Evidence grading and generation invoke one structured chain each. The
    # verifier may short-circuit lexically, so only its observed entailment
    # claim calls are counted as LLM calls below.
    llm_calls = {"evidence_grader": 0, "generation": 0, "citation_entailment": 0}
    failures: list[str] = []
    stage_failures: list[dict[str, str]] = []
    supplied_chunk_ids = [chunk.chunk_id for chunk in chunks]
    if not gate.permitted:
        return PathEvaluation(
            case_id=case_id,
            path=path,
            expected_outcome=expected_outcome,
            final_outcome="abstain",
            classification=_classification(expected_outcome, "abstain"),
            reason=gate.reason,
            **result_timing(),
            component_latency_ms=components,
            llm_calls=llm_calls,
            provider_retries={"voyage": shared_provider_retries},
            gate=asdict(gate),
            supplied_chunk_ids=supplied_chunk_ids,
        )

    llm_calls["evidence_grader"] = 1
    grade_started = time.perf_counter()
    grade = grade_evidence()
    components["evidence_grader_ms"] = (time.perf_counter() - grade_started) * 1000.0
    grade_dump = grade.model_dump()
    answerability_dump = (
        grade_dump if "proposition_status" in grade_dump else {}
    )
    answerability_failure_category = str(getattr(grade, "failure_category", ""))
    answerability_failure_reason = str(getattr(grade, "failure_reason", ""))
    answerability_failure_phase = str(getattr(grade, "failure_phase", ""))
    answerability_failure_exception_type = str(
        getattr(grade, "failure_exception_type", "")
    )
    grade_failure = bool(
        grade.deterministic_only
        and any("unavailable" in item or "invalid" in item for item in grade.missing_information)
    )
    if answerability_failure_category:
        failures.append(f"evidence_grader_{answerability_failure_category}")
        stage_failure = {
            "stage": "answerability_grader",
            "category": answerability_failure_category,
            "reason": answerability_failure_reason or "sanitized failure reason unavailable",
        }
        if answerability_failure_phase:
            stage_failure["phase"] = answerability_failure_phase
        if answerability_failure_exception_type:
            stage_failure["exception_type"] = answerability_failure_exception_type
        stage_failures.append(stage_failure)
    elif grade_failure:
        failures.append("evidence_grader_unavailable_or_invalid")
    if not grade.sufficient:
        return PathEvaluation(
            case_id=case_id,
            path=path,
            expected_outcome=expected_outcome,
            final_outcome="abstain",
            classification=_classification(expected_outcome, "abstain"),
            reason="evidence_grader_rejected",
            evidence_grader_rejected=True,
            **result_timing(),
            component_latency_ms=components,
            llm_calls=llm_calls,
            provider_failures=failures,
            stage_failures=stage_failures,
            provider_retries={"voyage": shared_provider_retries},
            gate=asdict(gate),
            evidence_grade=grade_dump,
            answerability=answerability_dump,
            supplied_chunk_ids=supplied_chunk_ids,
        )

    llm_calls["generation"] = 1
    generation_started = time.perf_counter()
    response = generate_answer()
    components["generation_ms"] = (time.perf_counter() - generation_started) * 1000.0
    if response.outcome != "answered":
        if response.outcome in {"provider_error", "invalid_output"}:
            failures.append(f"generation_{response.outcome}")
        return PathEvaluation(
            case_id=case_id,
            path=path,
            expected_outcome=expected_outcome,
            final_outcome="abstain",
            classification=_classification(expected_outcome, "abstain"),
            reason=f"generation_{response.outcome}",
            evidence_grader_rejected=False,
            generation_attempted=True,
            **result_timing(),
            component_latency_ms=components,
            llm_calls=llm_calls,
            provider_failures=failures,
            stage_failures=stage_failures,
            provider_retries={"voyage": shared_provider_retries},
            gate=asdict(gate),
            evidence_grade=grade_dump,
            answerability=answerability_dump,
            supplied_chunk_ids=supplied_chunk_ids,
        )

    verification_started = time.perf_counter()
    verification = verify(response)
    components["citation_verifier_ms"] = (time.perf_counter() - verification_started) * 1000.0
    llm_calls["citation_entailment"] = sum(
        claim.get("method") in {"entailment", "entailment-unavailable"}
        for claim in verification.claims
    )
    if not verification.checked:
        failures.append("citation_verifier_unavailable")
    supported = bool(verification.checked and verification.supported)
    final_outcome: Literal["answer", "abstain"] = "answer" if supported else "abstain"
    return PathEvaluation(
        case_id=case_id,
        path=path,
        expected_outcome=expected_outcome,
        final_outcome=final_outcome,
        classification=_classification(expected_outcome, final_outcome),
        reason="verified_answer" if supported else "citation_verification_failed",
        citation_verification_failed=not supported,
        citation_supported=supported,
        generation_attempted=True,
        verification_attempted=True,
        **result_timing(),
        component_latency_ms=components,
        llm_calls=llm_calls,
        provider_failures=failures,
        stage_failures=stage_failures,
        provider_retries={"voyage": shared_provider_retries},
        gate=asdict(gate),
        evidence_grade=grade_dump,
        answerability=answerability_dump,
        verification=verification.model_dump(),
        citations=list(response.citation_ids),
        supplied_chunk_ids=supplied_chunk_ids,
    )


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
    }


def aggregate_path(results: list[PathEvaluation]) -> dict[str, Any]:
    """Aggregate final outcomes; no pre-grader permit is called an answer."""
    total = len(results)
    counts = Counter(result.classification for result in results)
    expected_answers = sum(result.expected_outcome == "answer" for result in results)
    expected_abstentions = total - expected_answers
    generation_attempts = sum(result.generation_attempted for result in results)
    llm_calls: Counter[str] = Counter()
    provider_failures: Counter[str] = Counter()
    provider_retries: Counter[str] = Counter()
    stage_failure_categories: Counter[str] = Counter()
    for result in results:
        llm_calls.update(result.llm_calls)
        provider_failures.update(result.provider_failures)
        provider_retries.update(result.provider_retries)
        stage_failure_categories.update(
            failure["category"] for failure in result.stage_failures if failure.get("category")
        )

    correct = counts["correct_answer"] + counts["correct_abstention"]
    component_names = sorted(
        {name for result in results for name in result.component_latency_ms}
    )
    return {
        "final_answer_abstain_accuracy": correct / total if total else 0.0,
        "answer_recall": counts["correct_answer"] / expected_answers if expected_answers else 0.0,
        "abstention_recall": (
            counts["correct_abstention"] / expected_abstentions if expected_abstentions else 0.0
        ),
        "unsafe_answer_rate": (
            counts["unsafe_incorrect_answer"] / expected_abstentions
            if expected_abstentions
            else 0.0
        ),
        "citation_supported_answer_rate": (
            sum(result.citation_supported for result in results) / generation_attempts
            if generation_attempts
            else 0.0
        ),
        "citation_supported_answer_denominator": generation_attempts,
        "evidence_grader_rejection_rate": (
            sum(result.evidence_grader_rejected for result in results) / total if total else 0.0
        ),
        "outcomes": {
            "correct_answer": counts["correct_answer"],
            "correct_abstention": counts["correct_abstention"],
            "unsafe_incorrect_answer": counts["unsafe_incorrect_answer"],
            "unnecessary_abstention": counts["unnecessary_abstention"],
            "citation_verification_failure": sum(
                result.citation_verification_failed for result in results
            ),
            "evidence_grader_rejection": sum(
                result.evidence_grader_rejected for result in results
            ),
        },
        # ``latency_ms`` remains the legacy name for end-to-end wall time.
        "latency_ms": latency_summary([result.end_to_end_ms for result in results]),
        "timing_ms": {
            "end_to_end_ms": latency_summary([result.end_to_end_ms for result in results]),
            "path_after_shared_ms": latency_summary(
                [result.path_after_shared_ms for result in results]
            ),
            "components": {
                name: latency_summary(
                    [result.component_latency_ms.get(name, 0.0) for result in results]
                )
                for name in component_names
            },
        },
        "llm_calls": dict(llm_calls),
        "provider_failures": dict(provider_failures),
        "provider_retries": dict(provider_retries),
        "stage_failure_categories": dict(stage_failure_categories),
    }


def aggregate_paired_timing(
    baseline: list[PathEvaluation], candidate: list[PathEvaluation]
) -> dict[str, Any]:
    """Summarise matched cases without treating sequential LLM work as causal.

    The deterministic BGE pass is reported separately.  The paired end-to-end
    delta includes independent sequential LLM calls and is therefore
    descriptive rather than a causal speedup measurement.
    """
    candidates = {result.case_id: result for result in candidate}
    deltas: list[float] = []
    ratios: list[float] = []
    bge_costs: list[float] = []
    rows: list[dict[str, float | str]] = []
    for base in baseline:
        comparison = candidates.get(base.case_id)
        if comparison is None:
            continue
        delta = comparison.end_to_end_ms - base.end_to_end_ms
        ratio = (
            base.end_to_end_ms / comparison.end_to_end_ms
            if comparison.end_to_end_ms > 0.0
            else 0.0
        )
        bge_cost = base.component_latency_ms.get("bge_fixed_order_confidence_ms", 0.0)
        deltas.append(delta)
        ratios.append(ratio)
        bge_costs.append(bge_cost)
        rows.append(
            {
                "case_id": base.case_id,
                "bge_incremental_cost_ms": bge_cost,
                "candidate_extra_downstream_cost_ms": comparison.path_after_shared_ms,
                "paired_end_to_end_delta_ms": delta,
                "paired_end_to_end_speedup_ratio": ratio,
            }
        )
    return {
        "interpretation": (
            "bge_incremental_cost_ms is deterministic local work removed by the candidate. "
            "Paired end-to-end deltas include sequential, independently variable downstream "
            "LLM work and must not be interpreted as causal speedup."
        ),
        "bge_incremental_cost_ms": latency_summary(bge_costs),
        "candidate_extra_downstream_cost_ms": latency_summary(
            [row["candidate_extra_downstream_cost_ms"] for row in rows]
        ),
        "paired_end_to_end_delta_ms": latency_summary(deltas),
        "paired_end_to_end_speedup_ratio": latency_summary(ratios),
        "cases": rows,
    }


def disagreement_diagnostics(
    baseline: list[PathEvaluation], candidate: list[PathEvaluation]
) -> list[dict[str, Any]]:
    """Return final-outcome disagreements, keeping reasons and gates auditable."""
    candidate_by_id = {result.case_id: result for result in candidate}
    rows: list[dict[str, Any]] = []
    for base in baseline:
        comparison = candidate_by_id.get(base.case_id)
        if comparison is None:
            continue
        if (
            base.final_outcome != comparison.final_outcome
            or base.classification != comparison.classification
            or base.reason != comparison.reason
        ):
            rows.append(
                {
                    "case_id": base.case_id,
                    "expected_outcome": base.expected_outcome,
                    "baseline": base.to_dict(),
                    "candidate": comparison.to_dict(),
                }
            )
    return rows
