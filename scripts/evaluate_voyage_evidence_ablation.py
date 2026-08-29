"""Compare BGE-backed and semantic-only evidence gates after Voyage reranking.

This is an evaluation harness, never a serving entry point. It leaves the
configured production graph unchanged and writes a report only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_voyage_confidence import (  # noqa: E402
    RetryAfterTrackingClient,
    rerank_voyage_with_rate_limit_retry,
)
from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.evaluation.metrics import golden_dataset_version, load_golden_dataset  # noqa: E402
from src.evaluation.voyage_evidence_ablation import (  # noqa: E402
    PathEvaluation,
    SanityGateResult,
    aggregate_paired_timing,
    aggregate_path,
    candidate_minimal_sanity_gate,
    disagreement_diagnostics,
    evaluate_path,
    semantic_only_evidence_grade,
)


def pace_voyage_request(
    *,
    previous_success: bool,
    delay_seconds: float,
    prefix: str,
    progress: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Apply evaluation-only pacing between successful Voyage requests."""
    if previous_success and delay_seconds:
        progress(f"{prefix} Voyage pacing - waiting {delay_seconds:.1f}s")
        sleep(delay_seconds)


def _with_llm_permit(settings: Any, stage: str, invoke: Any) -> Any:
    """Reserve the initial call; factory fallback reserves its own extra call."""
    from src.self_healing.execution_budget import reserve_llm_call

    permit = reserve_llm_call(
        stage,
        default_timeout_s=settings.llm_request_timeout_s,
        default_max_retries=settings.llm_max_retries,
    )
    return invoke(permit)


def _evaluate_in_graph_context(settings: Any, evaluate: Any) -> PathEvaluation:
    """Give one evaluation path the same route and bounded call budget as a graph run."""
    from src.generation.llm_routing import route_context
    from src.self_healing.execution_budget import ExecutionBudget, request_budget

    budget = ExecutionBudget(
        timeout_s=settings.graph_request_timeout_s,
        max_llm_calls=settings.effective_graph_llm_call_limit,
    )
    with route_context(settings, "evaluation") as route, request_budget(budget):
        result = evaluate()
        result.execution_context = {"route": route.snapshot(), "budget": budget.snapshot()}
    return result


def select_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a stable prefix; zero means every answer/abstain golden case."""
    eligible = [case for case in cases if case["expected_outcome"] in {"answer", "abstain"}]
    if limit < 0:
        raise ValueError("limit must be zero or greater")
    return eligible if limit == 0 else eligible[:limit]


def select_explicit_cases(
    cases: list[dict[str, Any]], case_ids: list[str]
) -> list[dict[str, Any]]:
    """Select requested golden cases exactly once and in supplied order."""
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate --case-id value(s): {', '.join(duplicates)}")
    cases_by_id = {case["id"]: case for case in cases}
    unknown = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if unknown:
        raise ValueError(f"unknown --case-id value(s): {', '.join(unknown)}")
    return [cases_by_id[case_id] for case_id in case_ids]


def validate_explicit_selection_options(
    *,
    case_ids: list[str] | None,
    limit: int,
    stratified: bool,
    answer_cases_supplied: bool = False,
    abstain_cases_supplied: bool = False,
) -> None:
    """Reject mutually exclusive CLI selection modes before provider setup."""
    if not case_ids:
        return
    conflicts: list[str] = []
    if limit:
        conflicts.append("--limit")
    if stratified:
        conflicts.append("--stratified")
    if answer_cases_supplied:
        conflicts.append("--answer-cases")
    if abstain_cases_supplied:
        conflicts.append("--abstain-cases")
    if conflicts:
        raise ValueError("--case-id cannot be combined with " + ", ".join(conflicts))


def abstention_category(case: dict[str, Any]) -> str:
    """Return the safety category used only for stratified evaluation sampling."""
    if case.get("case_type") == "prompt_injection":
        return "injection_adversarial"
    return str(case.get("category") or "uncategorized")


def select_stratified_cases(
    cases: list[dict[str, Any]],
    *,
    answer_cases: int,
    abstain_cases: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    """Select a reproducible outcome-balanced sample with abstention coverage.

    At least one case is taken from every available abstention category while
    capacity permits; remaining abstention capacity is filled round-robin over
    the same seeded category order.  This avoids a nominally "stratified" run
    accidentally concentrating all safety cases in one category.
    """
    if answer_cases < 0 or abstain_cases < 0:
        raise ValueError("stratified answer and abstain counts must be zero or greater")
    answers = sorted(
        (case for case in cases if case.get("expected_outcome") == "answer"),
        key=lambda case: case["id"],
    )
    abstentions = [case for case in cases if case.get("expected_outcome") == "abstain"]
    if answer_cases > len(answers) or abstain_cases > len(abstentions):
        raise ValueError("requested stratified counts exceed available eligible golden cases")

    rng = random.Random(selection_seed)
    rng.shuffle(answers)
    selected_answers = answers[:answer_cases]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in abstentions:
        by_category[abstention_category(case)].append(case)
    categories = sorted(by_category)
    for category in categories:
        by_category[category].sort(key=lambda case: case["id"])
        rng.shuffle(by_category[category])
    rng.shuffle(categories)

    selected_abstentions: list[dict[str, Any]] = []
    while len(selected_abstentions) < abstain_cases:
        selected_this_round = False
        for category in categories:
            if len(selected_abstentions) >= abstain_cases:
                break
            if by_category[category]:
                selected_abstentions.append(by_category[category].pop())
                selected_this_round = True
        if not selected_this_round:
            break

    selected = selected_answers + selected_abstentions
    # The sampling is seeded above; original-dataset order makes live-run
    # progress and reports easy to compare across equivalent selections.
    original_positions = {case["id"]: index for index, case in enumerate(cases)}
    return sorted(selected, key=lambda case: original_positions[case["id"]])


def selection_metadata(
    cases: list[dict[str, Any]], *, selection_mode: str, selection_seed: int | None
) -> dict[str, Any]:
    """Make every live evaluation sample auditable and reproducible."""
    abstention_categories = Counter(
        abstention_category(case) for case in cases if case.get("expected_outcome") == "abstain"
    )
    return {
        "selection_mode": selection_mode,
        "seed": selection_seed,
        "selected_case_ids": [case["id"] for case in cases],
        "case_count": len(cases),
        "answer_case_count": sum(case.get("expected_outcome") == "answer" for case in cases),
        "abstain_case_count": sum(case.get("expected_outcome") == "abstain" for case in cases),
        "abstention_category_counts": dict(sorted(abstention_categories.items())),
    }


def _blocked_baseline(
    case: dict[str, Any],
    *,
    reason: str,
    shared_latency_ms: float,
    bge_latency_ms: float,
    voyage_retries: int,
    chunks: list[Any],
) -> PathEvaluation:
    expected = case["expected_outcome"]
    return PathEvaluation(
        case_id=case["id"],
        path="baseline",
        expected_outcome=expected,
        final_outcome="abstain",
        classification=("unnecessary_abstention" if expected == "answer" else "correct_abstention"),
        reason=reason,
        latency_ms=shared_latency_ms + bge_latency_ms,
        shared_retrieval_voyage_ms=shared_latency_ms,
        path_after_shared_ms=bge_latency_ms,
        end_to_end_ms=shared_latency_ms + bge_latency_ms,
        component_latency_ms={"bge_fixed_order_confidence_ms": bge_latency_ms},
        provider_failures=[reason],
        provider_retries={"voyage": voyage_retries},
        supplied_chunk_ids=[chunk.chunk_id for chunk in chunks],
    )


def run(
    *,
    limit: int = 0,
    case_ids: list[str] | None = None,
    stratified: bool = False,
    answer_cases: int = 18,
    abstain_cases: int = 12,
    selection_seed: int = 42,
    voyage_max_retries: int = 4,
    voyage_retry_base_seconds: float = 5.0,
    voyage_request_delay_seconds: float = 1.0,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run both paths over the same retrieved candidates and Voyage order."""
    from src.evaluation.answerability_ablation import (
        _build_answerability_chain,
        grade_answerability,
    )

    enhanced_generation_instruction = (
        "When comparing or summarising policy branches with distinct numeric "
        "ranges, preserve each branch and its exact range separately. Do not "
        "collapse distinct ranges into one overall numeric envelope."
    )
    from src.generation.answer_chain import build_answer_chain, generate_grounded_answer
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.reranking.provider import VoyageReranker
    from src.retrieval.hybrid import get_hybrid_retriever
    from src.self_healing.evidence_grader import (  # noqa: PLC2701
        _build_grader_chain,
        grade_evidence,
    )
    from src.self_healing.verification import get_default_verifier

    if voyage_max_retries < 0 or voyage_retry_base_seconds < 0 or voyage_request_delay_seconds < 0:
        raise ValueError("Voyage retry and pacing values must be zero or greater")
    validate_explicit_selection_options(case_ids=case_ids, limit=limit, stratified=stratified)
    if stratified and limit:
        raise ValueError("--limit cannot be combined with stratified selection")
    dataset_cases = load_golden_dataset()
    if case_ids:
        cases = select_explicit_cases(dataset_cases, case_ids)
    elif stratified:
        cases = select_stratified_cases(
            dataset_cases,
            answer_cases=answer_cases,
            abstain_cases=abstain_cases,
            selection_seed=selection_seed,
        )
    else:
        cases = select_cases(dataset_cases, limit)
    if not cases:
        raise RuntimeError("no answer/abstain golden cases selected")

    # Selection validation deliberately precedes all provider setup, so an
    # unknown or duplicate explicit ID can never trigger a Voyage or LLM call.
    settings = get_settings()
    if settings.reranker_provider != "voyage" or not settings.reranker_remote_allowed:
        raise RuntimeError("set RERANKER_PROVIDER=voyage and RERANKER_REMOTE_ALLOWED=true")
    if not settings.voyage_api_key:
        raise RuntimeError("set VOYAGE_API_KEY")

    retriever = get_hybrid_retriever()
    def emit(message: str) -> None:
        if progress is None:
            print(message, flush=True)
        else:
            progress(message)

    voyage_client = RetryAfterTrackingClient(
        api_key=settings.voyage_api_key,
        timeout_seconds=settings.hosted_rerank_timeout_seconds,
    )
    voyage = VoyageReranker(
        api_key=settings.voyage_api_key,
        model_name=settings.voyage_rerank_model,
        timeout_seconds=settings.hosted_rerank_timeout_seconds,
        # The reused calibration wrapper owns all evaluation-only 429 pacing.
        max_retries=0,
        client=voyage_client,
    )
    # Deliberately lazy: this is not allocated until the first successful
    # Voyage request actually needs the baseline BGE confidence pass.
    local: CrossEncoderReranker | None = None
    verifier = get_default_verifier()
    baseline_results: list[PathEvaluation] = []
    candidate_results: list[PathEvaluation] = []
    enhanced_candidate_results: list[PathEvaluation] = []
    voyage_request_records: list[dict[str, Any]] = []
    rate_limit_events: list[dict[str, Any]] = []
    previous_voyage_success = False

    def run_candidate_path(
        case: dict[str, Any],
        chunks: list[Any],
        gate: SanityGateResult,
        shared_latency_ms: float,
        candidate_sanity_gate_ms: float,
        voyage_retries: int,
    ) -> PathEvaluation:
        question = case["question"]

        def grade() -> Any:
            return _with_llm_permit(
                settings,
                "evidence_grader",
                lambda permit: semantic_only_evidence_grade(
                    question,
                    chunks,
                    chain=_build_grader_chain(
                        timeout_s=permit.timeout_s, max_retries=permit.max_retries
                    ),
                    signals=gate.signals,
                ),
            )

        def generate() -> Any:
            return _with_llm_permit(
                settings,
                "generate_answer",
                lambda permit: generate_grounded_answer(
                    question,
                    chunks,
                    chain=build_answer_chain(
                        timeout_s=permit.timeout_s, max_retries=permit.max_retries
                    ),
                ),
            )

        return _evaluate_in_graph_context(
            settings,
            lambda: evaluate_path(
                case_id=case["id"],
                question=question,
                expected_outcome=case["expected_outcome"],
                path="candidate",
                chunks=chunks,
                gate=gate,
                grade_evidence=grade,
                generate_answer=generate,
                verify=lambda response: verifier.verify(
                    response.answer,
                    response.citation_ids,
                    chunks,
                    response.claim_citations,
                ),
                shared_latency_ms=shared_latency_ms,
                component_latency_ms={"candidate_sanity_gate_ms": candidate_sanity_gate_ms},
                initial_path_after_shared_ms=candidate_sanity_gate_ms,
                shared_provider_retries=voyage_retries,
            ),
        )

    def run_baseline_path(
        case: dict[str, Any],
        chunks: list[Any],
        shared_latency_ms: float,
        component_latency_ms: dict[str, float],
        initial_path_after_shared_ms: float,
        voyage_retries: int,
    ) -> PathEvaluation:
        question = case["question"]

        def grade() -> Any:
            return _with_llm_permit(
                settings,
                "evidence_grader",
                lambda permit: grade_evidence(
                    question,
                    chunks,
                    use_llm=True,
                    chain=_build_grader_chain(
                        timeout_s=permit.timeout_s, max_retries=permit.max_retries
                    ),
                ),
            )

        def generate() -> Any:
            return _with_llm_permit(
                settings,
                "generate_answer",
                lambda permit: generate_grounded_answer(
                    question,
                    chunks,
                    chain=build_answer_chain(
                        timeout_s=permit.timeout_s, max_retries=permit.max_retries
                    ),
                ),
            )

        return _evaluate_in_graph_context(
            settings,
            lambda: evaluate_path(
                case_id=case["id"],
                question=question,
                expected_outcome=case["expected_outcome"],
                path="baseline",
                chunks=chunks,
                gate=SanityGateResult(True, "current_bge_deterministic_gate", {}),
                grade_evidence=grade,
                generate_answer=generate,
                verify=lambda response: verifier.verify(
                    response.answer,
                    response.citation_ids,
                    chunks,
                    response.claim_citations,
                ),
                shared_latency_ms=shared_latency_ms,
                component_latency_ms=component_latency_ms,
                initial_path_after_shared_ms=initial_path_after_shared_ms,
                shared_provider_retries=voyage_retries,
            ),
        )

    def run_enhanced_candidate_path(
        case: dict[str, Any],
        chunks: list[Any],
        gate: SanityGateResult,
        shared_latency_ms: float,
        sanity_gate_ms: float,
        voyage_retries: int,
    ) -> PathEvaluation:
        """Evaluation-only path; does not alter the existing candidate path."""
        question = case["question"]

        def grade() -> Any:
            return _with_llm_permit(
                settings,
                "answerability_grader",
                lambda permit: grade_answerability(
                    question,
                    chunks,
                    chain=_build_answerability_chain(
                        timeout_s=permit.timeout_s, max_retries=permit.max_retries
                    ),
                    signals=gate.signals,
                ),
            )

        def generate() -> Any:
            return _with_llm_permit(
                settings,
                "generate_answer",
                lambda permit: generate_grounded_answer(
                    question,
                    chunks,
                    chain=build_answer_chain(
                        timeout_s=permit.timeout_s,
                        max_retries=permit.max_retries,
                        additional_system_instructions=enhanced_generation_instruction,
                    ),
                ),
            )

        return _evaluate_in_graph_context(
            settings,
            lambda: evaluate_path(
                case_id=case["id"],
                question=question,
                expected_outcome=case["expected_outcome"],
                path="enhanced_candidate",
                chunks=chunks,
                gate=gate,
                grade_evidence=grade,
                generate_answer=generate,
                verify=lambda response: verifier.verify(
                    response.answer,
                    response.citation_ids,
                    chunks,
                    response.claim_citations,
                ),
                shared_latency_ms=shared_latency_ms,
                component_latency_ms={"enhanced_candidate_sanity_gate_ms": sanity_gate_ms},
                initial_path_after_shared_ms=sanity_gate_ms,
                shared_provider_retries=voyage_retries,
            ),
        )

    for index, case in enumerate(cases, start=1):
        prefix = f"[{index}/{len(cases)}]"
        emit(f"{prefix} retrieval...")
        retrieval_started = time.perf_counter()
        candidates = retriever.retrieve(case["question"])
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        # Evaluation pacing and rate-limit backoff are part of the elapsed
        # Voyage stage, never retrieval or either downstream path.
        voyage_started = time.perf_counter()
        pace_voyage_request(
            previous_success=previous_voyage_success,
            delay_seconds=voyage_request_delay_seconds,
            prefix=prefix,
            progress=emit,
            sleep=sleep,
        )
        emit(f"{prefix} Voyage...")
        try:
            voyage_result, voyage_retry_count, case_rate_limit_events = (
                rerank_voyage_with_rate_limit_retry(
                    voyage,
                    case["question"],
                    candidates,
                    top_k=min(settings.rerank_top_k, settings.hosted_rerank_top_k),
                    candidate_top_k=min(
                        settings.rerank_candidate_top_k, settings.hosted_rerank_max_candidates
                    ),
                    max_retries=voyage_max_retries,
                    retry_base_seconds=voyage_retry_base_seconds,
                    retry_after=lambda: voyage_client.retry_after,
                    progress=emit,
                    prefix=prefix,
                    sleep=sleep,
                )
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"evaluation blocked at {case['id']}: Voyage unavailable ({exc})"
            ) from exc
        previous_voyage_success = True
        voyage_ms = (time.perf_counter() - voyage_started) * 1000.0
        shared_latency_ms = retrieval_ms + voyage_ms
        voyage_request_records.append(
            {
                "case_id": case["id"],
                "retrieval_ms": retrieval_ms,
                "voyage_ms": voyage_ms,
                "shared_retrieval_voyage_ms": shared_latency_ms,
                "retry_count": voyage_retry_count,
                "rate_limit_events": case_rate_limit_events,
            }
        )
        rate_limit_events.extend(
            [{"case_id": case["id"], **event} for event in case_rate_limit_events]
        )

        voyage_chunks = list(voyage_result.chunks)
        voyage_scores = dict(voyage_result.provider_raw_scores)
        candidate_gate_started = time.perf_counter()
        candidate_gate = candidate_minimal_sanity_gate(
            case["question"], voyage_chunks, voyage_scores
        )
        candidate_sanity_gate_ms = (time.perf_counter() - candidate_gate_started) * 1000.0
        candidate_results.append(
            run_candidate_path(
                case,
                voyage_chunks,
                candidate_gate,
                shared_latency_ms,
                candidate_sanity_gate_ms,
                voyage_retry_count,
            )
        )
        enhanced_candidate_results.append(
            run_enhanced_candidate_path(
                case,
                voyage_chunks,
                candidate_gate,
                shared_latency_ms,
                candidate_sanity_gate_ms,
                voyage_retry_count,
            )
        )

        emit(f"{prefix} BGE baseline confidence...")
        if local is None:
            local = CrossEncoderReranker(device=settings.resolved_reranker_device)
        bge_started = time.perf_counter()
        bge_result = local.score_fixed_order_with_diagnostics(case["question"], voyage_chunks)
        bge_latency_ms = (time.perf_counter() - bge_started) * 1000.0
        baseline_components = {
            "bge_fixed_order_confidence_ms": bge_latency_ms,
        }
        if not bge_result.reranker_used:
            baseline_results.append(
                _blocked_baseline(
                    case,
                    reason=f"bge_confidence_unavailable:{bge_result.failure or 'unknown'}",
                    shared_latency_ms=shared_latency_ms,
                    bge_latency_ms=bge_latency_ms,
                    voyage_retries=voyage_retry_count,
                    chunks=voyage_chunks,
                )
            )
        else:
            bge_chunks = list(bge_result.chunks)
            baseline_results.append(
                run_baseline_path(
                    case,
                    bge_chunks,
                    shared_latency_ms,
                    baseline_components,
                    bge_latency_ms,
                    voyage_retry_count,
                )
            )
        emit(f"{prefix} done")

    baseline_summary = aggregate_path(baseline_results)
    candidate_summary = aggregate_path(candidate_results)
    enhanced_candidate_summary = aggregate_path(enhanced_candidate_results)
    paired_timing = aggregate_paired_timing(baseline_results, candidate_results)
    return {
        "report": "voyage_evidence_ablation_v1",
        "dataset_version": golden_dataset_version(),
        "cases": len(cases),
        "selection": selection_metadata(
            cases,
            selection_mode=(
                "explicit_case_ids" if case_ids else "stratified" if stratified else "limit_prefix"
            ),
            selection_seed=selection_seed if stratified else None,
        ),
        "production_changed": False,
        "recommendation": "EVALUATION ONLY: no candidate approval or serving change is automatic.",
        "outcome_definition": (
            "correct_answer means an expected-answer golden case reached a citation-verified "
            "final answer; semantic answer-text equivalence remains a separate evaluation. "
            "unsafe_incorrect_answer means an expected-abstain golden case reached a "
            "citation-verified final answer."
        ),
        "provider_retry_observability": {
            "voyage": "actual retry counts are recorded per request",
            "llm": "provider SDK retry counts are not exposed by the existing chain adapters",
        },
        "voyage_rate_limits": {
            "max_retries": voyage_max_retries,
            "retry_base_seconds": voyage_retry_base_seconds,
            "request_delay_seconds": voyage_request_delay_seconds,
            "total_rate_limit_events": len(rate_limit_events),
            "events": rate_limit_events,
            "per_case": voyage_request_records,
            "total_retries": sum(record["retry_count"] for record in voyage_request_records),
        },
        "shared_stage_timing": {
            "description": (
                "retrieval and Voyage are each measured once per case. Voyage time includes "
                "evaluation-only pacing and rate-limit backoff."
            ),
            "retrieval_ms": {
                "mean": sum(record["retrieval_ms"] for record in voyage_request_records)
                / len(voyage_request_records),
                "p50": sorted(record["retrieval_ms"] for record in voyage_request_records)[
                    len(voyage_request_records) // 2
                ],
                "p95": sorted(record["retrieval_ms"] for record in voyage_request_records)[
                    round((len(voyage_request_records) - 1) * 0.95)
                ],
            },
            "voyage_ms": {
                "mean": sum(record["voyage_ms"] for record in voyage_request_records)
                / len(voyage_request_records),
                "p50": sorted(record["voyage_ms"] for record in voyage_request_records)[
                    len(voyage_request_records) // 2
                ],
                "p95": sorted(record["voyage_ms"] for record in voyage_request_records)[
                    round((len(voyage_request_records) - 1) * 0.95)
                ],
            },
            "shared_retrieval_voyage_ms": {
                "mean": sum(
                    record["shared_retrieval_voyage_ms"] for record in voyage_request_records
                )
                / len(voyage_request_records),
                "p50": sorted(
                    record["shared_retrieval_voyage_ms"] for record in voyage_request_records
                )[len(voyage_request_records) // 2],
                "p95": sorted(
                    record["shared_retrieval_voyage_ms"] for record in voyage_request_records
                )[round((len(voyage_request_records) - 1) * 0.95)],
            },
            "cases": voyage_request_records,
        },
        "paired_timing": paired_timing,
        "paths": {
            "baseline": {
                "description": "Voyage top-5 -> BGE fixed-order confidence -> current evidence grader",
                "summary": baseline_summary,
                "cases": [result.to_dict() for result in baseline_results],
            },
            "candidate": {
                "description": "Voyage top-5 -> minimal sanity -> structured grader -> verifier",
                "summary": candidate_summary,
                "cases": [result.to_dict() for result in candidate_results],
            },
            "enhanced_candidate": {
                "description": (
                    "Voyage top-5 -> minimal sanity -> evaluation-only structured "
                    "answerability grader -> generation -> verifier"
                ),
                "summary": enhanced_candidate_summary,
                "cases": [result.to_dict() for result in enhanced_candidate_results],
            },
        },
        "focused_answerability_comparisons": [
            {
                "case_id": case_id,
                "baseline": next(
                    (result.to_dict() for result in baseline_results if result.case_id == case_id),
                    None,
                ),
                "candidate": next(
                    (result.to_dict() for result in candidate_results if result.case_id == case_id),
                    None,
                ),
                "enhanced_candidate": next(
                    (
                        result.to_dict()
                        for result in enhanced_candidate_results
                        if result.case_id == case_id
                    ),
                    None,
                ),
            }
            for case_id in ("GC-052", "GC-055", "GC-062")
            if any(result.case_id == case_id for result in enhanced_candidate_results)
        ],
        "final_outcome_disagreements": disagreement_diagnostics(
            baseline_results, candidate_results
        ),
        "safety_gate": {
            "candidate_must_not_increase_unsafe_answers": (
                candidate_summary["outcomes"]["unsafe_incorrect_answer"]
                <= baseline_summary["outcomes"]["unsafe_incorrect_answer"]
            ),
            "candidate_unsafe_answers": candidate_summary["outcomes"]["unsafe_incorrect_answer"],
            "baseline_unsafe_answers": baseline_summary["outcomes"]["unsafe_incorrect_answer"],
            "enhanced_candidate_must_not_increase_unsafe_answers": (
                enhanced_candidate_summary["outcomes"]["unsafe_incorrect_answer"]
                <= baseline_summary["outcomes"]["unsafe_incorrect_answer"]
            ),
            "enhanced_candidate_unsafe_answers": enhanced_candidate_summary["outcomes"][
                "unsafe_incorrect_answer"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation-only Voyage evidence ablation")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="confirm sending questions and candidate policy chunks to Voyage",
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="confirm live structured grading, generation, and citation verification calls",
    )
    parser.add_argument("--limit", type=int, default=0, help="first N eligible cases; 0 runs all")
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="evaluate one golden case ID; repeat to preserve an explicit order",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="use deterministic answer/abstain-stratified case selection",
    )
    parser.add_argument(
        "--answer-cases",
        type=int,
        default=None,
        help="answer cases for --stratified (default: 18)",
    )
    parser.add_argument(
        "--abstain-cases",
        type=int,
        default=None,
        help="abstention cases for --stratified (default: 12)",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="reproducible seed for --stratified (default: 42)",
    )
    parser.add_argument(
        "--voyage-max-retries",
        type=int,
        default=4,
        help="evaluation-only retries for a Voyage 429 (default: 4)",
    )
    parser.add_argument(
        "--voyage-retry-base-seconds",
        type=float,
        default=5.0,
        help="evaluation-only exponential-backoff base without Retry-After (default: 5)",
    )
    parser.add_argument(
        "--voyage-request-delay-seconds",
        type=float,
        default=1.0,
        help="delay between successful Voyage calls in evaluation (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "voyage_evidence_ablation.json",
    )
    args = parser.parse_args()
    if not args.allow_remote or not args.allow_llm:
        parser.error("--allow-remote and --allow-llm are both required for this live evaluation")
    if (
        args.limit < 0
        or (args.answer_cases is not None and args.answer_cases < 0)
        or (args.abstain_cases is not None and args.abstain_cases < 0)
        or args.voyage_max_retries < 0
        or args.voyage_retry_base_seconds < 0
        or args.voyage_request_delay_seconds < 0
    ):
        parser.error("selection counts and Voyage retry/pacing values must be zero or greater")
    try:
        validate_explicit_selection_options(
            case_ids=args.case_id,
            limit=args.limit,
            stratified=args.stratified,
            answer_cases_supplied=args.answer_cases is not None,
            abstain_cases_supplied=args.abstain_cases is not None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.stratified and args.limit:
        parser.error("--limit cannot be combined with --stratified")
    try:
        report = run(
            limit=args.limit,
            case_ids=args.case_id,
            stratified=args.stratified,
            answer_cases=args.answer_cases if args.answer_cases is not None else 18,
            abstain_cases=args.abstain_cases if args.abstain_cases is not None else 12,
            selection_seed=args.selection_seed,
            voyage_max_retries=args.voyage_max_retries,
            voyage_retry_base_seconds=args.voyage_retry_base_seconds,
            voyage_request_delay_seconds=args.voyage_request_delay_seconds,
        )
    except RuntimeError as exc:
        print(f"EVALUATION_BLOCKED: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"safety_gate": report["safety_gate"], "recommendation": report["recommendation"]}, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
