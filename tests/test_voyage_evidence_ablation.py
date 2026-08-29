from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from scripts.evaluate_voyage_evidence_ablation import (
    _evaluate_in_graph_context,
    _with_llm_permit,
    pace_voyage_request,
    rerank_voyage_with_rate_limit_retry,
    select_cases,
    select_explicit_cases,
    select_stratified_cases,
    selection_metadata,
    validate_explicit_selection_options,
)
from src.config import Settings
from src.evaluation.voyage_evidence_ablation import (
    PathEvaluation,
    SanityGateResult,
    aggregate_paired_timing,
    aggregate_path,
    candidate_minimal_sanity_gate,
    disagreement_diagnostics,
    evaluate_path,
)
from src.generation.llm_routing import current_provider, current_route
from src.generation.schemas import AnswerResponse, Citation
from src.reranking.provider import ConfiguredReranker
from src.retrieval.types import RetrievedChunk
from src.self_healing.execution_budget import current_budget
from src.self_healing.state import EvidenceGrade, VerificationResult


def chunk(chunk_id: int = 1, *, content: str = "Policy REF-001 permits refunds.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source="refund_policy.txt",
        chunk_index=0,
        doc_id="REF-001",
    )


def answered_response(item: RetrievedChunk) -> AnswerResponse:
    return AnswerResponse(
        question="q",
        answer="Refunds are permitted.",
        outcome="answered",
        citations=[Citation.from_chunk(item)],
        more_info_required=False,
    )


def sufficient_grade() -> EvidenceGrade:
    return EvidenceGrade(relevant=True, sufficient=True, confidence=0.95)


def supported_verification() -> VerificationResult:
    return VerificationResult(supported=True, checked=True, verifier="deterministic")


def selection_case(case_id: str, outcome: str, category: str, *, case_type: str = "normal"):
    return {
        "id": case_id,
        "expected_outcome": outcome,
        "category": category,
        "case_type": case_type,
    }


def test_stratified_selection_respects_requested_outcome_counts_and_has_no_duplicates():
    cases = [
        *(selection_case(f"A-{index}", "answer", "normal") for index in range(5)),
        *(selection_case(f"N-{index}", "abstain", "near_miss") for index in range(2)),
        *(selection_case(f"W-{index}", "abstain", "wrong_policy") for index in range(2)),
        *(selection_case(f"O-{index}", "abstain", "out_of_scope") for index in range(2)),
    ]

    selected = select_stratified_cases(
        cases, answer_cases=3, abstain_cases=4, selection_seed=42
    )

    assert sum(case["expected_outcome"] == "answer" for case in selected) == 3
    assert sum(case["expected_outcome"] == "abstain" for case in selected) == 4
    assert len({case["id"] for case in selected}) == len(selected)


def test_stratified_selection_is_seeded_and_distributes_abstention_categories():
    cases = [
        *(selection_case(f"A-{index}", "answer", "normal") for index in range(4)),
        selection_case("N-1", "abstain", "near_miss"),
        selection_case("W-1", "abstain", "wrong_policy"),
        selection_case("O-1", "abstain", "out_of_scope"),
        selection_case("I-1", "abstain", "security", case_type="prompt_injection"),
    ]

    first = select_stratified_cases(cases, answer_cases=2, abstain_cases=4, selection_seed=99)
    second = select_stratified_cases(cases, answer_cases=2, abstain_cases=4, selection_seed=99)
    metadata = selection_metadata(first, selection_mode="stratified", selection_seed=99)

    assert [case["id"] for case in first] == [case["id"] for case in second]
    assert metadata["abstention_category_counts"] == {
        "injection_adversarial": 1,
        "near_miss": 1,
        "out_of_scope": 1,
        "wrong_policy": 1,
    }
    assert metadata["seed"] == 99


def test_limit_selection_remains_the_legacy_eligible_prefix():
    cases = [
        selection_case("skip", "clarify", "normal"),
        selection_case("A-1", "answer", "normal"),
        selection_case("X-1", "abstain", "near_miss"),
        selection_case("A-2", "answer", "normal"),
    ]

    assert [case["id"] for case in select_cases(cases, 2)] == ["A-1", "X-1"]


def test_explicit_case_selection_preserves_one_or_multiple_supplied_ids():
    cases = [
        selection_case("GC-001", "answer", "normal"),
        selection_case("GC-002", "abstain", "near_miss"),
        selection_case("GC-003", "answer", "normal"),
    ]

    one = select_explicit_cases(cases, ["GC-002"])
    multiple = select_explicit_cases(cases, ["GC-003", "GC-001", "GC-002"])
    metadata = selection_metadata(
        multiple, selection_mode="explicit_case_ids", selection_seed=None
    )

    assert [case["id"] for case in one] == ["GC-002"]
    assert [case["id"] for case in multiple] == ["GC-003", "GC-001", "GC-002"]
    assert metadata["selected_case_ids"] == ["GC-003", "GC-001", "GC-002"]
    assert metadata["case_count"] == 3


def test_explicit_case_selection_rejects_unknown_and_duplicate_ids():
    cases = [selection_case("GC-001", "answer", "normal")]

    with pytest.raises(ValueError, match="unknown --case-id value"):
        select_explicit_cases(cases, ["GC-404"])
    with pytest.raises(ValueError, match="duplicate --case-id value"):
        select_explicit_cases(cases, ["GC-001", "GC-001"])


@pytest.mark.parametrize(
    ("limit", "stratified", "answer_cases_supplied", "abstain_cases_supplied", "conflict"),
    [
        (1, False, False, False, "--limit"),
        (0, True, False, False, "--stratified"),
        (0, False, True, False, "--answer-cases"),
        (0, False, False, True, "--abstain-cases"),
    ],
)
def test_explicit_case_selection_rejects_other_selection_options(
    limit, stratified, answer_cases_supplied, abstain_cases_supplied, conflict
):
    with pytest.raises(ValueError, match=conflict):
        validate_explicit_selection_options(
            case_ids=["GC-001"],
            limit=limit,
            stratified=stratified,
            answer_cases_supplied=answer_cases_supplied,
            abstain_cases_supplied=abstain_cases_supplied,
        )


def test_existing_selection_modes_do_not_conflict_without_explicit_ids():
    validate_explicit_selection_options(case_ids=None, limit=1, stratified=False)
    validate_explicit_selection_options(case_ids=None, limit=0, stratified=True)


def test_candidate_minimal_gate_does_not_use_bge_scores_or_a_relevance_threshold():
    item = chunk()
    result = candidate_minimal_sanity_gate("What does REF-001 say?", [item], {1: 0.001})

    assert result.permitted is True
    assert result.reason == "minimal_sanity_passed"
    assert item.normalised_rerank_score is None


def test_candidate_minimal_gate_rejects_only_unusable_retrieval_states():
    item = chunk()

    assert not candidate_minimal_sanity_gate("q", [], {}).permitted
    assert not candidate_minimal_sanity_gate("q", [item], {}).permitted
    assert not candidate_minimal_sanity_gate("What does PAY-999 mean?", [item], {1: 0.9}).permitted


def test_evaluation_classifies_a_verified_answer_to_an_abstention_case_as_unsafe():
    item = chunk()
    result = evaluate_path(
        case_id="GC-test",
        question="q",
        expected_outcome="abstain",
        path="candidate",
        chunks=[item],
        gate=SanityGateResult(True, "minimal_sanity_passed", {}),
        grade_evidence=sufficient_grade,
        generate_answer=lambda: answered_response(item),
        verify=lambda _response: supported_verification(),
        shared_latency_ms=1.0,
        component_latency_ms={},
    )

    assert result.final_outcome == "answer"
    assert result.classification == "unsafe_incorrect_answer"
    assert result.citation_supported is True


def test_evidence_grader_rejection_fails_closed_without_generation():
    generated = False

    def generation_should_not_run() -> AnswerResponse:
        nonlocal generated
        generated = True
        return answered_response(chunk())

    result = evaluate_path(
        case_id="GC-test",
        question="q",
        expected_outcome="answer",
        path="candidate",
        chunks=[chunk()],
        gate=SanityGateResult(True, "minimal_sanity_passed", {}),
        grade_evidence=lambda: EvidenceGrade(relevant=True, sufficient=False, confidence=0.2),
        generate_answer=generation_should_not_run,
        verify=lambda _response: supported_verification(),
        shared_latency_ms=0.0,
        component_latency_ms={},
    )

    assert generated is False
    assert result.final_outcome == "abstain"
    assert result.classification == "unnecessary_abstention"
    assert result.evidence_grader_rejected is True


def test_path_timers_are_non_overlapping_and_end_to_end_is_accounted():
    item = chunk()

    def timed_grade() -> EvidenceGrade:
        import time

        time.sleep(0.003)
        return sufficient_grade()

    def timed_generation() -> AnswerResponse:
        import time

        time.sleep(0.003)
        return answered_response(item)

    def timed_verify(_response: AnswerResponse) -> VerificationResult:
        import time

        time.sleep(0.003)
        return supported_verification()

    result = evaluate_path(
        case_id="GC-timing",
        question="q",
        expected_outcome="answer",
        path="baseline",
        chunks=[item],
        gate=SanityGateResult(True, "current_bge_deterministic_gate", {}),
        grade_evidence=timed_grade,
        generate_answer=timed_generation,
        verify=timed_verify,
        shared_latency_ms=12.0,
        component_latency_ms={"bge_fixed_order_confidence_ms": 8.0},
        initial_path_after_shared_ms=8.0,
    )

    assert result.shared_retrieval_voyage_ms == 12.0
    assert result.end_to_end_ms == pytest.approx(
        result.shared_retrieval_voyage_ms + result.path_after_shared_ms
    )
    assert result.latency_ms == result.end_to_end_ms
    assert set(result.component_latency_ms) == {
        "bge_fixed_order_confidence_ms",
        "evidence_grader_ms",
        "generation_ms",
        "citation_verifier_ms",
    }
    assert sum(result.component_latency_ms.values()) == pytest.approx(
        result.path_after_shared_ms, abs=5.0
    )


def test_candidate_timing_cannot_include_bge_and_paired_metrics_separate_it():
    item = chunk()
    with pytest.raises(ValueError, match="non-baseline evaluation must not invoke"):
        evaluate_path(
            case_id="GC-timing",
            question="q",
            expected_outcome="answer",
            path="candidate",
            chunks=[item],
            gate=SanityGateResult(True, "minimal_sanity_passed", {}),
            grade_evidence=sufficient_grade,
            generate_answer=lambda: answered_response(item),
            verify=lambda _response: supported_verification(),
            shared_latency_ms=10.0,
            component_latency_ms={"bge_fixed_order_confidence_ms": 2.0},
        )

    baseline = PathEvaluation(
        case_id="GC-paired",
        path="baseline",
        expected_outcome="answer",
        final_outcome="answer",
        classification="correct_answer",
        reason="verified_answer",
        shared_retrieval_voyage_ms=10.0,
        path_after_shared_ms=15.0,
        end_to_end_ms=25.0,
        latency_ms=25.0,
        component_latency_ms={"bge_fixed_order_confidence_ms": 8.0},
    )
    candidate = PathEvaluation(
        case_id="GC-paired",
        path="candidate",
        expected_outcome="answer",
        final_outcome="answer",
        classification="correct_answer",
        reason="verified_answer",
        shared_retrieval_voyage_ms=10.0,
        path_after_shared_ms=11.0,
        end_to_end_ms=21.0,
        latency_ms=21.0,
        component_latency_ms={"candidate_sanity_gate_ms": 1.0},
    )
    paired = aggregate_paired_timing([baseline], [candidate])

    assert paired["bge_incremental_cost_ms"]["mean"] == 8.0
    assert paired["paired_end_to_end_delta_ms"]["mean"] == -4.0
    assert paired["cases"][0]["candidate_extra_downstream_cost_ms"] == 11.0


def test_aggregate_reports_final_outcomes_not_pre_grader_permits():
    item = chunk()
    baseline = evaluate_path(
        case_id="GC-001",
        question="q",
        expected_outcome="answer",
        path="baseline",
        chunks=[item],
        gate=SanityGateResult(True, "current_bge_deterministic_gate", {}),
        grade_evidence=sufficient_grade,
        generate_answer=lambda: answered_response(item),
        verify=lambda _response: supported_verification(),
        shared_latency_ms=0.0,
        component_latency_ms={},
    )
    candidate = evaluate_path(
        case_id="GC-001",
        question="q",
        expected_outcome="answer",
        path="candidate",
        chunks=[item],
        gate=SanityGateResult(False, "voyage_scores_missing_or_invalid", {}),
        grade_evidence=sufficient_grade,
        generate_answer=lambda: answered_response(item),
        verify=lambda _response: supported_verification(),
        shared_latency_ms=0.0,
        component_latency_ms={},
    )

    summary = aggregate_path([candidate])
    disagreements = disagreement_diagnostics([baseline], [candidate])

    assert summary["outcomes"]["unnecessary_abstention"] == 1
    assert summary["llm_calls"] == {
        "evidence_grader": 0,
        "generation": 0,
        "citation_entailment": 0,
    }
    assert len(disagreements) == 1


def test_evaluation_module_cannot_replace_the_production_voyage_bge5_path():
    provider_source = inspect.getsource(ConfiguredReranker)

    assert "voyage_evidence_ablation" not in provider_source
    assert "score_fixed_order_with_diagnostics" in provider_source


@pytest.mark.parametrize("path", ["baseline", "candidate"])
def test_each_evaluation_path_runs_inside_a_graph_route_and_budget(path):
    settings = Settings(
        _env_file=None,
        llm_routing_mode="dynamic",
        google_api_key="g" * 32,
        groq_api_key="r" * 32,
        graph_request_timeout_s=60,
        graph_llm_call_limit=3,
    )
    seen: dict[str, object] = {}

    def evaluate() -> PathEvaluation:
        seen["provider"] = current_provider(settings)
        seen["route"] = current_route()
        seen["budget"] = current_budget()
        _with_llm_permit(settings, "evidence_grader", lambda _permit: None)
        return PathEvaluation(
            case_id="GC-test",
            path=path,
            expected_outcome="answer",
            final_outcome="abstain",
            classification="unnecessary_abstention",
            reason="test",
        )

    result = _evaluate_in_graph_context(settings, evaluate)

    assert seen["provider"] == "groq"
    assert seen["route"] is not None
    assert seen["budget"] is not None
    assert result.execution_context["route"]["workload"] == "evaluation"
    assert result.execution_context["budget"]["llm_calls_used"] == 1


def test_evaluation_paces_between_successful_voyage_requests_only():
    sleeps: list[float] = []
    messages: list[str] = []

    pace_voyage_request(
        previous_success=True,
        delay_seconds=1.5,
        prefix="[2/10]",
        progress=messages.append,
        sleep=sleeps.append,
    )
    pace_voyage_request(
        previous_success=False,
        delay_seconds=1.5,
        prefix="[1/10]",
        progress=messages.append,
        sleep=sleeps.append,
    )

    assert sleeps == [1.5]
    assert messages == ["[2/10] Voyage pacing - waiting 1.5s"]


def test_evaluation_retries_same_voyage_case_using_retry_after_without_bge_substitution():
    class FakeVoyage:
        def __init__(self):
            self.calls = 0
            self.bge_fallback_calls = 0

        def rerank_with_diagnostics(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                reranker_used=self.calls == 2,
                failure="voyage_rate_limited" if self.calls == 1 else None,
            )

        def local_bge_fallback(self):
            self.bge_fallback_calls += 1

    voyage = FakeVoyage()
    sleeps: list[float] = []
    messages: list[str] = []
    result, retries, events = rerank_voyage_with_rate_limit_retry(
        voyage,
        "q",
        [],
        top_k=5,
        candidate_top_k=20,
        max_retries=4,
        retry_base_seconds=5.0,
        retry_after=lambda: 7.0,
        progress=messages.append,
        prefix="[10/10]",
        sleep=sleeps.append,
    )

    assert result.reranker_used is True
    assert voyage.calls == 2
    assert voyage.bge_fallback_calls == 0
    assert retries == 1
    assert sleeps == [7.0]
    assert events == [{"retry": 1, "delay_seconds": 7.0, "source": "retry_after"}]
    assert messages == ["[10/10] Voyage rate limited - retry 1/4 in 7.0s"]


def test_evaluation_uses_bounded_exponential_backoff_without_retry_after():
    class LimitedThenSuccessfulVoyage:
        def __init__(self):
            self.calls = 0

        def rerank_with_diagnostics(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                reranker_used=self.calls == 3,
                failure="voyage_rate_limited" if self.calls < 3 else None,
            )

    sleeps: list[float] = []
    _, retries, events = rerank_voyage_with_rate_limit_retry(
        LimitedThenSuccessfulVoyage(),
        "q",
        [],
        top_k=5,
        candidate_top_k=20,
        max_retries=4,
        retry_base_seconds=5.0,
        retry_after=lambda: None,
        progress=lambda _message: None,
        prefix="[1/1]",
        sleep=sleeps.append,
    )

    assert retries == 2
    assert sleeps == [5.0, 10.0]
    assert [event["source"] for event in events] == [
        "exponential_backoff",
        "exponential_backoff",
    ]


def test_evaluation_rate_limit_exhaustion_and_unrelated_failures_are_not_retried():
    class AlwaysLimited:
        def __init__(self):
            self.calls = 0

        def rerank_with_diagnostics(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(reranker_used=False, failure="voyage_rate_limited")

    limited = AlwaysLimited()
    with pytest.raises(RuntimeError, match="voyage_rate_limited"):
        rerank_voyage_with_rate_limit_retry(
            limited,
            "q",
            [],
            top_k=5,
            candidate_top_k=20,
            max_retries=1,
            retry_base_seconds=1.0,
            retry_after=lambda: None,
            progress=lambda _message: None,
            prefix="[1/1]",
            sleep=lambda _seconds: None,
        )
    assert limited.calls == 2

    class UnavailableVoyage:
        def __init__(self):
            self.calls = 0

        def rerank_with_diagnostics(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(reranker_used=False, failure="voyage_unavailable")

    unavailable = UnavailableVoyage()
    with pytest.raises(RuntimeError, match="voyage_unavailable"):
        rerank_voyage_with_rate_limit_retry(
            unavailable,
            "q",
            [],
            top_k=5,
            candidate_top_k=20,
            max_retries=4,
            retry_base_seconds=1.0,
            retry_after=lambda: None,
            progress=lambda _message: None,
            prefix="[1/1]",
            sleep=lambda _seconds: None,
        )
    assert unavailable.calls == 1
