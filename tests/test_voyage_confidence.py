from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.calibrate_voyage_confidence import (
    rerank_voyage_with_rate_limit_retry,
    retry_after_seconds,
    select_cases,
)
from src.evaluation.voyage_confidence import (
    VoyageConfidenceProfile,
    case_diagnostic,
    choose_safety_first_profile,
    decision_metrics,
    split_composition,
    stratified_split_assignments,
    top_score_threshold_candidates,
    voyage_confidence,
)


def profile(**overrides):
    values = {
        "profile_id": "test-v1",
        "status": "draft",
        "provider": "voyage",
        "model": "rerank-2.5-lite",
        "dataset_version": "test-v1",
        "corpus_fingerprint": "sha256:test",
        "chunking_fingerprint": "chunk_size=800;chunk_overlap=120",
        "candidate_top_k": 20,
        "final_top_k": 5,
        "permit_score_min": 0.50,
        "support_score_min": 0.50,
        "min_supporting_chunks": 2,
        "ambiguous_margin_max": 0.10,
        "minimum_answer_recall": 0.80,
        "maximum_abstention_false_permit_rate": 0.0,
        "calibration_split": "stratified-v2",
        "development_split": "stratified-v2",
        "held_out_split": "stratified-v2-locked",
    }
    return VoyageConfidenceProfile(**(values | overrides))


def test_voyage_values_never_mutate_or_require_bge_chunk_score_fields():
    result = voyage_confidence([0.82, 0.75, 0.21], profile())

    assert result.level == "high"
    assert result.supporting_chunks == 2
    assert result.deterministic_eligible is True


def test_voyage_binary_eligibility_uses_only_the_top_score():
    rejected = voyage_confidence([0.45, 0.44], profile())
    accepted_despite_diagnostics = voyage_confidence([0.55, 0.54], profile())

    assert rejected.level == "insufficient"
    assert rejected.rejection_reason == "top_score_below_permit_score_min"
    assert accepted_despite_diagnostics.deterministic_eligible is True
    assert accepted_despite_diagnostics.margin_diagnostic_passes is False
    assert accepted_despite_diagnostics.support_count_diagnostic_passes is True


def test_profile_requires_exact_environment_compatibility():
    candidate = profile()
    metadata = {
        "provider": "voyage",
        "model": "rerank-2.5-lite",
        "dataset_version": "test-v1",
        "corpus_fingerprint": "sha256:test",
        "chunking_fingerprint": "chunk_size=800;chunk_overlap=120",
        "candidate_top_k": 20,
        "final_top_k": 5,
    }
    assert candidate.compatible_with(metadata)
    assert not candidate.compatible_with(metadata | {"model": "another-model"})


def test_split_is_deterministic_stratified_and_keeps_locked_held_out_data():
    cases = [
        {
            "id": f"GC-{index:03d}",
            "expected_outcome": outcome,
            "case_type": case_type,
            "category": category,
        }
        for index, (outcome, case_type, category) in enumerate(
            [
                *(('answer', 'normal', 'refund') for _ in range(5)),
                *(('abstain', 'unanswerable', 'near_miss') for _ in range(3)),
                *(('abstain', 'prompt_injection', 'security') for _ in range(3)),
            ],
            start=1,
        )
    ]

    first = stratified_split_assignments(cases)
    assert first == stratified_split_assignments(list(reversed(cases)))
    composition = split_composition(cases, first)
    assert set(first.values()) == {"calibration", "development", "held_out"}
    assert composition["held_out"]["answer"] >= 1
    assert composition["held_out"]["abstain"] >= 1


def test_calibration_uses_ground_truth_not_bge_rejection_for_false_permits():
    records = [
        {
            "id": "answer-bge-rejects",
            "scores": [0.90, 0.80],
            "expected_outcome": "answer",
            "bge_permits": False,
        },
        {
            "id": "abstain",
            "scores": [0.10, 0.08],
            "expected_outcome": "abstain",
            "bge_permits": False,
        },
    ]
    metrics = decision_metrics(records, profile(permit_score_min=0.5))

    assert metrics["confusion_matrix"] == {"TP": 1, "FP": 0, "TN": 1, "FN": 0}
    assert metrics["abstention_false_permit_rate"] == 0.0
    assert metrics["bge_decision_agreement"] == 0.5


def test_reject_all_cannot_satisfy_minimum_answer_recall():
    records = [
        {
            "scores": [0.7],
            "expected_outcome": "answer",
            "bge_permits": True,
        }
    ]

    metrics = decision_metrics(records, profile(permit_score_min=1.0))

    assert metrics["answer_recall"] == 0.0
    assert metrics["answer_recall"] < profile().minimum_answer_recall


def test_threshold_candidates_use_only_rank_one_scores():
    records = [
        {"scores": [0.91, 0.123, 0.12]},
        {"scores": [0.72, 0.456, 0.20]},
    ]

    assert top_score_threshold_candidates(records) == [0.72, 0.91]


def test_impossible_constraints_return_no_valid_profile():
    metadata = {
        "provider": "voyage",
        "model": "rerank-2.5-lite",
        "dataset_version": "test-v1",
        "corpus_fingerprint": "sha256:test",
        "chunking_fingerprint": "chunk_size=800;chunk_overlap=120",
        "candidate_top_k": 20,
        "final_top_k": 5,
    }
    records = [
        {
            "id": "answer-low",
            "scores": [0.4, 0.2],
            "expected_outcome": "answer",
            "bge_permits": True,
        },
        {
            "id": "abstain-high",
            "scores": [0.9, 0.1],
            "expected_outcome": "abstain",
            "bge_permits": False,
        },
    ]

    selected, fit = choose_safety_first_profile(
        records,
        metadata=metadata,
        minimum_answer_recall=0.8,
        maximum_abstention_false_permit_rate=0.0,
    )

    assert selected is None
    assert fit["status"] == "no_valid_profile"
    assert fit["valid_candidate_count"] == 0


def test_every_case_receives_complete_diagnostics():
    records = [
        {
            "id": "answer",
            "split": "calibration",
            "expected_outcome": "answer",
            "case_type": "normal",
            "confidence_category": "answer:normal",
            "scores": [0.9, 0.7, 0.4, 0.3, 0.2],
            "bge_permits": False,
        },
        {
            "id": "abstain",
            "split": "held_out",
            "expected_outcome": "abstain",
            "case_type": "unanswerable",
            "confidence_category": "abstain:near_miss",
            "scores": [0.2, 0.1],
            "bge_permits": True,
        },
    ]

    diagnostics = [case_diagnostic(record, profile()) for record in records]

    assert [row["id"] for row in diagnostics] == ["answer", "abstain"]
    assert all("confidence_predicates" in row for row in diagnostics)
    assert all("rejection_reason" in row for row in diagnostics)
    assert len(diagnostics[0]["voyage_top_5_scores"]) == 5


def test_held_out_data_never_participates_in_threshold_selection():
    metadata = {
        "provider": "voyage",
        "model": "rerank-2.5-lite",
        "dataset_version": "test-v1",
        "corpus_fingerprint": "sha256:test",
        "chunking_fingerprint": "chunk_size=800;chunk_overlap=120",
        "candidate_top_k": 20,
        "final_top_k": 5,
    }
    calibration = [
        {
            "id": "cal-answer",
            "split": "calibration",
            "scores": [0.9],
            "expected_outcome": "answer",
            "bge_permits": True,
        },
        {
            "id": "cal-abstain",
            "split": "calibration",
            "scores": [0.1],
            "expected_outcome": "abstain",
            "bge_permits": False,
        },
    ]
    held_out = {
        "id": "held-out",
        "split": "held_out",
        "scores": [0.99],
        "expected_outcome": "abstain",
        "bge_permits": False,
    }

    selected, fit = choose_safety_first_profile(
        [*calibration, held_out], metadata=metadata
    )

    assert selected is not None
    assert fit["selection_case_ids"] == ["cal-answer", "cal-abstain"]
    assert held_out["id"] not in fit["selection_case_ids"]


def test_calibration_limit_preserves_order_and_zero_means_all():
    cases = [{"id": "GC-001"}, {"id": "GC-002"}, {"id": "GC-003"}]

    assert select_cases(cases, 0) == cases
    assert select_cases(cases, 2) == cases[:2]


def test_calibration_rate_limit_retries_same_voyage_case_with_retry_after():
    class FakeVoyage:
        def __init__(self):
            self.calls = 0

        def rerank_with_diagnostics(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                reranker_used=self.calls == 2,
                failure="voyage_rate_limited" if self.calls == 1 else None,
            )

    runner = FakeVoyage()
    sleeps: list[float] = []
    messages: list[str] = []
    result, retries, events = rerank_voyage_with_rate_limit_retry(
        runner,
        "q",
        [],
        top_k=5,
        candidate_top_k=20,
        max_retries=4,
        retry_base_seconds=5.0,
        retry_after=lambda: 7.0,
        progress=messages.append,
        prefix="[5/15]",
        sleep=sleeps.append,
    )

    assert result.reranker_used is True
    assert runner.calls == 2
    assert retries == 1
    assert sleeps == [7.0]
    assert events == [{"retry": 1, "delay_seconds": 7.0, "source": "retry_after"}]
    assert messages == ["[5/15] Voyage rate limited - retry 1/4 in 7.0s"]


def test_calibration_rate_limit_exhaustion_fails_without_a_local_substitute():
    class AlwaysLimitedVoyage:
        def rerank_with_diagnostics(self, *_args, **_kwargs):
            return SimpleNamespace(reranker_used=False, failure="voyage_rate_limited")

    with pytest.raises(RuntimeError, match="voyage_rate_limited"):
        rerank_voyage_with_rate_limit_retry(
            AlwaysLimitedVoyage(),
            "q",
            [],
            top_k=5,
            candidate_top_k=20,
            max_retries=1,
            retry_base_seconds=2.0,
            retry_after=lambda: None,
            progress=lambda _message: None,
            prefix="[1/1]",
            sleep=lambda _seconds: None,
        )


def test_retry_after_accepts_seconds_and_rejects_invalid_values():
    assert retry_after_seconds("5") == 5.0
    assert retry_after_seconds("not-a-date") is None
