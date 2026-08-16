"""Phase D: the deterministic metric primitives.

These functions decide whether a build is green. If one of them is wrong, every
number downstream is wrong and nobody finds out, so they are tested directly and
exhaustively rather than through the evaluation that uses them.
"""

from __future__ import annotations

import pytest

from src.evaluation.deterministic_metrics import (
    LatencyStats,
    citation_id_validity,
    find_regressions,
    invalid_citation_ids,
    latency_stats,
    percentile,
    structured_output_errors,
    structured_output_validity,
)

KNOWN_IDS = {"REF-001", "RET-002", "DMG-003", "DEL-004", "PAY-005", "MAN-006"}


def valid_answer(**overrides) -> dict:
    payload = {
        "answer": "Card refunds take 5 to 7 business days.",
        "citations": ["refund_policy.txt#1"],
        "abstained": False,
        "confidence": 0.82,
        "outcome": "answer",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Citation ID validity
# --------------------------------------------------------------------------


def test_all_known_ids_are_fully_valid():
    assert citation_id_validity(["REF-001", "RET-002"], KNOWN_IDS) == 1.0


def test_invented_id_reduces_validity():
    assert citation_id_validity(["REF-001", "REF-999"], KNOWN_IDS) == pytest.approx(0.5)


def test_all_invented_ids_score_zero():
    assert citation_id_validity(["XX-000"], KNOWN_IDS) == 0.0


def test_no_citations_is_not_applicable_rather_than_zero():
    """A correct abstention cites nothing and must not be scored as inventing."""
    assert citation_id_validity([], KNOWN_IDS) is None


def test_invalid_ids_are_reported_in_order_with_duplicates():
    assert invalid_citation_ids(["REF-001", "ZZ-1", "ZZ-1"], KNOWN_IDS) == ["ZZ-1", "ZZ-1"]


# --------------------------------------------------------------------------
# Structured output validity
# --------------------------------------------------------------------------


def test_well_formed_payload_has_no_errors():
    assert structured_output_errors(valid_answer()) == []


def test_non_object_payload_is_rejected():
    assert structured_output_errors("just a string")


@pytest.mark.parametrize(
    "field", ["answer", "citations", "abstained", "confidence", "outcome"]
)
def test_every_required_field_is_enforced(field):
    payload = valid_answer()
    del payload[field]
    errors = structured_output_errors(payload)
    assert any(field in e for e in errors)


def test_confidence_outside_the_unit_interval_is_rejected():
    assert any("[0, 1]" in e for e in structured_output_errors(valid_answer(confidence=1.4)))
    assert any("[0, 1]" in e for e in structured_output_errors(valid_answer(confidence=-0.1)))


def test_boolean_confidence_is_rejected():
    """bool subclasses int, so a naive isinstance check would let True through."""
    assert any("bool" in e for e in structured_output_errors(valid_answer(confidence=True)))


def test_unknown_outcome_is_rejected():
    errors = structured_output_errors(valid_answer(outcome="maybe"))
    assert any("outcome must be one of" in e for e in errors)


def test_non_string_citation_is_rejected():
    errors = structured_output_errors(valid_answer(citations=["ok", 7]))
    assert any("citations must all be strings" in e for e in errors)


def test_abstention_carrying_citations_is_rejected():
    payload = valid_answer(abstained=True, outcome="abstain", citations=["refund_policy.txt#1"])
    assert any("abstention must not carry citations" in e for e in structured_output_errors(payload))


def test_empty_non_abstaining_answer_is_rejected():
    payload = valid_answer(answer="   ")
    assert any("must not be empty" in e for e in structured_output_errors(payload))


def test_validity_is_the_fraction_of_well_formed_payloads():
    payloads = [valid_answer(), valid_answer(confidence=2.0), valid_answer()]
    assert structured_output_validity(payloads) == pytest.approx(2 / 3)


def test_validity_of_nothing_is_zero():
    assert structured_output_validity([]) == 0.0


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------


def test_percentile_uses_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 0.5) == 50
    assert percentile(values, 0.95) == 100
    assert percentile(values, 1.0) == 100


def test_percentile_of_a_single_sample():
    assert percentile([42.0], 0.5) == 42.0


def test_percentile_of_nothing_is_zero():
    assert percentile([], 0.5) == 0.0


def test_percentile_rejects_an_impossible_fraction():
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0], 0.0)


def test_latency_stats_are_computed_over_the_distribution():
    stats = latency_stats([100.0, 200.0, 300.0, 400.0])
    assert stats.count == 4
    assert stats.mean_ms == pytest.approx(250.0)
    assert stats.p50_ms == 200.0
    assert stats.max_ms == 400.0


def test_latency_stats_of_nothing_are_zero():
    assert latency_stats([]) == LatencyStats(0, 0.0, 0.0, 0.0, 0.0)


def test_latency_ordering_does_not_affect_the_result():
    assert latency_stats([300.0, 100.0, 200.0]).to_dict() == latency_stats(
        [100.0, 200.0, 300.0]
    ).to_dict()


# --------------------------------------------------------------------------
# Regression counting
# --------------------------------------------------------------------------


def test_metric_below_target_is_a_regression():
    regressions = find_regressions({"mrr_at_5": 0.70}, {"mrr_at_5": 0.84})
    assert len(regressions) == 1
    assert regressions[0].shortfall == pytest.approx(0.14)


def test_metric_at_target_is_not_a_regression():
    assert find_regressions({"mrr_at_5": 0.84}, {"mrr_at_5": 0.84}) == []


def test_tolerance_absorbs_small_drops():
    assert find_regressions({"mrr_at_5": 0.83}, {"mrr_at_5": 0.84}, tolerance=0.02) == []
    assert find_regressions({"mrr_at_5": 0.81}, {"mrr_at_5": 0.84}, tolerance=0.02)


def test_unmeasured_metric_is_not_counted_as_a_regression():
    """Unknown is not the same as failed; conflating them fails builds wrongly."""
    assert find_regressions({}, {"mrr_at_5": 0.84}) == []


def test_regressions_are_ordered_worst_first():
    regressions = find_regressions(
        {"a": 0.9, "b": 0.1, "c": 0.5}, {"a": 1.0, "b": 1.0, "c": 1.0}
    )
    assert [r.metric for r in regressions] == ["b", "c", "a"]


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        find_regressions({"a": 1.0}, {"a": 1.0}, tolerance=-0.1)


def test_regression_renders_a_readable_message():
    regression = find_regressions({"mrr_at_5": 0.70}, {"mrr_at_5": 0.84})[0]
    assert "mrr_at_5" in str(regression)
    assert "0.7000" in str(regression)
