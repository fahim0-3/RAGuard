"""Phase H: frontend response handling.

The UI's decisions live in `frontend.presenter` as pure functions, so they are
tested here without Streamlit, a browser, or a running API. What is asserted is
the behaviour that matters to a user: an abstention must not look like an
error, an escalation must not look like an answer, and citation metadata must
be whatever the server said it was.

`frontend.app` itself is not imported: importing it executes Streamlit page
setup at module scope.
"""

from __future__ import annotations

import pytest

from frontend.presenter import (
    OUTCOME_STYLES,
    TRACE_LABELS,
    citation_rows,
    error_view,
    present,
    trace_rows,
    verification_note,
)


def api_response(**overrides) -> dict:
    payload = {
        "request_id": "req-1",
        "outcome": "answer",
        "answer": "Refunds take 5 to 7 business days.",
        "citations": [
            {
                "citation_label": "refund_policy.txt#2",
                "policy_id": "REF-001",
                "source": "refund_policy.txt",
                "chunk_index": 2,
                "chunk_id": 40,
                "excerpt": "Credit and debit cards: 5 to 7 business days.",
            }
        ],
        "confidence": 0.9,
        "more_info_required": False,
        "retry_count": 0,
        "max_retries": 2,
        "rewritten_queries": [],
        "risk_level": "none",
        "verification_status": "supported",
        "verified_claim_count": 1,
        "unsupported_claim_count": 0,
        "retrieved_chunk_count": 5,
        "reranker_used": True,
        "failure_reason": None,
        "trace": [
            {"step": 1, "node": "sanitize_and_classify"},
            {"step": 2, "node": "verify_citations"},
        ],
        "latency_ms": 1500.0,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Outcome handling
# --------------------------------------------------------------------------


def test_answer_is_presented_as_success():
    view = present(api_response())

    assert view.kind == "success"
    assert view.outcome == "answer"
    assert view.body == "Refunds take 5 to 7 business days."
    assert view.is_error is False


def test_clarify_is_informational_not_a_failure():
    view = present(
        api_response(outcome="clarify", answer="Is this about a refund or a return?")
    )

    assert view.kind == "info"
    assert view.is_error is False
    assert "guess" in view.explanation.lower()


def test_abstain_is_a_warning_not_an_error():
    """An abstention is the system working, and must not read as a crash."""
    view = present(api_response(outcome="abstain", answer="I don't have enough information."))

    assert view.kind == "warning"
    assert view.is_error is False


def test_escalate_is_distinct_from_abstain():
    view = present(api_response(outcome="escalate", answer="Passing this to our team."))

    assert view.outcome == "escalate"
    assert view.kind == "warning"
    assert view.heading != OUTCOME_STYLES["abstain"].heading


def test_unknown_outcome_falls_back_to_error():
    view = present(api_response(outcome="something_new"))

    assert view.kind == "error"


@pytest.mark.parametrize("outcome", ["answer", "clarify", "abstain", "escalate", "error"])
def test_every_outcome_has_a_style(outcome):
    assert outcome in OUTCOME_STYLES
    assert OUTCOME_STYLES[outcome].heading
    assert OUTCOME_STYLES[outcome].explanation


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_api_error_payload_becomes_an_error_view():
    view = present({"error": "provider_unavailable", "detail": "Provider is unavailable."})

    assert view.is_error is True
    assert view.kind == "error"
    assert view.body == "Provider is unavailable."


def test_unreachable_api_is_an_error_view():
    view = error_view("Could not reach the API: connection refused")

    assert view.is_error is True
    assert "connection refused" in view.body


def test_error_view_has_no_metrics_or_citations():
    view = present({"error": "internal_error", "detail": "Something failed."})

    assert view.citations == []
    assert view.metrics == {}


# --------------------------------------------------------------------------
# Citations come from the server, never reconstructed
# --------------------------------------------------------------------------


def test_citation_metadata_is_taken_verbatim():
    rows = citation_rows(api_response())

    assert rows[0]["policy_id"] == "REF-001"
    assert rows[0]["source"] == "refund_policy.txt"
    assert rows[0]["chunk_id"] == 40
    assert rows[0]["excerpt"]


def test_missing_citation_fields_do_not_crash_the_view():
    rows = citation_rows({"citations": [{"citation_label": "x#1"}]})

    assert rows[0]["policy_id"] == ""
    assert rows[0]["chunk_id"] is None


def test_no_citations_yields_no_rows():
    assert citation_rows(api_response(citations=[])) == []


# --------------------------------------------------------------------------
# Self-healing surface
# --------------------------------------------------------------------------


def test_retry_information_is_surfaced():
    view = present(api_response(retry_count=2, rewritten_queries=["refund processing time"]))

    assert view.metrics["Retries"] == "2/2"
    assert view.rewritten_queries == ["refund processing time"]


def test_metrics_are_formatted_for_display():
    view = present(api_response(confidence=0.876, latency_ms=2500.0))

    assert view.metrics["Confidence"] == "0.88"
    assert view.metrics["Latency"] == "2.5s"


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


def test_trace_nodes_get_readable_labels():
    rows = trace_rows(api_response())

    assert rows[0]["label"] == "Sanitised input"
    assert rows[1]["label"] == "Citation verification"


def test_unknown_trace_node_still_renders():
    rows = trace_rows({"trace": [{"step": 1, "node": "brand_new_node"}]})

    assert rows[0]["label"] == "Brand New Node"


def test_every_graph_node_has_a_trace_label():
    """A new node must not appear in the UI as a raw identifier."""
    from src.self_healing.graph import NODE_NAMES

    for node in NODE_NAMES:
        assert node in TRACE_LABELS, f"{node} has no display label"


def test_trace_contains_no_reasoning():
    view = present(api_response())

    blob = str(view.trace).lower()
    for leak in ("prompt", "reasoning", "chain_of_thought", "because"):
        assert leak not in blob


# --------------------------------------------------------------------------
# Verification note
# --------------------------------------------------------------------------


def test_supported_verification_note():
    assert "verified" in verification_note(api_response()).lower()


def test_unsupported_verification_note_explains_the_withholding():
    note = verification_note(
        api_response(verification_status="unsupported", unsupported_claim_count=2)
    )

    assert "withheld" in note.lower()
    assert "2" in note


def test_unchecked_verification_note():
    note = verification_note(api_response(verification_status="not_checked"))

    assert "no answer reached" in note.lower()
