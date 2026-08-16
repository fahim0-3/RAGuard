"""Pure presentation logic for the Streamlit UI.

Streamlit widgets are awkward to unit test, so every decision the UI makes
lives here as functions over the API response and the widgets render what these
return. That keeps the interesting part — how an abstention differs from an
escalation, what a failed verification should say — under test without a
browser or a running server.

The frontend never recomputes anything. Citation metadata, confidence, and the
verification verdict all come from the validated API response; reconstructing
them client-side would create a second, unverified source of truth for exactly
the claims the system exists to guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "OUTCOME_STYLES",
    "OutcomeView",
    "TRACE_LABELS",
    "citation_rows",
    "present",
    "trace_rows",
    "verification_note",
]

Kind = Literal["success", "info", "warning", "error"]


@dataclass(frozen=True)
class OutcomeStyle:
    heading: str
    kind: Kind
    explanation: str


#: How each terminal outcome is presented. The wording matters: an abstention
#: is not an error, and an escalation is not a failure.
OUTCOME_STYLES: dict[str, OutcomeStyle] = {
    "answer": OutcomeStyle(
        heading="Answered from policy",
        kind="success",
        explanation="Every claim below was checked against the cited policy text.",
    ),
    "clarify": OutcomeStyle(
        heading="Needs one detail",
        kind="info",
        explanation="The question matches several policies. Answering it as written would be a guess.",
    ),
    "abstain": OutcomeStyle(
        heading="Not answered",
        kind="warning",
        explanation="The available policies do not cover this, so no answer was produced.",
    ),
    "escalate": OutcomeStyle(
        heading="Passed to a person",
        kind="warning",
        explanation="This was routed to a human before any policy answer was generated.",
    ),
    "error": OutcomeStyle(
        heading="Something went wrong",
        kind="error",
        explanation="The request could not be completed.",
    ),
}

#: Human labels for graph nodes, so the trace reads as a decision path.
TRACE_LABELS: dict[str, str] = {
    "sanitize_and_classify": "Sanitised input",
    "risk_router": "Risk screening",
    "ambiguity_detector": "Ambiguity check",
    "hybrid_retrieve": "Hybrid retrieval",
    "rerank": "Cross-encoder rerank",
    "evidence_grader": "Evidence grading",
    "query_rewriter": "Query rewrite (retry)",
    "generate_answer": "Answer generation",
    "verify_citations": "Citation verification",
    "finalize_answer": "Answer released",
    "abstain": "Abstained",
    "clarify": "Clarification requested",
    "escalate": "Escalated",
    "_count_regeneration": "Regeneration allowed",
}


@dataclass
class OutcomeView:
    """Everything the UI needs to render one response."""

    heading: str
    kind: Kind
    explanation: str
    outcome: str
    body: str
    request_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    verification: str = ""
    failure_reason: str | None = None
    is_error: bool = False


def error_view(message: str, request_id: str = "") -> OutcomeView:
    """Transport-level failure: the API was unreachable or returned an error."""
    style = OUTCOME_STYLES["error"]
    return OutcomeView(
        heading=style.heading,
        kind="error",
        explanation=style.explanation,
        outcome="error",
        body=message,
        request_id=request_id,
        is_error=True,
    )


def citation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Citation metadata exactly as the server validated it."""
    rows = []
    for citation in payload.get("citations") or []:
        rows.append(
            {
                "label": citation.get("citation_label", ""),
                "policy_id": citation.get("policy_id", ""),
                "source": citation.get("source", ""),
                "chunk_index": citation.get("chunk_index"),
                "chunk_id": citation.get("chunk_id"),
                "excerpt": citation.get("excerpt", ""),
            }
        )
    return rows


def trace_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The executed node path, labelled. Operational state only."""
    rows = []
    for step in payload.get("trace") or []:
        node = step.get("node", "")
        rows.append(
            {
                "step": step.get("step"),
                "node": node,
                "label": TRACE_LABELS.get(node, node.replace("_", " ").title()),
            }
        )
    return rows


def verification_note(payload: dict[str, Any]) -> str:
    """One line describing what verification concluded."""
    status = payload.get("verification_status", "not_checked")
    supported = int(payload.get("verified_claim_count") or 0)
    unsupported = int(payload.get("unsupported_claim_count") or 0)

    if status == "supported":
        return f"All {supported} claim(s) verified against the cited policy text."
    if status == "unsupported":
        return (
            f"{unsupported} claim(s) could not be verified against the citations, "
            "so the draft answer was withheld."
        )
    return "No answer reached the citation-verification stage."


def present(payload: dict[str, Any]) -> OutcomeView:
    """Turn an API response into the view the UI renders."""
    if payload.get("error"):
        return error_view(
            payload.get("detail") or str(payload.get("error")),
            request_id=str(payload.get("request_id") or ""),
        )

    outcome = str(payload.get("outcome") or "error")
    style = OUTCOME_STYLES.get(outcome, OUTCOME_STYLES["error"])

    return OutcomeView(
        heading=style.heading,
        kind=style.kind,
        explanation=style.explanation,
        outcome=outcome,
        body=payload.get("answer") or "",
        request_id=str(payload.get("request_id") or ""),
        metrics={
            "Confidence": f"{float(payload.get('confidence') or 0.0):.2f}",
            "Retries": f"{payload.get('retry_count', 0)}/{payload.get('max_retries', 0)}",
            "Passages": payload.get("retrieved_chunk_count", 0),
            "Latency": f"{float(payload.get('latency_ms') or 0.0) / 1000:.1f}s",
        },
        citations=citation_rows(payload),
        trace=trace_rows(payload),
        rewritten_queries=list(payload.get("rewritten_queries") or []),
        verification=verification_note(payload),
        failure_reason=payload.get("failure_reason"),
        is_error=outcome == "error",
    )
