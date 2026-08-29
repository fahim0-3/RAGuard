"""Abstention: what to say when the evidence will not support an answer.

An abstention is a product surface, not an error path. The customer sees it, so
it has to be useful: say plainly that the answer is not in the available
policies, name the missing detail when one is known, and route onward. What it
must never do is soften the refusal into a hedged half-answer, because a hedged
half-answer is a policy claim with a disclaimer attached.

The messages are templates over the grader's `missing_information`, so the
specific missing detail comes from the graded evidence rather than from a model
writing a fresh apology each time.
"""

from __future__ import annotations

import re
from typing import Literal

from src.self_healing.state import EvidenceGrade

__all__ = [
    "ABSTENTION_BASE",
    "AbstainReason",
    "abstention_message",
    "customer_missing_information",
]

AbstainReason = Literal[
    "no_evidence",
    "insufficient_evidence",
    "retries_exhausted",
    "generation_failed",
    "unverified_citations",
    "provider_error",
    "request_budget_exhausted",
]

ABSTENTION_BASE = (
    "I don't have enough information in the available policies to answer that accurately."
)

_CONTACT = "Please contact support"

# Evidence grading deliberately fails closed when structured model execution is
# unavailable.  These operational diagnostics are useful in logs and state,
# but they are neither a customer-actionable missing detail nor safe retrieval
# vocabulary for a retry query.
_INTERNAL_DIAGNOSTIC = re.compile(
    r"\b(?:semantic\s+evidence\s+grad(?:er|ing)|provider|request\s+budget|"
    r"invalid\s+output|malformed\s+output)\b",
    re.IGNORECASE,
)

_REASON_SUFFIX: dict[str, str] = {
    "no_evidence": (
        f"{_CONTACT}, or rephrase the question using the wording that appears in the "
        "policy you have in mind."
    ),
    "insufficient_evidence": (f"{_CONTACT}, or add the missing detail so I can look again."),
    "retries_exhausted": (
        f"{_CONTACT}. I rephrased the search and still could not find a policy passage "
        "that covers this."
    ),
    "generation_failed": (
        f"{_CONTACT}. I found relevant policy text but could not produce a grounded answer from it."
    ),
    "unverified_citations": (
        f"{_CONTACT}. I could not confirm that the answer I drafted was supported by "
        "the policy text, so I am not showing it."
    ),
    "provider_error": (
        f"{_CONTACT}. The answering service is temporarily unavailable, so I cannot "
        "answer right now."
    ),
    "request_budget_exhausted": (
        f"{_CONTACT}. The request reached its safe processing limit before the answer "
        "could be fully verified."
    ),
}


def customer_missing_information(missing_information: list[str] | None) -> list[str]:
    """Return only customer-actionable evidence gaps.

    The original values remain on ``EvidenceGrade`` for internal diagnostics;
    this is strictly a presentation and retry-query boundary.
    """
    return [
        text
        for item in missing_information or []
        if (text := " ".join(str(item).split())) and not _INTERNAL_DIAGNOSTIC.search(text)
    ]


def abstention_message(reason: AbstainReason, grade: EvidenceGrade | None = None) -> str:
    """Compose the customer-facing abstention.

    Names the specific missing detail when the grader identified one, which is
    the difference between a useful refusal and a dead end.
    """
    parts = [ABSTENTION_BASE]

    missing = customer_missing_information(grade.missing_information if grade else None)
    if missing:
        # One detail, not a list: an abstention that asks for five things reads
        # as an interrogation rather than a next step.
        parts.append(f"To answer this I would need: {missing[0]}.")

    parts.append(_REASON_SUFFIX.get(reason, _REASON_SUFFIX["insufficient_evidence"]))
    return " ".join(parts)
