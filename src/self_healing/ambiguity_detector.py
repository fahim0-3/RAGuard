"""Ambiguity detection: is the question answerable as written?

The failure mode this guards against is not "the question is short". It is "the
question has several policy answers and picking one silently would be a guess".
"Where is my order?" has one answer path; "I have a problem with my order" has
at least three, and choosing tracking over cancellation is a coin flip dressed
as service.

So the rule is *underspecification*, not brevity. A question is ambiguous when
it names a subject area but omits the one detail that selects between materially
different policy answers. Everything else proceeds to retrieval, where weak
evidence is caught by the grader rather than by an interrogation of the
customer.

The detector is deterministic. An LLM classifier here would make clarification
non-reproducible, and clarification is user-visible behaviour that the golden
dataset asserts on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.self_healing.evidence_grader import policy_ids_in
from src.self_healing.state import AmbiguityDecision

logger = logging.getLogger(__name__)

__all__ = ["AmbiguityRule", "detect_ambiguity"]


@dataclass(frozen=True)
class AmbiguityRule:
    """A subject that needs one more detail before it can be answered."""

    name: str
    #: The question is about this subject.
    subject: re.Pattern[str]
    #: Any of these present means the question is already specific enough.
    resolvers: re.Pattern[str]
    clarifying_question: str
    missing_dimension: str


#: Ordered; the first matching rule wins.
AMBIGUITY_RULES: tuple[AmbiguityRule, ...] = (
    AmbiguityRule(
        name="delayed_order",
        subject=re.compile(
            r"\b(delayed|late|delay|held up|still waiting|hasn'?t arrived|"
            r"not arrived|where is my (order|parcel|package))\b",
            re.IGNORECASE,
        ),
        # Stems with \w*, so "refunds" and "tracking" match as readily as
        # "refund" and "track". A trailing \b here silently fails on plurals.
        #
        # The damage terms are resolvers, not an afterthought: this rule's
        # subject pattern matches the bare word "late", which also occurs in
        # "is it too late to report this?" — a damage-reporting-window question,
        # not an underspecified delivery one. Golden case GC-008 was routed to
        # clarification for exactly that reason. The other three rules already
        # treat damage as a resolver; this one was the outlier.
        resolvers=re.compile(
            r"\b(track\w*|cancel\w*|refund\w*|redeliver\w*|miss\w*|lost|"
            r"deliver\w*|date|when will|address|depot|"
            r"damag\w*|broken|crack\w*|faulty|smashed|unbox\w*)\b",
            re.IGNORECASE,
        ),
        clarifying_question=(
            "I can help with a delayed order. Is your question about tracking status, "
            "a missed delivery date, or a request to cancel the order?"
        ),
        missing_dimension="intent: tracking, missed delivery, or cancellation",
    ),
    AmbiguityRule(
        name="unspecified_problem",
        subject=re.compile(
            r"\b(problem|issue|trouble|something (is )?wrong|help me with|"
            r"not happy|complaint)\b",
            re.IGNORECASE,
        ),
        resolvers=re.compile(
            r"\b(refund\w*|return\w*|damag\w*|broken|crack\w*|deliver\w*|payment\w*|"
            r"pay|card|declin\w*|charg\w*|error|warrant\w*|descal\w*|cancel\w*|"
            r"track\w*|lost|exchang\w*|replac\w*)\b",
            re.IGNORECASE,
        ),
        clarifying_question=(
            "I can help with that. Is it about a refund, a return, delivery, a damaged "
            "item, or a payment problem?"
        ),
        missing_dimension="problem category",
    ),
    AmbiguityRule(
        name="bare_timeframe",
        subject=re.compile(
            r"^\s*(how long|how many days|what'?s the (time )?limit|when does it expire)\b",
            re.IGNORECASE,
        ),
        resolvers=re.compile(
            r"\b(refund\w*|return\w*|damag\w*|deliver\w*|payment\w*|claim\w*|"
            r"warrant\w*|replac\w*|exchang\w*|descal\w*|authoris\w*|authoriz\w*)\b",
            re.IGNORECASE,
        ),
        clarifying_question=(
            "Time limits differ by policy. Is this about a refund, a return, "
            "a damaged-item claim, or a delivery?"
        ),
        missing_dimension="which policy the time limit belongs to",
    ),
    AmbiguityRule(
        name="bare_it_reference",
        subject=re.compile(
            r"^\s*(can i (send|give|take) (it|this) back|"
            r"i want to (send|give) (it|this) back|"
            r"what about (it|this))\b",
            re.IGNORECASE,
        ),
        resolvers=re.compile(
            r"\b(damag\w*|broken|crack\w*|faulty|wrong|changed my mind|electronic\w*|"
            r"laptop|phone|machine|refund\w*|hygiene|seal)\b",
            re.IGNORECASE,
        ),
        clarifying_question=(
            "I can help with sending an item back. Is the item damaged or faulty, "
            "the wrong item, or a change of mind?"
        ),
        missing_dimension="return reason",
    ),
)


def detect_ambiguity(question: str) -> AmbiguityDecision:
    """Decide whether the question needs one clarifying reply before retrieval."""
    text = question.strip()

    if not text:
        return AmbiguityDecision(
            ambiguous=True,
            clarifying_question="Could you tell me what you need help with?",
            missing_dimension="the question itself",
            reason="empty question",
        )

    # A named policy or error code makes the target unambiguous whatever else
    # the sentence looks like.
    if policy_ids_in(text):
        return AmbiguityDecision(ambiguous=False, reason="question names a policy identifier")

    for rule in AMBIGUITY_RULES:
        if rule.subject.search(text) and not rule.resolvers.search(text):
            logger.debug("Ambiguity rule %s matched", rule.name)
            return AmbiguityDecision(
                ambiguous=True,
                clarifying_question=rule.clarifying_question,
                missing_dimension=rule.missing_dimension,
                reason=f"underspecified: {rule.name}",
            )

    return AmbiguityDecision(ambiguous=False, reason="question is specific enough to retrieve")
