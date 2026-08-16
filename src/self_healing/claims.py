"""Claim extraction: turning an answer into independently checkable units.

Verification operates on claims, not on answers. "Refunds take 5 to 7 business
days. Your refund will definitely arrive tomorrow." is one answer and two
claims, and only the first is in the policy. Grading the answer as a whole would
either accept both or reject both; splitting it makes the second one visible.

Extraction is deterministic. Sentence splitting and type classification are
regex work, and doing them with a model would make verification results vary
between runs, which is the opposite of what a guard rail needs. The model's
judgement is spent on entailment, where it is actually required.

`claim_type` drives how strictly a claim is checked:

- `numeric` and `temporal` carry figures that must appear in the evidence
  verbatim, so they get a hard deterministic gate no model can override.
- `policy` names an identifier such as RT-014, checked the same way.
- `factual` and `procedural` are paraphrasable, so they go to entailment.

No chain-of-thought is stored. A claim records what was asserted and which
passages were cited for it, nothing about how any decision was reached.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Claim",
    "ClaimType",
    "extract_claims",
    "numeric_tokens",
    "policy_ids",
]

ClaimType = Literal["factual", "numeric", "policy", "procedural", "temporal"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Document and rule identifiers: REF-001, RT-014, PAY-402, DMG-RVW-04.
_POLICY_ID = re.compile(r"\b[A-Z]{2,5}-(?:[A-Z]{2,5}-)?\d{1,4}\b")

#: Model and error codes that are not policy IDs: E04, AB-X200-EU.
_CODE = re.compile(r"\b(?:[A-Z]{1,3}\d{2,4}|[A-Z]{2,}-[A-Z0-9]{2,}-[A-Z0-9]{2,})\b")

#: Bare numbers, percentages, and money. Ordinals and decimals included.
_NUMBER = re.compile(
    r"(?:[£$€]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\d[\d,]*(?:\.\d+)?\s?(?:percent|%))"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\b)"
)

_TEMPORAL_HINT = re.compile(
    r"\b(day|days|hour|hours|week|weeks|month|months|year|years|business day|"
    r"calendar day|immediately|within|deadline|expire[sd]?|window)\b",
    re.IGNORECASE,
)

_PROCEDURAL_HINT = re.compile(
    r"\b(step|steps|first|then|next|select|choose|click|open|print|submit|"
    r"contact|follow|complete|hand the parcel|you should|you must|"
    r"you (?:can|need to) (?:start|request|raise))\b",
    re.IGNORECASE,
)

#: Sentences shorter than this are fragments, not claims.
MIN_CLAIM_CHARS = 16


class Claim(BaseModel):
    """One independently checkable assertion from an answer."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    citation_labels: list[str] = Field(default_factory=list)
    claim_type: ClaimType = "factual"

    #: Figures and identifiers that must appear in the cited evidence verbatim.
    required_tokens: list[str] = Field(default_factory=list)

    @property
    def has_citation(self) -> bool:
        return bool(self.citation_labels)

    @property
    def requires_exact_evidence(self) -> bool:
        """True when a model may not paraphrase its way to "supported"."""
        return self.claim_type in {"numeric", "temporal", "policy"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def policy_ids(text: str) -> list[str]:
    """Policy, rule, and product codes named in a string."""
    upper = text.upper()
    found = list(_POLICY_ID.findall(upper))
    found.extend(code for code in _CODE.findall(upper) if code not in found)
    return list(dict.fromkeys(found))


def numeric_tokens(text: str) -> list[str]:
    """Numbers, money, and percentages, normalised for comparison."""
    tokens = []
    for match in _NUMBER.finditer(text):
        token = match.group(0).strip()
        # Strip separators so "1,500" and "1500" compare equal.
        tokens.append(token.replace(",", "").replace(" ", ""))
    return list(dict.fromkeys(tokens))


def classify(sentence: str) -> ClaimType:
    """Assign the strictest type the sentence qualifies for."""
    if policy_ids(sentence):
        return "policy"
    has_number = bool(numeric_tokens(sentence))
    if has_number and _TEMPORAL_HINT.search(sentence):
        return "temporal"
    if has_number:
        return "numeric"
    if _PROCEDURAL_HINT.search(sentence):
        return "procedural"
    return "factual"


def _required_tokens(sentence: str, claim_type: ClaimType) -> list[str]:
    """What must be present verbatim in the evidence for this claim to hold."""
    if claim_type == "policy":
        return policy_ids(sentence)
    if claim_type in {"numeric", "temporal"}:
        # Identifiers too: "rule RT-014 allows 14 days" is both.
        return [*policy_ids(sentence), *numeric_tokens(sentence)]
    return []


def extract_claims(answer: str, citation_labels: list[str] | None = None) -> list[Claim]:
    """Split an answer into typed, citation-bearing claims.

    Every claim inherits the answer's citation list. The generator cites for the
    answer as a whole rather than per sentence, so a claim is checked against
    all cited passages collectively — which is also what lets one claim be
    supported by two chunks at once.
    """
    labels = list(citation_labels or [])
    text = (answer or "").strip()
    if not text:
        return []

    claims: list[Claim] = []
    for index, sentence in enumerate(_SENTENCE_SPLIT.split(text), start=1):
        sentence = sentence.strip()
        if len(sentence) < MIN_CLAIM_CHARS:
            continue
        claim_type = classify(sentence)
        claims.append(
            Claim(
                claim_id=f"c{index}",
                claim_text=sentence,
                citation_labels=labels,
                claim_type=claim_type,
                required_tokens=_required_tokens(sentence, claim_type),
            )
        )
    return claims
