"""Citation verification: the last gate before an answer reaches the user.

Two layers, cheapest first.

1. **Deterministic checks (always run, no quota, no flakiness).**
   - Every cited label must exist in the retrieved set.
   - Every claim sentence must share enough vocabulary with a cited passage.
   - Every identifier or number in a claim (`PAY-402`, `30 calendar days`,
     `15 percent`) must appear literally in a cited passage. This catches the
     most damaging hallucination class in a policy assistant, where a fabricated
     number reads exactly like a correct one.

2. **LLM entailment check (optional).** Used for claims that pass the lexical
   check but may still be unsupported. Off by default in CI, because a
   non-deterministic gate cannot block a merge reliably.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_settings
from src.generation.llm_provider import get_chat_model
from src.generation.prompts import (
    CITATION_CHECK_HUMAN_PROMPT,
    CITATION_CHECK_SYSTEM_PROMPT,
    ENTAILMENT_OUTPUT_SCHEMA,
)
from src.retrieval.bm25 import tokenize
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Numbers, percentages, currency amounts, and alphanumeric codes.
_FACTUAL_TOKEN = re.compile(r"\b(?:[A-Z]{1,5}[-_]?\d{1,5}(?:[-_][A-Z0-9]+)*|\d+(?:\.\d+)?)\b")


@dataclass(slots=True)
class ClaimVerdict:
    claim: str
    supported: bool
    overlap: float
    missing_facts: list[str] = field(default_factory=list)
    method: str = "lexical"


@dataclass(slots=True)
class CitationReport:
    valid: bool
    invalid_labels: list[str] = field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def support_ratio(self) -> float:
        if not self.claim_verdicts:
            return 0.0
        supported = sum(1 for v in self.claim_verdicts if v.supported)
        return supported / len(self.claim_verdicts)

    @property
    def unsupported_claims(self) -> list[str]:
        return [v.claim for v in self.claim_verdicts if not v.supported]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "support_ratio": round(self.support_ratio, 4),
            "invalid_labels": self.invalid_labels,
            "unsupported_claims": self.unsupported_claims,
            "claims": [
                {
                    "claim": v.claim,
                    "supported": v.supported,
                    "overlap": round(v.overlap, 4),
                    "missing_facts": v.missing_facts,
                    "method": v.method,
                }
                for v in self.claim_verdicts
            ],
        }


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(answer.strip()) if len(s.strip()) > 15]


def lexical_overlap(claim: str, passage: str) -> float:
    claim_tokens = set(tokenize(claim))
    if not claim_tokens:
        return 0.0
    passage_tokens = set(tokenize(passage))
    return len(claim_tokens & passage_tokens) / len(claim_tokens)


def missing_factual_tokens(claim: str, passage: str) -> list[str]:
    """Identifiers and numbers present in the claim but absent from the passage."""
    passage_upper = passage.upper()
    facts = dict.fromkeys(_FACTUAL_TOKEN.findall(claim.upper()))
    return [fact for fact in facts if fact not in passage_upper]


def _build_entailment_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", CITATION_CHECK_SYSTEM_PROMPT), ("human", CITATION_CHECK_HUMAN_PROMPT)]
    ).partial(output_schema=ENTAILMENT_OUTPUT_SCHEMA)
    return prompt | get_chat_model("judge") | JsonOutputParser()


def _llm_supports(claim: str, passage: str) -> bool | None:
    try:
        raw = _build_entailment_chain().invoke({"claim": claim, "passage": passage[:2000]})
    except Exception:
        logger.warning("Entailment check failed; falling back to lexical verdict")
        return None
    return bool(raw.get("supported")) if isinstance(raw, dict) else None


def verify_citations(
    answer: str,
    citations: list[str],
    retrieved: list[RetrievedChunk],
    use_llm: bool = False,
) -> CitationReport:
    """Verify that an answer is grounded in the passages it cites."""
    settings = get_settings()

    by_label = {chunk.citation_label: chunk for chunk in retrieved}
    invalid_labels = [label for label in citations if label not in by_label]
    cited_chunks = [by_label[label] for label in citations if label in by_label]

    if not answer.strip():
        return CitationReport(valid=False, invalid_labels=invalid_labels)

    if not cited_chunks:
        # An answer with no usable citation is unverifiable by construction.
        return CitationReport(
            valid=False,
            invalid_labels=invalid_labels,
            claim_verdicts=[
                ClaimVerdict(claim=c, supported=False, overlap=0.0, method="no-citation")
                for c in split_claims(answer)
            ],
        )

    verdicts: list[ClaimVerdict] = []
    for claim in split_claims(answer):
        best_overlap = 0.0
        best_missing: list[str] = []
        best_passage = ""
        for chunk in cited_chunks:
            overlap = lexical_overlap(claim, chunk.content)
            if overlap > best_overlap:
                best_overlap = overlap
                best_missing = missing_factual_tokens(claim, chunk.content)
                best_passage = chunk.content

        supported = (
            best_overlap >= settings.citation_support_threshold and not best_missing
        )
        method = "lexical"

        if not supported and use_llm and best_passage:
            llm_verdict = _llm_supports(claim, best_passage)
            if llm_verdict is not None:
                # The LLM may rescue a paraphrase, but never a missing identifier.
                supported = llm_verdict and not best_missing
                method = "llm-entailment"

        verdicts.append(
            ClaimVerdict(
                claim=claim,
                supported=supported,
                overlap=best_overlap,
                missing_facts=best_missing,
                method=method,
            )
        )

    report = CitationReport(
        valid=False, invalid_labels=invalid_labels, claim_verdicts=verdicts
    )
    # An answer is valid only when no label was invented and every claim holds.
    report.valid = not invalid_labels and all(v.supported for v in verdicts)
    return report
