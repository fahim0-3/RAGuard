"""Grounded answer generation (Phase E).

Pipeline for one question:

    final top-5 chunks -> prompt -> model -> JSON -> Pydantic -> citation check

Three properties matter more than the wording of the prompt, because a prompt
is a request and these are guarantees:

**Context restriction.** The model sees the final reranked chunks and nothing
else. Not the corpus, not the BM25 list, not the dense list, not the RRF
candidates. The chunks supplied are recorded on the response by ID, so the
evidence behind any answer is reconstructable after the fact.

**Citation metadata is never generated.** The model may only name a passage by
its `citation_label`. Every other field of a citation — policy ID, source file,
chunk index, chunk ID — is copied from the retrieved chunk that label resolves
to. A model cannot invent a policy ID it is never asked to produce.

**An invented citation rejects the answer.** Not "drop the bad label and keep
the prose": a model citing a passage it was not given is a grounding failure,
and the remaining text carries no more warrant than the discarded label did.

Provider failures never become answers. A timeout, a rate limit, and an honest
"the evidence does not cover this" are all non-answers, but they are reported as
different outcomes so an outage cannot hide behind a polite refusal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.config import get_settings
from src.generation.llm_factory import LLMProviderError, get_chat_model, model_name_for
from src.generation.prompts import (
    ANSWER_HUMAN_PROMPT,
    ANSWER_OUTPUT_SCHEMA,
    ANSWER_SYSTEM_PROMPT,
    PROMPT_VERSION,
)
from src.generation.schemas import AnswerResponse, Citation, ClaimCitation, RawAnswerPayload
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerDraft",
    "build_answer_chain",
    "format_context",
    "generate_answer",
    "generate_grounded_answer",
    "restrict_context",
    "validate_claim_citations",
    "validate_citations",
]


@dataclass(slots=True)
class AnswerDraft:
    """Legacy view of :class:`AnswerResponse`, kept for the self-healing layer."""

    answer: str
    citations: list[str] = field(default_factory=list)
    sufficient_context: bool = True
    parse_failed: bool = False
    prompt_version: str = PROMPT_VERSION


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


def restrict_context(
    chunks: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Cut the evidence to the final top-k the model is allowed to see."""
    limit = top_k if top_k is not None else get_settings().rerank_top_k
    return list(chunks[:limit])


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render passages with the exact citation labels the model must reuse."""
    if not chunks:
        return "(no passages retrieved)"
    return "\n\n".join(
        f"[{index}] citation_label: {chunk.citation_label}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )


# --------------------------------------------------------------------------
# Citation safety
# --------------------------------------------------------------------------


#: Strips the decoration a model may echo from the rendered context line,
#: for example "[3] citation_label: refund_policy.txt#1". Smaller models copy
#: the whole line rather than the label.
_LABEL_DECORATION = re.compile(
    r"^\s*(?:\[\d+\]\s*)?(?:citation[_ ]label\s*[:=]\s*)?", re.IGNORECASE
)


def normalise_label(label: str) -> str:
    """Reduce a model-written label to the bare citation label.

    This repairs formatting only. The result must still match a supplied label
    exactly, so normalisation cannot admit a passage the model was not given.
    """
    return _LABEL_DECORATION.sub("", label).strip().strip("\"'").strip()


def validate_citations(
    labels: list[str], chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Resolve labels against supplied chunks.

    Returns the resolved citations and any label that did not correspond to a
    chunk actually given to the model. Metadata comes from the chunk, so no
    field of a returned citation originates in generated text.
    """
    by_label = {chunk.citation_label: chunk for chunk in chunks}
    resolved: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for label in labels:
        chunk = by_label.get(normalise_label(label))
        if chunk is None:
            # Report the label as the model wrote it, so the trace shows what
            # was actually returned rather than a cleaned-up version.
            invalid.append(label)
            continue
        if chunk.citation_label in seen:
            continue
        seen.add(chunk.citation_label)
        resolved.append(Citation.from_chunk(chunk))

    return resolved, invalid


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _answer_sentences(answer: str) -> list[str]:
    """Every non-empty sentence is a separately grounded assertion.

    Do not use a minimum length here. A brief sentence such as "Fees apply."
    can carry an unsupported policy claim and must not bypass verification.
    """
    return [" ".join(part.split()) for part in _SENTENCE_SPLIT.split(answer.strip()) if part.strip()]


def validate_claim_citations(
    answer: str, claim_citations: list[ClaimCitation], chunks: list[RetrievedChunk]
) -> tuple[list[ClaimCitation], list[Citation], list[str], str | None]:
    """Validate an exact, per-sentence mapping from answer text to evidence.

    The response may carry a convenient flattened citation list, but that list
    is derived from this map. A citation for one sentence can no longer silently
    warrant a different sentence in the answer.
    """
    expected_claims = _answer_sentences(answer)
    supplied_claims = [claim.claim for claim in claim_citations]
    if not expected_claims or supplied_claims != expected_claims:
        return [], [], [], "claim citations must match every answer sentence in order"

    canonical: list[ClaimCitation] = []
    resolved: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for claim in claim_citations:
        claim_resolved, claim_invalid = validate_citations(claim.citations, chunks)
        invalid.extend(claim_invalid)
        if not claim_resolved:
            return [], [], invalid, "every answer sentence requires a supplied citation"
        labels = [citation.citation_label for citation in claim_resolved]
        canonical.append(ClaimCitation(claim=claim.claim, citations=labels))
        for citation in claim_resolved:
            if citation.citation_label not in seen:
                seen.add(citation.citation_label)
                resolved.append(citation)

    if invalid:
        return [], [], invalid, "one or more claim citations were not supplied"
    return canonical, resolved, [], None


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------


def build_answer_chain() -> Any:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM_PROMPT), ("human", ANSWER_HUMAN_PROMPT)]
    ).partial(output_schema=ANSWER_OUTPUT_SCHEMA)
    return prompt | get_chat_model("generator") | JsonOutputParser()


def _failure(
    question: str,
    outcome: str,
    reason: str,
    supplied: list[RetrievedChunk],
    *,
    rejected: list[str] | None = None,
) -> AnswerResponse:
    """Build a non-answer. No answer text, more information always required."""
    return AnswerResponse(
        question=question,
        answer="",
        outcome=outcome,  # type: ignore[arg-type]
        citations=[],
        confidence=0.0,
        confidence_source="default",
        more_info_required=True,
        failure_reason=reason,
        rejected_citations=rejected or [],
        supplied_chunk_ids=[c.chunk_id for c in supplied],
        supplied_citation_labels=[c.citation_label for c in supplied],
        prompt_version=PROMPT_VERSION,
    )


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    chain: Any | None = None,
) -> AnswerResponse:
    """Answer `question` using only `chunks`, validated end to end.

    `chain` is injectable so the deterministic tests can exercise every branch
    without a provider or an API key.
    """
    supplied = restrict_context(chunks, top_k)

    if not supplied:
        return _failure(
            question, "insufficient_evidence", "no passages were retrieved", supplied
        )

    try:
        chain = chain if chain is not None else build_answer_chain()
    except LLMProviderError as exc:
        logger.error("Generator unavailable: %s", exc)
        return _failure(question, "provider_error", "language model provider unavailable", supplied)

    try:
        raw = chain.invoke({"context": format_context(supplied), "question": question})
    except LLMProviderError as exc:
        logger.error("Generator unavailable: %s", exc)
        return _failure(question, "provider_error", "language model provider unavailable", supplied)
    except Exception:  # noqa: BLE001 - provider SDKs raise their own types
        # Timeouts, rate limits, transport errors. Never an answer.
        logger.exception("Answer generation failed")
        return _failure(question, "provider_error", "language model request failed", supplied)

    try:
        payload = RawAnswerPayload.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Generator returned an unusable payload: %s", exc)
        return _failure(question, "invalid_output", "model output did not match the required schema", supplied)

    answer_text = payload.answer.strip()

    if not payload.sufficient_context or not answer_text:
        return _failure(
            question,
            "insufficient_evidence",
            "the model reported that the evidence does not cover the question",
            supplied,
        )

    claim_citations, citations, invalid, claim_error = validate_claim_citations(
        answer_text, payload.claim_citations, supplied
    )
    if invalid:
        logger.warning("Rejecting answer citing unsupplied passages: %s", invalid)
        return _failure(
            question,
            "rejected_invalid_citation",
            "answer cited evidence that was not supplied",
            supplied,
            rejected=invalid,
        )
    if claim_error or not citations:
        return _failure(
            question,
            "rejected_invalid_citation",
            claim_error or "answer supplied no citation, so it cannot be traced to evidence",
            supplied,
        )

    return AnswerResponse(
        question=question,
        answer=answer_text,
        outcome="answered",
        citations=citations,
        claim_citations=claim_citations,
        confidence=payload.confidence if payload.confidence is not None else 0.0,
        confidence_source="model" if payload.confidence is not None else "default",
        more_info_required=False,
        supplied_chunk_ids=[c.chunk_id for c in supplied],
        supplied_citation_labels=[c.citation_label for c in supplied],
        prompt_version=PROMPT_VERSION,
        model_name=_safe_model_name(),
    )


def _safe_model_name() -> str | None:
    """Model ID for the trace. Never raises, never touches credentials."""
    try:
        return model_name_for("generator")
    except Exception:  # noqa: BLE001 - reporting must not break generation
        return None


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> AnswerDraft:
    """Legacy entry point used by the self-healing layer.

    Any non-answer becomes `sufficient_context=False`, so the caller fails
    closed: a rejected citation or a provider outage abstains rather than
    emitting unverified text.
    """
    response = generate_grounded_answer(question, chunks)
    return AnswerDraft(
        answer=response.answer,
        citations=response.citation_ids,
        sufficient_context=response.outcome == "answered",
        parse_failed=response.outcome in {"invalid_output", "provider_error"},
        prompt_version=response.prompt_version,
    )
