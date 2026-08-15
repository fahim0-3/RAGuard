"""Citation-constrained answer generation (LCEL).

The chain is `prompt | model | JsonOutputParser`. Structured output is not a
stylistic preference here: the self-healing layer needs machine-readable
citations and an explicit `sufficient_context` flag to decide whether to retry
or abstain. Free-text answers cannot be verified automatically.

A parse failure is deliberately treated as insufficient context rather than
raising. A malformed answer is exactly the situation in which the system should
abstain rather than emit unverified text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.generation.llm_provider import get_chat_model
from src.generation.prompts import (
    ANSWER_HUMAN_PROMPT,
    ANSWER_OUTPUT_SCHEMA,
    ANSWER_SYSTEM_PROMPT,
    PROMPT_VERSION,
)
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnswerDraft:
    answer: str
    citations: list[str] = field(default_factory=list)
    sufficient_context: bool = True
    parse_failed: bool = False
    prompt_version: str = PROMPT_VERSION


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render passages with the exact citation labels the model must reuse."""
    if not chunks:
        return "(no passages retrieved)"
    blocks = [
        f"[{index}] citation_label: {chunk.citation_label}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_answer_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM_PROMPT), ("human", ANSWER_HUMAN_PROMPT)]
    ).partial(output_schema=ANSWER_OUTPUT_SCHEMA)
    return prompt | get_chat_model("generator") | JsonOutputParser()


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> AnswerDraft:
    """Generate a grounded answer over the supplied passages."""
    if not chunks:
        return AnswerDraft(answer="", citations=[], sufficient_context=False)

    valid_labels = {chunk.citation_label for chunk in chunks}
    try:
        raw = build_answer_chain().invoke(
            {"context": format_context(chunks), "question": question}
        )
    except Exception:
        logger.exception("Answer generation failed; treating as insufficient context")
        return AnswerDraft(answer="", sufficient_context=False, parse_failed=True)

    if not isinstance(raw, dict):
        logger.warning("Unexpected generator output type: %s", type(raw))
        return AnswerDraft(answer="", sufficient_context=False, parse_failed=True)

    citations = [c for c in raw.get("citations") or [] if isinstance(c, str)]
    hallucinated = [c for c in citations if c not in valid_labels]
    if hallucinated:
        # A label the model invented is itself a grounding failure signal; drop it
        # here and let the citation verifier decide what the answer is worth.
        logger.warning("Dropping citation labels not present in context: %s", hallucinated)

    return AnswerDraft(
        answer=(raw.get("answer") or "").strip(),
        citations=[c for c in citations if c in valid_labels],
        sufficient_context=bool(raw.get("sufficient_context", True)),
    )
