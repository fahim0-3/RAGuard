"""Query rewriting for the retry step.

When retrieval confidence is weak, the usual cause is vocabulary mismatch: the
customer writes "they sent me a smashed mug", the policy says "concealed damage
must be reported within 7 calendar days". Rewriting produces variants closer to
the corpus vocabulary, and the retriever fuses all variants with RRF.

A deterministic fallback is provided so the retry path still functions without
an API key. That matters for CI: the rewrite step must be exercisable in a test
that consumes no provider quota.
"""

from __future__ import annotations

import logging
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_settings
from src.generation.llm_provider import get_chat_model
from src.generation.prompts import (
    QUERY_REWRITE_HUMAN_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    REWRITE_OUTPUT_SCHEMA,
)
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Colloquial phrasing observed in support tickets, mapped to corpus vocabulary.
_DOMAIN_SYNONYMS: dict[str, str] = {
    "money back": "refund eligibility",
    "my money": "refund processing time",
    "send back": "return process",
    "sent back": "return process",
    "broken": "damaged product claim",
    "smashed": "damaged product claim",
    "cracked": "damaged product claim",
    "never arrived": "lost parcel investigation",
    "not turned up": "lost parcel investigation",
    "did not arrive": "delivery non-receipt claim",
    "card declined": "payment failure gateway error code",
    "payment failed": "payment failure gateway error code",
    "wont turn on": "product manual error code",
    "how long": "time window policy",
}

_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]{1,5}[-_]?\d{2,5}(?:[-_][A-Z0-9]+)*\b")


def extract_identifiers(text: str) -> list[str]:
    """Pull out codes such as PAY-402, RF-101, or AB-X200-EU."""
    return list(dict.fromkeys(_IDENTIFIER_PATTERN.findall(text.upper())))


def heuristic_rewrites(question: str, n_variants: int) -> list[str]:
    """Deterministic fallback: synonym substitution plus identifier isolation."""
    lowered = question.lower()
    variants: list[str] = []

    for phrase, replacement in _DOMAIN_SYNONYMS.items():
        if phrase in lowered:
            variants.append(lowered.replace(phrase, replacement))

    for identifier in extract_identifiers(question):
        variants.append(f"{identifier} error code meaning and resolution")

    # Last resort: strip question framing so BM25 sees only content words.
    stripped = re.sub(
        r"^(what|how|when|why|who|where|can|do|does|is|are|i|my|the)\b\s*",
        "",
        lowered,
    ).strip()
    if stripped and stripped != lowered:
        variants.append(stripped)

    deduped = [v for v in dict.fromkeys(variants) if v and v != lowered]
    return deduped[:n_variants]


def _build_rewrite_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", QUERY_REWRITE_SYSTEM_PROMPT), ("human", QUERY_REWRITE_HUMAN_PROMPT)]
    ).partial(output_schema=REWRITE_OUTPUT_SCHEMA)
    return prompt | get_chat_model("rewriter") | JsonOutputParser()


def rewrite_query(
    question: str,
    weak_chunks: list[RetrievedChunk] | None = None,
    n_variants: int | None = None,
    use_llm: bool = True,
) -> list[str]:
    """Return alternative queries, excluding the original.

    Falls back to heuristics when the LLM is unavailable or returns nothing
    usable, so the caller never has to handle an empty result specially.
    """
    settings = get_settings()
    n_variants = n_variants or settings.query_rewrite_variants

    fallback = heuristic_rewrites(question, n_variants)

    if not use_llm:
        return fallback

    weak_context = "(nothing retrieved)"
    if weak_chunks:
        weak_context = weak_chunks[0].content[:600]

    try:
        raw = _build_rewrite_chain().invoke(
            {
                "question": question,
                "weak_context": weak_context,
                "n_variants": n_variants,
            }
        )
    except Exception:
        logger.warning("LLM rewrite failed; using heuristic variants", exc_info=True)
        return fallback

    queries = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(queries, list):
        return fallback

    cleaned = [
        q.strip()
        for q in queries
        if isinstance(q, str) and q.strip() and q.strip().lower() != question.lower()
    ]
    return list(dict.fromkeys(cleaned))[:n_variants] or fallback
