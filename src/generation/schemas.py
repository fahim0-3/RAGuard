"""Structured generation contracts.

Two schemas, deliberately separate, because they are trusted differently.

`RawAnswerPayload` is what the *model* returns. Every field in it is untrusted:
the model may omit fields, invent citation labels, or claim a confidence it has
not earned. Pydantic here buys shape validation, nothing more.

`AnswerResponse` is what RAGuard returns. Its citations are rebuilt from the
retrieved chunks that were actually supplied to the model, so citation metadata
can never originate from generated text. That is the whole point: a model can
name a passage, but it cannot describe one.

The outcome field is exhaustive on purpose. A provider timeout and an honest
"the evidence does not cover this" are both non-answers, but conflating them
hides outages behind a polite refusal, so they stay distinguishable.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.retrieval.types import RetrievedChunk

__all__ = [
    "AnswerOutcome",
    "AnswerResponse",
    "Citation",
    "RawAnswerPayload",
]

AnswerOutcome = Literal[
    "answered",
    "insufficient_evidence",
    "rejected_invalid_citation",
    "provider_error",
    "invalid_output",
]

#: Outcomes in which no answer text may be shown to a customer.
NON_ANSWER_OUTCOMES: frozenset[str] = frozenset(
    {
        "insufficient_evidence",
        "rejected_invalid_citation",
        "provider_error",
        "invalid_output",
    }
)


class RawAnswerPayload(BaseModel):
    """The model's JSON, validated for shape only. Contents remain untrusted."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    sufficient_context: bool = True
    confidence: float | None = None

    @field_validator("citations", mode="before")
    @classmethod
    def _drop_non_strings(cls, value: Any) -> Any:
        """A model that emits nested objects should not fail the whole parse."""
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str)]
        return value if value is not None else []

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return max(0.0, min(1.0, float(value)))

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> Any:
        return "" if value is None else value


class Citation(BaseModel):
    """A citation, built only from a chunk that was supplied to the model."""

    model_config = ConfigDict(frozen=True)

    citation_label: str
    policy_id: str
    source: str
    chunk_index: int
    chunk_id: int

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> Citation:
        return cls(
            citation_label=chunk.citation_label,
            policy_id=chunk.policy_id,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
        )


class AnswerResponse(BaseModel):
    """RAGuard's answer. Safe to serialise straight to an API or a report."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str = ""
    outcome: AnswerOutcome = "insufficient_evidence"
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_source: Literal["model", "default"] = "default"
    more_info_required: bool = True
    failure_reason: str | None = None
    rejected_citations: list[str] = Field(default_factory=list)
    supplied_chunk_ids: list[int] = Field(default_factory=list)
    supplied_citation_labels: list[str] = Field(default_factory=list)
    prompt_version: str = ""
    model_name: str | None = None

    @property
    def citation_ids(self) -> list[str]:
        """Citation labels, the identifier used everywhere else in the system."""
        return [c.citation_label for c in self.citations]

    @property
    def grounded(self) -> bool:
        """An answer is grounded only if it exists and rests on real citations."""
        return self.outcome == "answered" and bool(self.answer) and bool(self.citations)

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["citation_ids"] = self.citation_ids
        payload["grounded"] = self.grounded
        return payload
