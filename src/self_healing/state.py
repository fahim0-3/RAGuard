"""Typed state for the self-healing graph.

The graph state is a `TypedDict` because that is what LangGraph merges between
nodes: each node returns a partial dict and the runtime applies it. The
structured values *inside* the state are Pydantic models, so a grade or a
verdict cannot be a loosely shaped dict that drifts between nodes.

Nothing here stores chain-of-thought. `EvidenceGrade.rationale` and
`RiskAssessment.reason` are short operational strings meant for an on-call
engineer reading a trace, not a record of model deliberation. The distinction
matters: a trace is read by people who did not run the query, and hidden
reasoning in it is both a privacy problem and a prompt-injection surface.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.retrieval.types import RetrievedChunk

__all__ = [
    "AmbiguityDecision",
    "EvidenceGrade",
    "FinalOutcome",
    "GraphState",
    "RiskAssessment",
    "RiskLevel",
    "VerificationResult",
    "initial_state",
]

#: Terminal outcomes. Every route through the graph ends on exactly one.
FinalOutcome = Literal["answer", "clarify", "abstain", "escalate", "error"]

RiskLevel = Literal["none", "low", "high"]

#: Longest rationale kept in state. A cap is the cheapest structural guard
#: against a model narrating its reasoning into an operational field.
MAX_RATIONALE_CHARS = 240


class _Truncating(BaseModel):
    """Base for models whose free-text fields are operational, not narrative."""

    model_config = ConfigDict(extra="ignore")

    @staticmethod
    def _short(value: str) -> str:
        value = " ".join(str(value).split())
        return value[:MAX_RATIONALE_CHARS]


class EvidenceGrade(_Truncating):
    """Verdict on whether the retrieved evidence can support an answer."""

    relevant: bool = False
    sufficient: bool = False
    confidence: float = 0.0
    missing_information: list[str] = Field(default_factory=list)
    rationale: str = ""

    #: Deterministic measurements behind the decision. Numbers only.
    signals: dict[str, Any] = Field(default_factory=dict)
    #: True when no structured grader ran and the decision is deterministic.
    deterministic_only: bool = False

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("rationale")
    @classmethod
    def _cap(cls, value: str) -> str:
        return cls._short(value)

    @field_validator("missing_information", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [" ".join(str(v).split())[:120] for v in value if str(v).strip()][:5]


class AmbiguityDecision(_Truncating):
    """Whether the question can be acted on as written."""

    ambiguous: bool = False
    clarifying_question: str = ""
    missing_dimension: str = ""
    reason: str = ""

    @field_validator("reason", "clarifying_question")
    @classmethod
    def _cap(cls, value: str) -> str:
        return cls._short(value)


class RiskAssessment(_Truncating):
    """Whether the question must leave the automated path."""

    level: RiskLevel = "none"
    category: str = ""
    reason: str = ""

    @field_validator("reason")
    @classmethod
    def _cap(cls, value: str) -> str:
        return cls._short(value)


class VerificationResult(_Truncating):
    """Citation verification outcome.

    Phase F filled this from the lexical verifier; Phase G adds semantic
    entailment. The field set grew, but every Phase F field kept its name and
    meaning, so the graph's routing and the `Verifier` protocol are unchanged.
    """

    supported: bool = False
    checked: bool = False
    verifier: str = "deterministic"
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    support_ratio: float = 0.0
    reason: str = ""

    # --- Phase G ---
    confidence: float = 0.0
    #: What a claim needed but no cited passage contained, for example a number
    #: or a policy ID that appears in the answer and nowhere in the evidence.
    missing_evidence: list[str] = Field(default_factory=list)
    #: Per-claim verdicts. Operational fields only; no model deliberation.
    claims: list[dict[str, Any]] = Field(default_factory=list)
    claim_count: int = 0
    supported_claim_count: int = 0
    unsupported_claim_count: int = 0
    uncited_claim_count: int = 0
    latency_ms: float = 0.0

    @property
    def verification_reason(self) -> str:
        """Specification name for `reason`, which Phase F already established."""
        return self.reason

    @field_validator("support_ratio", "confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("reason")
    @classmethod
    def _cap(cls, value: str) -> str:
        return cls._short(value)

    @field_validator("missing_evidence", mode="before")
    @classmethod
    def _clean_missing(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [" ".join(str(v).split())[:120] for v in value if str(v).strip()][:20]


def _last(_existing: Any, incoming: Any) -> Any:
    """Reducer: the writing node wins. Explicit beats LangGraph's default."""
    return incoming


class GraphState(TypedDict, total=False):
    """State threaded through every node.

    `total=False` because nodes return partial updates; `initial_state` seeds
    every key so no node has to guess whether one is present.
    """

    # Identity and timing
    request_id: str
    timestamps: dict[str, str]
    node_sequence: Annotated[list[str], lambda old, new: (old or []) + (new or [])]

    # Query
    original_query: str
    current_query: str
    rewritten_queries: list[str]
    sanitized: bool
    sanitization_notes: list[str]

    # Routing
    risk_level: RiskLevel
    risk_assessment: dict[str, Any]
    ambiguity: dict[str, Any]

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    reranked: bool
    reranker_used: bool

    # Grading
    evidence_grade: dict[str, Any]

    # Retry accounting
    retry_count: int
    max_retries: int
    regeneration_count: int
    max_regenerations: int

    # Request-wide execution budget (safe numeric observability only)
    request_timeout_s: float
    llm_call_limit: int
    llm_calls_used: int
    budget_exhausted: bool
    budget_exhaustion_reason: str
    budget_exhaustion_stage: str
    budget_elapsed_ms: float

    # Generation
    answer_draft: str
    citations: list[str]
    #: Exact per-sentence citation mapping built by grounded generation.
    claim_citations: list[dict[str, Any]]
    answer_confidence: float
    #: Phase E outcome string, used to distinguish a provider outage from an
    #: honest "the evidence does not cover this".
    generation_outcome: str

    # Verification (Phase G plugs in here)
    verification_result: dict[str, Any]

    # Termination
    final_outcome: FinalOutcome
    final_answer: str
    failure_reason: str
    abstain_reason: str


def initial_state(
    question: str,
    request_id: str,
    *,
    max_retries: int,
    max_regenerations: int,
    request_timeout_s: float,
    llm_call_limit: int,
    started_at: str,
) -> GraphState:
    """Seed every key so nodes never branch on a missing one."""
    return GraphState(
        request_id=request_id,
        timestamps={"started_at": started_at},
        node_sequence=[],
        original_query=question,
        current_query=question,
        rewritten_queries=[],
        sanitized=False,
        sanitization_notes=[],
        risk_level="none",
        risk_assessment={},
        ambiguity={},
        retrieved_chunks=[],
        reranked=False,
        reranker_used=False,
        evidence_grade={},
        retry_count=0,
        max_retries=max_retries,
        regeneration_count=0,
        max_regenerations=max_regenerations,
        request_timeout_s=request_timeout_s,
        llm_call_limit=llm_call_limit,
        llm_calls_used=0,
        budget_exhausted=False,
        budget_exhaustion_reason="",
        budget_exhaustion_stage="",
        budget_elapsed_ms=0.0,
        answer_draft="",
        citations=[],
        claim_citations=[],
        answer_confidence=0.0,
        generation_outcome="",
        verification_result={},
        final_outcome="error",
        final_answer="",
        failure_reason="",
        abstain_reason="",
    )
