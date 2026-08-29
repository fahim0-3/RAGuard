"""HTTP contracts for the RAGuard API.

These models exist to keep the transport layer honest in both directions.

Inbound, they reject what should never reach the workflow: an empty question, a
novel-length one, or a client-supplied request ID containing anything that
would be unpleasant in a log line.

Outbound, they are an allow-list. The graph's final state carries more than a
customer should see — retrieved chunk objects, grader internals, node
bookkeeping — so the response is *built* from named fields rather than dumped
from state. A field that is not listed here cannot leak, which is a stronger
guarantee than remembering to strip things.

Nothing here exposes prompts, credentials, or model deliberation.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CitationOut",
    "ErrorResponse",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "ReadyResponse",
    "TimingStatsOut",
    "TraceStep",
]

#: Conservative: request IDs end up in logs and trace headers.
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

MAX_QUERY_CHARS = 2000
MIN_QUERY_CHARS = 3

Outcome = Literal["answer", "clarify", "abstain", "escalate", "error"]


class QueryRequest(BaseModel):
    """A customer question.

    `question` is accepted as an alias for `query` because the existing UI and
    the earlier API both used that name; breaking it would buy nothing.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str = Field(
        default="",
        validation_alias="query",
        description="The customer's question.",
    )
    question: str = Field(default="", description="Deprecated alias for `query`.")
    request_id: str | None = Field(
        default=None, description="Client-supplied correlation ID. Generated when absent."
    )
    session_id: str | None = Field(default=None, description="Optional conversation grouping.")

    @model_validator(mode="after")
    def _resolve_query(self) -> QueryRequest:
        text = (self.query or self.question or "").strip()
        if len(text) < MIN_QUERY_CHARS:
            raise ValueError(
                f"query must contain at least {MIN_QUERY_CHARS} non-whitespace characters"
            )
        if len(text) > MAX_QUERY_CHARS:
            raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
        self.query = text
        self.question = text
        return self

    @field_validator("request_id", "session_id")
    @classmethod
    def _safe_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not REQUEST_ID_PATTERN.match(value):
            raise ValueError(
                "id must be 1-64 characters of letters, digits, dot, colon, dash or underscore"
            )
        return value


class CitationOut(BaseModel):
    """One citation, carrying only metadata the server validated."""

    model_config = ConfigDict(extra="forbid")

    citation_label: str
    policy_id: str = ""
    source: str = ""
    chunk_index: int | None = None
    chunk_id: int | None = None
    excerpt: str = ""


class TraceStep(BaseModel):
    """One executed graph node. Operational state only, never reasoning."""

    model_config = ConfigDict(extra="forbid")

    step: int
    node: str
    duration_ms: float = 0.0


class TimingStatsOut(BaseModel):
    """Bounded numeric latency summary for one allow-listed operational stage."""

    model_config = ConfigDict(extra="forbid")

    count: int = 0
    total_ms: float = 0.0
    average_ms: float = 0.0
    max_ms: float = 0.0


class QueryResponse(BaseModel):
    """What a client receives. Every field is deliberate."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    outcome: Outcome
    answer: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = 0.0
    more_info_required: bool = True

    # Self-healing surface: the point of the project, so it is first-class.
    retry_count: int = 0
    max_retries: int = 0
    rewritten_queries: list[str] = Field(default_factory=list)
    risk_level: str = "none"

    # Verification summary. The per-claim detail stays server-side.
    verification_status: Literal["supported", "unsupported", "not_checked"] = "not_checked"
    verified_claim_count: int = 0
    unsupported_claim_count: int = 0

    evidence_sufficient: bool | None = None
    retrieved_chunk_count: int = 0
    reranker_used: bool | None = None

    failure_reason: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, TimingStatsOut] = Field(default_factory=dict)
    retrieval_latency_ms: dict[str, TimingStatsOut] = Field(default_factory=dict)
    unattributed_latency_ms: float = 0.0
    prompt_version: str = ""
    llm_calls_used: int = 0
    llm_call_limit: int = 0
    budget_exhausted: bool = False


class HealthResponse(BaseModel):
    """Liveness. Deliberately free of dependency checks."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str = "raguard"
    version: str = ""


class ReadyResponse(BaseModel):
    """Readiness, with a per-dependency breakdown."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    checks: dict[str, Any] = Field(default_factory=dict)
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Client-facing error. Never carries a stack trace or a connection string."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str
    request_id: str | None = None
