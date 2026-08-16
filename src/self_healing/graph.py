"""The self-healing workflow, as a real LangGraph `StateGraph`.

    START -> sanitize_and_classify -> risk_router
                                        |-- high risk --> escalate --> END
                                        `-- clear ------> ambiguity_detector
                                                            |-- ambiguous --> clarify --> END
                                                            `-- clear ------> hybrid_retrieve
                                                                                  |
                                                        rerank <-----------------'
                                                          |
                                                    evidence_grader
                                                      |-- sufficient ------> generate_answer
                                                      |-- retry available -> query_rewriter -.
                                                      `-- exhausted -------> abstain --> END  |
                                                                                              |
                                              hybrid_retrieve <-------------------------------'

    generate_answer -> verify_citations
                          |-- supported ------> finalize_answer --> END
                          |-- regenerate -----> generate_answer
                          `-- exhausted ------> abstain --> END

Why a graph and not a `while` loop. The retry bound, the routing rules, and the
guarantee that every path terminates are properties of the *topology*, and a
topology can be inspected. `test_graph_structure` reads the compiled graph and
asserts the nodes and edges exist; that assertion is impossible against control
flow buried in a function body. The cycle back into `hybrid_retrieve` is a real
edge, so the retry is part of the declared workflow rather than an implementation
detail of one function.

Routing is deterministic. Every conditional edge is a pure function of state,
with no model call inside it, so the same state always routes the same way.

Trust boundary: the customer question, the retrieved passages, and any rewritten
query are all data. Nothing read from them can move the graph to a different
node; only the deterministic routers decide that.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.retrieval.types import RetrievedChunk
from src.self_healing.abstention import abstention_message
from src.self_healing.ambiguity_detector import detect_ambiguity
from src.self_healing.evidence_grader import grade_evidence
from src.self_healing.query_rewriter import rewrite_once
from src.self_healing.retry_policy import (
    may_regenerate,
    may_retry,
    record_regeneration,
    record_retry,
    retry_snapshot,
)
from src.self_healing.risk_router import assess_risk, escalation_message
from src.self_healing.state import (
    AmbiguityDecision,
    EvidenceGrade,
    GraphState,
    RiskAssessment,
    VerificationResult,
    initial_state,
)
from src.self_healing.verification import Verifier, get_default_verifier

logger = logging.getLogger(__name__)

__all__ = [
    "NODE_NAMES",
    "SelfHealingGraph",
    "build_graph",
    "get_graph",
    "run_query",
]

# Node names are part of the public contract: traces reference them and the
# structure test asserts on them.
NODE_SANITIZE = "sanitize_and_classify"
NODE_RISK = "risk_router"
NODE_AMBIGUITY = "ambiguity_detector"
NODE_RETRIEVE = "hybrid_retrieve"
NODE_RERANK = "rerank"
NODE_GRADER = "evidence_grader"
NODE_REWRITER = "query_rewriter"
NODE_GENERATE = "generate_answer"
NODE_VERIFY = "verify_citations"
NODE_FINALIZE = "finalize_answer"
NODE_ABSTAIN = "abstain"
NODE_CLARIFY = "clarify"
NODE_ESCALATE = "escalate"

NODE_NAMES: tuple[str, ...] = (
    NODE_SANITIZE,
    NODE_RISK,
    NODE_AMBIGUITY,
    NODE_RETRIEVE,
    NODE_RERANK,
    NODE_GRADER,
    NODE_REWRITER,
    NODE_GENERATE,
    NODE_VERIFY,
    NODE_FINALIZE,
    NODE_ABSTAIN,
    NODE_CLARIFY,
    NODE_ESCALATE,
)

#: Control characters and markdown fencing that only ever appear in an attempt
#: to restructure the prompt. Stripping them is hygiene, not a security control:
#: the real defence is that instructions in data never reach a router.
_CONTROL_CHARS = dict.fromkeys(range(0, 9))
_CONTROL_CHARS.update(dict.fromkeys(range(11, 32)))

MAX_QUERY_CHARS = 2000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp(state: GraphState, key: str) -> dict[str, str]:
    stamps = dict(state.get("timestamps") or {})
    stamps[key] = _now()
    return stamps


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def sanitize_and_classify(state: GraphState) -> dict[str, Any]:
    """Normalise the question. Never interprets instructions inside it."""
    raw = state.get("original_query", "") or ""
    notes: list[str] = []

    cleaned = raw.translate(_CONTROL_CHARS)
    if cleaned != raw:
        notes.append("stripped control characters")

    if len(cleaned) > MAX_QUERY_CHARS:
        cleaned = cleaned[:MAX_QUERY_CHARS]
        notes.append(f"truncated to {MAX_QUERY_CHARS} characters")

    cleaned = " ".join(cleaned.split())

    return {
        "current_query": cleaned,
        "original_query": raw,
        "sanitized": True,
        "sanitization_notes": notes,
        "timestamps": _stamp(state, "sanitized_at"),
        "node_sequence": [NODE_SANITIZE],
    }


def risk_router(state: GraphState) -> dict[str, Any]:
    """Classify safety risk before any retrieval or generation happens."""
    assessment: RiskAssessment = assess_risk(state.get("current_query", ""))
    return {
        "risk_level": assessment.level,
        "risk_assessment": assessment.model_dump(),
        "timestamps": _stamp(state, "risk_assessed_at"),
        "node_sequence": [NODE_RISK],
    }


def ambiguity_detector(state: GraphState) -> dict[str, Any]:
    """Decide whether one clarifying question is needed before retrieving."""
    decision: AmbiguityDecision = detect_ambiguity(state.get("current_query", ""))
    return {
        "ambiguity": decision.model_dump(),
        "timestamps": _stamp(state, "ambiguity_checked_at"),
        "node_sequence": [NODE_AMBIGUITY],
    }


def hybrid_retrieve(state: GraphState) -> dict[str, Any]:
    """BM25 + dense + RRF over the current query, original or rewritten."""
    from src.retrieval.hybrid import get_hybrid_retriever

    query = state.get("current_query") or state.get("original_query", "")
    try:
        chunks = get_hybrid_retriever().retrieve(query)
    except Exception as exc:  # noqa: BLE001 - retrieval outage must not crash the graph
        logger.exception("Retrieval failed")
        return {
            "retrieved_chunks": [],
            "failure_reason": f"retrieval_failed: {type(exc).__name__}: {exc}",
            "timestamps": _stamp(state, "retrieved_at"),
            "node_sequence": [NODE_RETRIEVE],
        }

    return {
        "retrieved_chunks": chunks,
        "reranked": False,
        "timestamps": _stamp(state, "retrieved_at"),
        "node_sequence": [NODE_RETRIEVE],
    }


def rerank(state: GraphState) -> dict[str, Any]:
    """Cross-encoder reranking. Degrades to RRF order rather than failing."""
    from src.reranking import get_reranker

    chunks: list[RetrievedChunk] = list(state.get("retrieved_chunks") or [])
    if not chunks:
        return {"reranked": True, "reranker_used": False, "node_sequence": [NODE_RERANK]}

    query = state.get("current_query") or state.get("original_query", "")
    result = get_reranker().rerank_with_diagnostics(query, chunks)

    return {
        "retrieved_chunks": result.chunks,
        "reranked": True,
        "reranker_used": result.reranker_used,
        "timestamps": _stamp(state, "reranked_at"),
        "node_sequence": [NODE_RERANK],
    }


def evidence_grader(state: GraphState) -> dict[str, Any]:
    """Combine deterministic signals with structured grading."""
    grade: EvidenceGrade = grade_evidence(
        state.get("current_query", ""), list(state.get("retrieved_chunks") or [])
    )
    return {
        "evidence_grade": grade.model_dump(),
        "timestamps": _stamp(state, "graded_at"),
        "node_sequence": [NODE_GRADER],
    }


def query_rewriter(state: GraphState) -> dict[str, Any]:
    """One rewrite per retry, seeded by what the grader said was missing."""
    grade = EvidenceGrade.model_validate(state.get("evidence_grade") or {})
    original = state.get("original_query", "")

    rewritten = rewrite_once(
        original,
        missing_information=grade.missing_information,
        weak_chunks=list(state.get("retrieved_chunks") or []),
        use_llm=get_settings().graph_use_llm,
    )

    history = list(state.get("rewritten_queries") or [])
    if rewritten and rewritten not in history:
        history.append(rewritten)

    return {
        # The retry counter advances here, once per rewrite, so the bound holds
        # even when a rewrite produces nothing new.
        "retry_count": record_retry(state),
        "current_query": rewritten or original,
        "rewritten_queries": history,
        "timestamps": _stamp(state, "rewritten_at"),
        "node_sequence": [NODE_REWRITER],
    }


def generate_answer(state: GraphState) -> dict[str, Any]:
    """Grounded generation over the final chunks. Reuses the Phase E path."""
    from src.generation.answer_chain import generate_grounded_answer

    chunks = list(state.get("retrieved_chunks") or [])
    response = generate_grounded_answer(state.get("current_query", ""), chunks)

    return {
        "answer_draft": response.answer,
        "citations": response.citation_ids,
        "answer_confidence": response.confidence,
        "failure_reason": response.failure_reason or "",
        "generation_outcome": response.outcome,
        "timestamps": _stamp(state, "generated_at"),
        "node_sequence": [NODE_GENERATE],
    }


def make_verify_node(verifier: Verifier):
    """Bind a verifier into the node. Phase G swaps the object, not the graph."""

    def verify_citations_node(state: GraphState) -> dict[str, Any]:
        result: VerificationResult = verifier.verify(
            state.get("answer_draft", ""),
            list(state.get("citations") or []),
            list(state.get("retrieved_chunks") or []),
        )
        return {
            "verification_result": result.model_dump(),
            "timestamps": _stamp(state, "verified_at"),
            "node_sequence": [NODE_VERIFY],
        }

    return verify_citations_node


def finalize_answer(state: GraphState) -> dict[str, Any]:
    return {
        "final_outcome": "answer",
        "final_answer": state.get("answer_draft", ""),
        "timestamps": _stamp(state, "finished_at"),
        "node_sequence": [NODE_FINALIZE],
    }


def clarify(state: GraphState) -> dict[str, Any]:
    decision = AmbiguityDecision.model_validate(state.get("ambiguity") or {})
    return {
        "final_outcome": "clarify",
        "final_answer": decision.clarifying_question,
        "failure_reason": decision.reason,
        "timestamps": _stamp(state, "finished_at"),
        "node_sequence": [NODE_CLARIFY],
    }


def escalate(state: GraphState) -> dict[str, Any]:
    assessment = RiskAssessment.model_validate(state.get("risk_assessment") or {})
    return {
        "final_outcome": "escalate",
        "final_answer": escalation_message(assessment),
        "failure_reason": assessment.reason,
        "timestamps": _stamp(state, "finished_at"),
        "node_sequence": [NODE_ESCALATE],
    }


def abstain(state: GraphState) -> dict[str, Any]:
    """Terminal refusal. The reason is chosen from state, never invented."""
    grade = EvidenceGrade.model_validate(state.get("evidence_grade") or {})
    verification = VerificationResult.model_validate(state.get("verification_result") or {})
    existing = state.get("failure_reason", "") or ""

    if verification.checked and not verification.supported:
        reason = "unverified_citations"
    elif existing.startswith("retrieval_failed") or state.get("generation_outcome") in {
        "provider_error",
        "invalid_output",
    }:
        reason = "provider_error"
    elif state.get("answer_draft") == "" and state.get("generation_outcome"):
        reason = "generation_failed"
    elif not state.get("retrieved_chunks"):
        reason = "no_evidence"
    elif int(state.get("retry_count", 0)) >= int(state.get("max_retries", 0)) > 0:
        reason = "retries_exhausted"
    else:
        reason = "insufficient_evidence"

    return {
        "final_outcome": "abstain",
        "final_answer": abstention_message(reason, grade),  # type: ignore[arg-type]
        "failure_reason": existing or reason,
        "abstain_reason": reason,
        "timestamps": _stamp(state, "finished_at"),
        "node_sequence": [NODE_ABSTAIN],
    }


# --------------------------------------------------------------------------
# Routing (pure functions of state; no model calls)
# --------------------------------------------------------------------------


def route_after_risk(state: GraphState) -> str:
    return "escalate" if state.get("risk_level") == "high" else "continue"


def route_after_ambiguity(state: GraphState) -> str:
    ambiguity = state.get("ambiguity") or {}
    return "clarify" if ambiguity.get("ambiguous") else "retrieve"


def route_after_grading(state: GraphState) -> str:
    """Sufficient evidence answers; otherwise retry while the budget allows."""
    grade = state.get("evidence_grade") or {}
    if grade.get("sufficient"):
        return "generate"
    return "rewrite" if may_retry(state) else "abstain"


def route_after_generation(state: GraphState) -> str:
    outcome = state.get("generation_outcome")
    if outcome == "answered" and state.get("answer_draft"):
        return "verify"
    return "abstain"


def route_after_verification(state: GraphState) -> str:
    """Supported answers finalise; one regeneration is allowed, then abstain."""
    verification = state.get("verification_result") or {}
    if verification.get("supported"):
        return "finalize"
    return "regenerate" if may_regenerate(state) else "abstain"


def _count_regeneration(state: GraphState) -> dict[str, Any]:
    """Consume a regeneration attempt before re-entering generation."""
    return {"regeneration_count": record_regeneration(state)}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


def build_graph(verifier: Verifier | None = None) -> Any:
    """Build and compile the workflow.

    `verifier` is the Phase G seam: pass an LLM-backed verifier and every node,
    edge, and route below stays exactly as it is.
    """
    verifier = verifier or get_default_verifier()
    builder: StateGraph = StateGraph(GraphState)

    builder.add_node(NODE_SANITIZE, sanitize_and_classify)
    builder.add_node(NODE_RISK, risk_router)
    builder.add_node(NODE_AMBIGUITY, ambiguity_detector)
    builder.add_node(NODE_RETRIEVE, hybrid_retrieve)
    builder.add_node(NODE_RERANK, rerank)
    builder.add_node(NODE_GRADER, evidence_grader)
    builder.add_node(NODE_REWRITER, query_rewriter)
    builder.add_node(NODE_GENERATE, generate_answer)
    builder.add_node(NODE_VERIFY, make_verify_node(verifier))
    builder.add_node(NODE_FINALIZE, finalize_answer)
    builder.add_node(NODE_ABSTAIN, abstain)
    builder.add_node(NODE_CLARIFY, clarify)
    builder.add_node(NODE_ESCALATE, escalate)
    builder.add_node("_count_regeneration", _count_regeneration)

    builder.add_edge(START, NODE_SANITIZE)
    builder.add_edge(NODE_SANITIZE, NODE_RISK)

    builder.add_conditional_edges(
        NODE_RISK,
        route_after_risk,
        {"escalate": NODE_ESCALATE, "continue": NODE_AMBIGUITY},
    )
    builder.add_conditional_edges(
        NODE_AMBIGUITY,
        route_after_ambiguity,
        {"clarify": NODE_CLARIFY, "retrieve": NODE_RETRIEVE},
    )

    builder.add_edge(NODE_RETRIEVE, NODE_RERANK)
    builder.add_edge(NODE_RERANK, NODE_GRADER)

    builder.add_conditional_edges(
        NODE_GRADER,
        route_after_grading,
        {
            "generate": NODE_GENERATE,
            "rewrite": NODE_REWRITER,
            "abstain": NODE_ABSTAIN,
        },
    )
    # The retry cycle, declared as an edge rather than hidden in a loop body.
    builder.add_edge(NODE_REWRITER, NODE_RETRIEVE)

    builder.add_conditional_edges(
        NODE_GENERATE,
        route_after_generation,
        {"verify": NODE_VERIFY, "abstain": NODE_ABSTAIN},
    )
    builder.add_conditional_edges(
        NODE_VERIFY,
        route_after_verification,
        {
            "finalize": NODE_FINALIZE,
            "regenerate": "_count_regeneration",
            "abstain": NODE_ABSTAIN,
        },
    )
    builder.add_edge("_count_regeneration", NODE_GENERATE)

    for terminal in (NODE_FINALIZE, NODE_ABSTAIN, NODE_CLARIFY, NODE_ESCALATE):
        builder.add_edge(terminal, END)

    return builder.compile(name="raguard_self_healing")


_graph: Any | None = None


def get_graph() -> Any:
    """Process-wide compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    global _graph
    _graph = None


class SelfHealingGraph:
    """Callable service wrapper. Phase H exposes this; it does not import the API."""

    def __init__(self, verifier: Verifier | None = None) -> None:
        self.graph = build_graph(verifier) if verifier is not None else get_graph()

    def invoke(self, question: str, request_id: str | None = None) -> dict[str, Any]:
        settings = get_settings()
        state = initial_state(
            question,
            request_id or str(uuid.uuid4()),
            max_retries=settings.graph_max_retries,
            max_regenerations=settings.graph_max_regenerations,
            started_at=_now(),
        )
        final = self.graph.invoke(state)
        return dict(final)

    def run(self, question: str, request_id: str | None = None) -> dict[str, Any]:
        """Invoke and return an observability-shaped summary."""
        final = self.invoke(question, request_id)
        return summarise(final)


def summarise(state: dict[str, Any]) -> dict[str, Any]:
    """Trace-shaped view. Contains no prompt text and no credentials."""
    return {
        "request_id": state.get("request_id"),
        "original_query": state.get("original_query"),
        "current_query": state.get("current_query"),
        "rewritten_queries": list(state.get("rewritten_queries") or []),
        "final_outcome": state.get("final_outcome"),
        "final_answer": state.get("final_answer"),
        "failure_reason": state.get("failure_reason"),
        "abstain_reason": state.get("abstain_reason"),
        "risk_level": state.get("risk_level"),
        "citations": list(state.get("citations") or []),
        "answer_confidence": state.get("answer_confidence"),
        "evidence_grade": state.get("evidence_grade") or {},
        "verification_result": state.get("verification_result") or {},
        "retrieved_chunk_count": len(state.get("retrieved_chunks") or []),
        "reranker_used": state.get("reranker_used"),
        "node_sequence": list(state.get("node_sequence") or []),
        "timestamps": state.get("timestamps") or {},
        **retry_snapshot(state),
    }


def run_query(question: str, request_id: str | None = None) -> dict[str, Any]:
    """Module-level entry point for callers that do not need the object."""
    return SelfHealingGraph().run(question, request_id)
