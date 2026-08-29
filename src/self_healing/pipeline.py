"""The self-healing RAG pipeline.

Control flow:

    retrieve -> rerank -> score confidence
        |
        +-- high         -> generate
        +-- weak         -> rewrite query, retrieve again, fuse, rerank  (up to N times)
        +-- insufficient -> abstain
                              |
    generate -> model declares insufficient context -> abstain
             -> verify citations -> invalid -> abstain
                                 -> valid   -> answer

Abstention is a first-class outcome, not an error. The design position is that
a support assistant which says "I do not know" is strictly better than one that
invents a refund window, so every failure path converges on abstention rather
than on a degraded answer.

Every decision is recorded in `trace`, which is what makes the behaviour
measurable in evaluation and explainable in the demonstration UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings
from src.generation.answer_chain import AnswerDraft, generate_answer
from src.generation.prompts import ABSTENTION_MESSAGE, PROMPT_VERSION
from src.reranking import get_reranker
from src.retrieval import get_hybrid_retriever
from src.retrieval.types import RetrievedChunk
from src.self_healing.citation_verifier import CitationReport, verify_citations
from src.self_healing.confidence import RetrievalConfidence, score_retrieval
from src.self_healing.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGResponse:
    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: dict[str, Any] = field(default_factory=dict)
    healing_attempts: int = 0
    rewritten_queries: list[str] = field(default_factory=list)
    citation_report: dict[str, Any] = field(default_factory=dict)
    contexts: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": self.citations,
            "confidence": self.confidence,
            "healing_attempts": self.healing_attempts,
            "rewritten_queries": self.rewritten_queries,
            "citation_report": self.citation_report,
            "trace": self.trace,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_version": self.prompt_version,
        }


class SelfHealingRAG:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = get_hybrid_retriever()
        self.reranker = get_reranker()

    # -- stages ------------------------------------------------------------

    def _retrieve_and_rerank(
        self, query: str, extra_queries: list[str] | None = None
    ) -> list[RetrievedChunk]:
        queries = [query, *(extra_queries or [])]
        candidates = (
            self.retriever.retrieve(query)
            if len(queries) == 1
            else self.retriever.retrieve_many(queries)
        )
        return self.reranker.rerank(query, candidates)

    def retrieve_only(self, question: str) -> tuple[list[RetrievedChunk], RetrievalConfidence]:
        """Retrieval path without any LLM call.

        Used by the deterministic CI gate, which must run with no API key and
        produce identical results on every execution.
        """
        chunks = self._retrieve_and_rerank(question)
        return chunks, score_retrieval(chunks)

    # -- orchestration -----------------------------------------------------

    def answer(
        self,
        question: str,
        use_llm_rewrite: bool = True,
        use_llm_verification: bool = False,
    ) -> RAGResponse:
        started = time.perf_counter()
        trace: list[dict[str, Any]] = []
        rewritten: list[str] = []
        attempts = 0

        chunks = self._retrieve_and_rerank(question)
        confidence = score_retrieval(chunks)
        trace.append(
            {
                "stage": "initial_retrieval",
                "query": question,
                "chunks": len(chunks),
                "confidence": confidence.to_dict(),
            }
        )

        # --- healing loop -------------------------------------------------
        while confidence.level != "high" and attempts < self.settings.max_healing_attempts:
            attempts += 1
            variants = rewrite_query(question, weak_chunks=chunks, use_llm=use_llm_rewrite)
            if not variants:
                trace.append({"stage": "rewrite", "attempt": attempts, "variants": []})
                break

            rewritten.extend(v for v in variants if v not in rewritten)
            retried = self._retrieve_and_rerank(question, extra_queries=variants)
            retried_confidence = score_retrieval(retried)

            improved = retried_confidence.top_score > confidence.top_score
            trace.append(
                {
                    "stage": "healing_retry",
                    "attempt": attempts,
                    "variants": variants,
                    "confidence": retried_confidence.to_dict(),
                    "accepted": improved,
                }
            )
            if improved:
                chunks, confidence = retried, retried_confidence
            else:
                # Retrying again with the same signal will not help.
                break

        if confidence.should_abstain:
            return self._abstain(
                question,
                "low_retrieval_confidence",
                confidence,
                chunks,
                attempts,
                rewritten,
                trace,
                started,
            )

        # --- generation ---------------------------------------------------
        draft: AnswerDraft = generate_answer(question, chunks)
        trace.append(
            {
                "stage": "generation",
                "sufficient_context": draft.sufficient_context,
                "citations": draft.citations,
                "parse_failed": draft.parse_failed,
            }
        )

        if not draft.sufficient_context or not draft.answer:
            return self._abstain(
                question,
                "model_declared_insufficient_context",
                confidence,
                chunks,
                attempts,
                rewritten,
                trace,
                started,
            )

        # --- citation verification -----------------------------------------
        report: CitationReport = verify_citations(
            draft.answer, draft.citations, chunks, use_llm=use_llm_verification
        )
        trace.append({"stage": "citation_verification", **report.to_dict()})

        if not report.valid:
            return self._abstain(
                question,
                "citation_verification_failed",
                confidence,
                chunks,
                attempts,
                rewritten,
                trace,
                started,
                citation_report=report.to_dict(),
                rejected_answer=draft.answer,
            )

        cited_labels = set(draft.citations)
        return RAGResponse(
            question=question,
            answer=draft.answer,
            abstained=False,
            citations=[c.to_dict() for c in chunks if c.citation_label in cited_labels],
            confidence=confidence.to_dict(),
            healing_attempts=attempts,
            rewritten_queries=rewritten,
            citation_report=report.to_dict(),
            contexts=[c.content for c in chunks],
            trace=trace,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- helpers -----------------------------------------------------------

    def _abstain(
        self,
        question: str,
        reason: str,
        confidence: RetrievalConfidence,
        chunks: list[RetrievedChunk],
        attempts: int,
        rewritten: list[str],
        trace: list[dict[str, Any]],
        started: float,
        citation_report: dict[str, Any] | None = None,
        rejected_answer: str | None = None,
    ) -> RAGResponse:
        logger.info("Abstaining on %r (reason=%s)", question, reason)
        trace.append(
            {
                "stage": "abstention",
                "reason": reason,
                "rejected_answer": rejected_answer,
            }
        )
        return RAGResponse(
            question=question,
            answer=ABSTENTION_MESSAGE,
            abstained=True,
            abstain_reason=reason,
            citations=[],
            confidence=confidence.to_dict(),
            healing_attempts=attempts,
            rewritten_queries=rewritten,
            citation_report=citation_report or {},
            contexts=[c.content for c in chunks],
            trace=trace,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


_pipeline: SelfHealingRAG | None = None


def get_pipeline() -> SelfHealingRAG:
    global _pipeline
    if _pipeline is None:
        _pipeline = SelfHealingRAG()
    return _pipeline
