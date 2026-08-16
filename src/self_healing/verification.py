"""Citation verification boundary (Phase F), designed for Phase G to replace.

Phase F does not build a new verifier. One already exists in
`citation_verifier`: it resolves citation labels, splits the answer into
claims, and scores lexical support against the cited passage. Duplicating that
logic here would give the system two disagreeing definitions of "supported".

What Phase F adds is the *seam*. The graph depends on a `Verifier` protocol and
writes its output into `state["verification_result"]` as a `VerificationResult`.
Phase G supplies an LLM-backed entailment verifier satisfying the same
protocol, and no node, edge, or routing function changes: only the object
passed to `build_graph(verifier=...)`.

The `verifier` field on the result records which implementation produced it, so
a trace never leaves you guessing whether a run was checked deterministically
or by a model.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.config import get_settings
from src.retrieval.types import RetrievedChunk
from src.self_healing.citation_verifier import verify_citations
from src.self_healing.state import VerificationResult

logger = logging.getLogger(__name__)

__all__ = [
    "DeterministicVerifier",
    "Verifier",
    "get_default_verifier",
]


@runtime_checkable
class Verifier(Protocol):
    """The contract Phase G must satisfy. Deliberately minimal."""

    name: str

    def verify(
        self, answer: str, citations: list[str], chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        """Decide whether `answer` is supported by the cited `chunks`."""
        ...


class DeterministicVerifier:
    """Phase F default: the existing lexical verifier, adapted to the contract.

    Reproducible and free, which is what lets the graph's verification route be
    exercised in the fast test tier. It under-detects paraphrase, and that is
    the specific weakness Phase G exists to fix.
    """

    name = "deterministic"

    def __init__(self, use_llm: bool = False) -> None:
        # `use_llm` is passed through to the existing verifier's optional
        # entailment step; Phase F leaves it off so the tier stays offline.
        self.use_llm = use_llm

    def verify(
        self, answer: str, citations: list[str], chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        if not answer.strip():
            return VerificationResult(
                supported=False,
                checked=False,
                verifier=self.name,
                reason="no answer text to verify",
            )

        report = verify_citations(answer, citations, chunks, use_llm=self.use_llm)

        reason = "all claims supported by cited passages"
        if report.invalid_labels:
            reason = f"citations not in the supplied passages: {report.invalid_labels}"
        elif report.unsupported_claims:
            reason = f"{len(report.unsupported_claims)} claim(s) not supported by the citation"

        return VerificationResult(
            supported=bool(report.valid),
            checked=True,
            verifier=self.name,
            unsupported_claims=list(report.unsupported_claims),
            invalid_citations=list(report.invalid_labels),
            support_ratio=report.support_ratio,
            reason=reason,
        )


def get_default_verifier() -> Verifier:
    """The verifier the graph uses unless one is injected."""
    settings = get_settings()
    return DeterministicVerifier(use_llm=bool(getattr(settings, "graph_verify_with_llm", False)))
