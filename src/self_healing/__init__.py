"""Self-healing package.

The guard-rail primitives (confidence, rewriting, citation verification) are
pure enough to import without a database or a model, which is what lets the
fast test tier cover them. The orchestrating pipeline is exported lazily
because it wires in retrieval, reranking, and the LLM provider.
"""

from typing import TYPE_CHECKING, Any

from src.self_healing.citation_verifier import CitationReport, verify_citations
from src.self_healing.confidence import RetrievalConfidence, score_retrieval
from src.self_healing.query_rewriter import rewrite_query

if TYPE_CHECKING:  # pragma: no cover
    from src.self_healing.pipeline import RAGResponse, SelfHealingRAG, get_pipeline

__all__ = [
    "CitationReport",
    "RAGResponse",
    "RetrievalConfidence",
    "SelfHealingRAG",
    "get_pipeline",
    "rewrite_query",
    "score_retrieval",
    "verify_citations",
]

_LAZY = {"RAGResponse", "SelfHealingRAG", "get_pipeline"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from src.self_healing import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
