"""Retrieval package.

`RetrievedChunk` is exported eagerly because it is a plain dataclass used
everywhere, including in tests and evaluation. The retriever itself is exported
lazily: importing it pulls in the PostgreSQL driver, and a unit test that only
needs the data structure should not require a database driver to be installed.
"""

from typing import TYPE_CHECKING, Any

from src.retrieval.types import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from src.retrieval.hybrid import HybridRetriever, get_hybrid_retriever

__all__ = ["HybridRetriever", "RetrievedChunk", "get_hybrid_retriever"]

_LAZY = {"HybridRetriever", "get_hybrid_retriever"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from src.retrieval import hybrid

        return getattr(hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
