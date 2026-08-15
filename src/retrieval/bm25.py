"""BM25 keyword retrieval.

Dense retrieval reliably misses exact identifiers such as `PAY-402`,
`AB-X200-EU`, or `RF-101`, because the embedding maps them into a fuzzy
semantic neighbourhood. BM25 is the half of the hybrid that recovers them, so
the tokeniser is deliberately identifier-aware: `pay-402` is indexed as the
whole token *and* as `pay` and `402`.

Scope note: `rank_bm25` is an in-memory index rebuilt from the database on
first use. That is correct for a corpus of this size (hundreds of chunks). At
tens of thousands of chunks, replace this with PostgreSQL full-text search
(`tsvector` + GIN) behind the same `search()` signature.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import replace

from rank_bm25 import BM25Okapi

from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*")

# Deliberately short. Aggressive stop-word removal hurts policy questions such
# as "how long do I have to return an item".
_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "to", "and", "or", "is", "are", "in", "on", "for", "it"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase, identifier-preserving tokeniser."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        tokens.append(match)
        if any(sep in match for sep in "-_/"):
            # Emit sub-tokens so `pay-402` also matches a bare `402` query.
            tokens.extend(part for part in re.split(r"[-_/]", match) if part)
    return tokens


class BM25Index:
    """In-memory BM25 index over the ingested corpus."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[RetrievedChunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def config(self) -> dict[str, object]:
        """Configuration block recorded in evaluation reports.

        Parameters are read off the live BM25Okapi instance rather than
        hard-coded, so the report always reflects what actually ran.
        """
        return {
            "library": "rank_bm25",
            "algorithm": "BM25Okapi",
            "k1": getattr(self._bm25, "k1", None),
            "b": getattr(self._bm25, "b", None),
            "epsilon": getattr(self._bm25, "epsilon", None),
            "corpus_size": self.size,
            "tokenizer": "lowercase; identifiers preserved; hyphenated sub-tokens expanded",
            "stopwords": sorted(_STOPWORDS),
            "zero_score_hits_dropped": True,
        }

    def build(self, chunks: list[RetrievedChunk] | None = None) -> None:
        if chunks is None:
            # Imported here so that `tokenize` and the index class remain usable
            # without the PostgreSQL driver installed.
            from src.retrieval.vector_store import fetch_all_chunks

            chunks = fetch_all_chunks()
        self._chunks = chunks
        if not self._chunks:
            self._bm25 = None
            logger.warning("BM25 index built over an empty corpus; run ingestion first")
            return
        corpus = [tokenize(chunk.content) for chunk in self._chunks]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d chunks", len(self._chunks))

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        # A zero score means no query term appeared; returning it only adds noise
        # to the fusion step.
        return [
            replace(self._chunks[idx], sparse_score=float(score))
            for idx, score in ranked
            if score > 0.0
        ]


_index: BM25Index | None = None
_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                index = BM25Index()
                index.build()
                _index = index
    return _index


def refresh_bm25_index() -> BM25Index:
    """Rebuild after ingestion. The API exposes this as POST /admin/reindex."""
    global _index
    with _lock:
        index = BM25Index()
        index.build()
        _index = index
    return _index
