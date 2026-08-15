"""Cross-encoder reranking with BGE-Reranker-v2-m3.

The reranker does two jobs in RAGuard:

1. It reorders fusion candidates by true query-document relevance.
2. Its top score is the *retrieval confidence signal* that drives self-healing.

Raw cross-encoder outputs are unbounded logits, so they are squashed through a
sigmoid into `normalised_rerank_score`. Only the normalised value is compared
against thresholds; the raw logit is retained for debugging.

Cost warning: this model is ~568 M parameters. On CPU expect roughly 0.3 to 1.0
seconds per 20 candidates. Keep `FUSION_TOP_K` modest unless a GPU is available.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import replace
from typing import TYPE_CHECKING

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    # Guard against overflow on large negative logits.
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.model_device
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("Loading reranker %s on %s", self.model_name, self.device)
                    self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k

        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores = self._get_model().predict(pairs)

        scored = [
            replace(
                chunk,
                rerank_score=float(score),
                normalised_rerank_score=sigmoid(float(score)),
            )
            for chunk, score in zip(chunks, raw_scores, strict=True)
        ]
        scored.sort(key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        return scored[:top_k]


_reranker: CrossEncoderReranker | None = None
_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker
