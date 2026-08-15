"""Retrieval confidence scoring: the trigger for self-healing.

The signal is the cross-encoder score of the top passage, squashed to [0, 1].
Two supporting signals guard against its failure modes:

- `margin` (top1 minus top2) detects the case where the reranker cannot
  separate candidates, which usually means the question is under-specified.
- `support` (how many of the top passages clear the threshold) detects the
  case where a single lucky passage sits above a field of noise.

Thresholds are configuration, not constants, because the correct cut-off is
corpus-specific and must be re-tuned when the corpus or reranker changes. The
golden-dataset evaluation is what justifies a given value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import get_settings
from src.retrieval.types import RetrievedChunk

ConfidenceLevel = Literal["high", "weak", "insufficient"]


@dataclass(slots=True)
class RetrievalConfidence:
    level: ConfidenceLevel
    top_score: float
    second_score: float
    margin: float
    mean_score: float
    supporting_chunks: int

    @property
    def should_heal(self) -> bool:
        """Weak retrieval is worth a rewrite-and-retry; insufficient is not."""
        return self.level == "weak"

    @property
    def should_abstain(self) -> bool:
        return self.level == "insufficient"

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "level": self.level,
            "top_score": round(self.top_score, 4),
            "second_score": round(self.second_score, 4),
            "margin": round(self.margin, 4),
            "mean_score": round(self.mean_score, 4),
            "supporting_chunks": self.supporting_chunks,
        }


def score_retrieval(chunks: list[RetrievedChunk]) -> RetrievalConfidence:
    """Classify a reranked candidate list.

    Requires `normalised_rerank_score` to be populated, that is, the chunks must
    have passed through the reranker.
    """
    settings = get_settings()

    if not chunks:
        return RetrievalConfidence(
            level="insufficient",
            top_score=0.0,
            second_score=0.0,
            margin=0.0,
            mean_score=0.0,
            supporting_chunks=0,
        )

    scores = [c.normalised_rerank_score or 0.0 for c in chunks]
    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    mean_score = sum(scores) / len(scores)
    supporting = sum(1 for s in scores if s >= settings.retrieval_confidence_threshold)

    if top_score < settings.abstain_threshold:
        level: ConfidenceLevel = "insufficient"
    elif top_score < settings.retrieval_confidence_threshold:
        level = "weak"
    elif supporting == 1 and (top_score - second_score) < 0.15:
        # One passage barely clears the bar and nothing corroborates it.
        level = "weak"
    else:
        level = "high"

    return RetrievalConfidence(
        level=level,
        top_score=top_score,
        second_score=second_score,
        margin=top_score - second_score,
        mean_score=mean_score,
        supporting_chunks=supporting,
    )
