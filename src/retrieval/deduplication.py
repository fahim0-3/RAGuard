"""Candidate deduplication.

Fusion can surface the same evidence several times: the identical chunk from
two retrievers, two chunks whose overlap window covers the same sentences, or a
long run of consecutive slices of one document crowding out every other source.

The rules below are applied in rank order, so the highest-ranked instance of any
duplicate always survives.

1. **Identical chunk ID.** Defensive; fusion already merges by ID.
2. **Near-identical content.** Jaccard similarity over token sets at or above
   `near_duplicate_threshold`.
3. **Adjacent-chunk overlap.** Two chunks from the same source with consecutive
   `chunk_index` values share an ingestion overlap window by construction, so
   they are compared against a lower threshold.
4. **Contiguous run cap.** At most `max_adjacent_run` consecutive chunks from a
   single source may occupy the result list.

The bias throughout is toward keeping evidence. Rule 4 in particular is capable
of removing genuinely distinct material: golden case GC-006 needs the error-code
table *and* the descaling procedure from the same manual.

The cap default of 5 is measured, not guessed. At a cap of 3 it fired on every
document in this corpus, dropping 11 distinct sections across the five probe
queries, because each policy is only 3 to 4 chunks long with contiguous indices.
Aggregate metrics were unchanged either way, so the cap bought no quality and
cost real evidence. Five leaves the rule armed for genuinely excessive runs
without biting at the current document length. The evaluation report records
what each rule removed, so an over-aggressive setting is visible rather than
silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.retrieval.bm25 import tokenize
from src.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90
DEFAULT_ADJACENT_THRESHOLD = 0.70
DEFAULT_MAX_ADJACENT_RUN = 5


@dataclass(slots=True)
class DroppedChunk:
    chunk_id: int
    source: str
    chunk_index: int
    reason: str
    similar_to: int | None = None
    similarity: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "reason": self.reason,
            "similar_to": self.similar_to,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
        }


@dataclass(slots=True)
class DeduplicationResult:
    kept: list[RetrievedChunk] = field(default_factory=list)
    dropped: list[DroppedChunk] = field(default_factory=list)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def jaccard_similarity(left: str, right: str) -> float:
    """Token-set overlap. 1.0 means the two texts use exactly the same tokens."""
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def _violates_run_cap(
    candidate: RetrievedChunk,
    kept: list[RetrievedChunk],
    max_adjacent_run: int,
) -> bool:
    """True when accepting `candidate` would exceed the contiguous-run cap."""
    indices = sorted(c.chunk_index for c in kept if c.source == candidate.source)
    if not indices:
        return False

    merged = sorted({*indices, candidate.chunk_index})
    run = 1
    longest = 1
    for previous, current in zip(merged, merged[1:], strict=False):
        run = run + 1 if current == previous + 1 else 1
        longest = max(longest, run)
    return longest > max_adjacent_run


def deduplicate(
    chunks: list[RetrievedChunk],
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    adjacent_threshold: float = DEFAULT_ADJACENT_THRESHOLD,
    max_adjacent_run: int | None = DEFAULT_MAX_ADJACENT_RUN,
) -> DeduplicationResult:
    """Remove duplicate and near-duplicate candidates, preserving rank order."""
    result = DeduplicationResult()
    seen_ids: set[int] = set()

    for candidate in chunks:
        if candidate.chunk_id in seen_ids:
            result.dropped.append(
                DroppedChunk(
                    chunk_id=candidate.chunk_id,
                    source=candidate.source,
                    chunk_index=candidate.chunk_index,
                    reason="duplicate_chunk_id",
                    similar_to=candidate.chunk_id,
                    similarity=1.0,
                )
            )
            continue

        duplicate_of: RetrievedChunk | None = None
        similarity = 0.0
        reason = ""

        for kept in result.kept:
            score = jaccard_similarity(candidate.content, kept.content)
            is_adjacent = (
                kept.source == candidate.source
                and abs(kept.chunk_index - candidate.chunk_index) == 1
            )
            threshold = adjacent_threshold if is_adjacent else near_duplicate_threshold
            if score >= threshold:
                duplicate_of = kept
                similarity = score
                reason = "adjacent_overlap" if is_adjacent else "near_duplicate_content"
                break

        if duplicate_of is not None:
            result.dropped.append(
                DroppedChunk(
                    chunk_id=candidate.chunk_id,
                    source=candidate.source,
                    chunk_index=candidate.chunk_index,
                    reason=reason,
                    similar_to=duplicate_of.chunk_id,
                    similarity=similarity,
                )
            )
            continue

        if max_adjacent_run is not None and _violates_run_cap(
            candidate, result.kept, max_adjacent_run
        ):
            result.dropped.append(
                DroppedChunk(
                    chunk_id=candidate.chunk_id,
                    source=candidate.source,
                    chunk_index=candidate.chunk_index,
                    reason="max_adjacent_run_exceeded",
                )
            )
            continue

        seen_ids.add(candidate.chunk_id)
        result.kept.append(candidate)

    if result.dropped:
        logger.debug("Deduplication removed %d of %d candidates", result.dropped_count, len(chunks))
    return result


def deduplication_config(
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    adjacent_threshold: float = DEFAULT_ADJACENT_THRESHOLD,
    max_adjacent_run: int | None = DEFAULT_MAX_ADJACENT_RUN,
) -> dict[str, object]:
    """Configuration block recorded in evaluation reports."""
    return {
        "similarity": "jaccard over BM25 token sets",
        "near_duplicate_threshold": near_duplicate_threshold,
        "adjacent_threshold": adjacent_threshold,
        "max_adjacent_run": max_adjacent_run,
        "order": "rank order preserved; highest-ranked duplicate survives",
    }
