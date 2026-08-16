"""Phase D: deterministic metric primitives.

Every function here is pure. Identical input produces an identical number on
every run, on every machine, with no model, no database, and no network. That
is the only property that makes a merge gate trustworthy: a gate that can go
red on unchanged code teaches the team to ignore it.

The retrieval-ranking primitives (HitRate@k, Recall@k, MRR@k) already live in
`retrieval_eval` and are imported rather than redefined. Two subtly different
definitions of MRR in one codebase is worse than none.

This module adds the metrics the ranking primitives cannot express:

- **citation ID validity** — do the cited document identifiers actually exist?
- **structured output validity** — is the answer envelope well formed?
- **expected-outcome accuracy** — did the system answer, clarify, abstain, or
  escalate when the golden case said it should?
- **latency** — wall-clock cost, reported as a distribution rather than a mean,
  because a mean hides the tail that users actually feel.
- **regression count** — how many measured metrics fell below their target.

Measured values and target thresholds are kept strictly apart. Nothing in this
module writes a threshold, and nothing reads a measurement into one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EXPECTED_OUTCOMES",
    "LatencyStats",
    "MetricRegression",
    "REQUIRED_ANSWER_FIELDS",
    "citation_id_validity",
    "expected_outcome_accuracy",
    "find_regressions",
    "invalid_citation_ids",
    "latency_stats",
    "percentile",
    "structured_output_errors",
    "structured_output_validity",
]

EXPECTED_OUTCOMES = ("answer", "clarify", "abstain", "escalate")

REQUIRED_ANSWER_FIELDS: dict[str, type | tuple[type, ...]] = {
    "answer": str,
    "citations": list,
    "abstained": bool,
    "confidence": (int, float),
    "outcome": str,
}


# --------------------------------------------------------------------------
# Citation validity
# --------------------------------------------------------------------------


def invalid_citation_ids(cited: Iterable[str], known: Iterable[str]) -> list[str]:
    """Cited identifiers that do not exist in the corpus, in order, with duplicates."""
    known_set = set(known)
    return [c for c in cited if c not in known_set]


def citation_id_validity(cited: Sequence[str], known: Iterable[str]) -> float | None:
    """Fraction of cited identifiers that exist in the corpus.

    Returns None when nothing was cited. That is *not applicable*, not zero: a
    correct abstention cites nothing and must not be scored as if it had
    invented a citation. Callers aggregate over the non-None values.
    """
    if not cited:
        return None
    return 1.0 - len(invalid_citation_ids(cited, known)) / len(cited)


# --------------------------------------------------------------------------
# Structured output validity
# --------------------------------------------------------------------------


def structured_output_errors(
    payload: Any, required: dict[str, type | tuple[type, ...]] | None = None
) -> list[str]:
    """Every way the answer envelope is malformed, as human-readable strings.

    Returning the reasons rather than a bare boolean matters in practice: a
    validity metric that drops to 0.94 is useless without knowing which field
    broke.
    """
    required = required or REQUIRED_ANSWER_FIELDS
    if not isinstance(payload, dict):
        return [f"payload must be an object, got {type(payload).__name__}"]

    errors: list[str] = []
    for name, expected_type in required.items():
        if name not in payload:
            errors.append(f"missing field: {name}")
            continue
        value = payload[name]
        # bool subclasses int, so isinstance(True, (int, float)) is True. A bool
        # where a number belongs is a defect, so reject it before the type check.
        if expected_type is not bool and isinstance(value, bool):
            errors.append(f"{name} must be {_type_name(expected_type)}, got bool")
            continue
        if not isinstance(value, expected_type):
            errors.append(f"{name} must be {_type_name(expected_type)}, got {type(value).__name__}")

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append(f"confidence must lie in [0, 1], got {confidence}")

    outcome = payload.get("outcome")
    if isinstance(outcome, str) and outcome not in EXPECTED_OUTCOMES:
        errors.append(f"outcome must be one of {list(EXPECTED_OUTCOMES)}, got {outcome!r}")

    citations = payload.get("citations")
    if isinstance(citations, list) and any(not isinstance(c, str) for c in citations):
        errors.append("citations must all be strings")

    abstained, answer = payload.get("abstained"), payload.get("answer")
    if abstained is True and isinstance(citations, list) and citations:
        errors.append("an abstention must not carry citations")
    if abstained is False and isinstance(answer, str) and not answer.strip():
        errors.append("a non-abstaining answer must not be empty")

    return errors


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def structured_output_validity(
    payloads: Sequence[Any], required: dict[str, type | tuple[type, ...]] | None = None
) -> float:
    """Fraction of payloads that are completely well formed."""
    if not payloads:
        return 0.0
    valid = sum(1 for p in payloads if not structured_output_errors(p, required))
    return valid / len(payloads)


# --------------------------------------------------------------------------
# Expected-outcome accuracy
# --------------------------------------------------------------------------


def expected_outcome_accuracy(
    predicted: Sequence[str], expected: Sequence[str]
) -> float:
    """Fraction of cases whose outcome matched the golden expectation."""
    if len(predicted) != len(expected):
        raise ValueError(
            f"predicted and expected must align: {len(predicted)} vs {len(expected)}"
        )
    if not expected:
        return 0.0
    return sum(1 for p, e in zip(predicted, expected, strict=True) if p == e) / len(expected)


def outcome_confusion(
    predicted: Sequence[str], expected: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Counts of expected outcome against predicted outcome.

    An aggregate accuracy hides the only distinction that matters here: a system
    that answers when it should have abstained is dangerous, while one that
    abstains when it could have answered is merely unhelpful.
    """
    matrix = {e: dict.fromkeys(EXPECTED_OUTCOMES, 0) for e in EXPECTED_OUTCOMES}
    for p, e in zip(predicted, expected, strict=True):
        if e in matrix and p in matrix[e]:
            matrix[e][p] += 1
    return matrix


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Deterministic, no interpolation, no numpy."""
    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass(slots=True)
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


def latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    if not samples_ms:
        return LatencyStats(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)
    return LatencyStats(
        count=len(samples_ms),
        mean_ms=sum(samples_ms) / len(samples_ms),
        p50_ms=percentile(samples_ms, 0.50),
        p95_ms=percentile(samples_ms, 0.95),
        max_ms=max(samples_ms),
    )


# --------------------------------------------------------------------------
# Regression counting
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MetricRegression:
    metric: str
    measured: float
    target: float
    tolerance: float
    shortfall: float = field(init=False)

    def __post_init__(self) -> None:
        self.shortfall = self.target - self.measured

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "measured": round(self.measured, 4),
            "target": round(self.target, 4),
            "tolerance": self.tolerance,
            "shortfall": round(self.shortfall, 4),
        }

    def __str__(self) -> str:
        return (
            f"{self.metric}: measured {self.measured:.4f} < "
            f"target {self.target:.4f} (tolerance {self.tolerance})"
        )


def find_regressions(
    measured: dict[str, float], targets: dict[str, float], tolerance: float = 0.0
) -> list[MetricRegression]:
    """Measured metrics that fell below their target, beyond the tolerance.

    Targets absent from `measured` are ignored rather than treated as zero: a
    metric that was not measured has not regressed, it is simply unknown, and
    conflating the two produces a gate that fails for the wrong reason.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    regressions = []
    for metric, target in targets.items():
        if metric not in measured:
            continue
        value = float(measured[metric])
        if value < float(target) - tolerance:
            regressions.append(
                MetricRegression(
                    metric=metric, measured=value, target=float(target), tolerance=tolerance
                )
            )
    return sorted(regressions, key=lambda r: -r.shortfall)
