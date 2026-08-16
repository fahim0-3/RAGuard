"""Adapter from RAGuard evaluation records into the RAGAS dataset format.

RAGAS expects four fields per sample: the question, the retrieved contexts, the
generated answer, and a reference answer. RAGuard's golden dataset carries a
`ground_truth` written by hand, so the reference is real rather than invented.

Two transformations are documented because they change what a metric means:

1. **Abstention and non-answer cases are excluded by default.** Faithfulness
   over an empty answer is undefined, and scoring a correct refusal as a
   faithfulness failure would punish the behaviour the system exists to
   produce. They are counted and reported separately.

2. **Multi-policy cases keep every expected source.** The reference contexts
   list all of them, so context recall measures what it claims to.

Nothing here fabricates a reference. A case without a usable `ground_truth` is
dropped and reported, not filled in with the model's own answer — that would
make faithfulness a comparison of the model against itself.

This module imports no RAGAS symbol, so the adapter and its tests run whether
or not RAGAS is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RagasSample",
    "AdapterReport",
    "build_ragas_samples",
]

#: Outcomes for which faithfulness and relevancy are meaningful.
ANSWERED_OUTCOMES = frozenset({"answer"})


@dataclass(frozen=True)
class RagasSample:
    """One RAGAS row. Field names match the current RAGAS single-turn schema."""

    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_input": self.user_input,
            "retrieved_contexts": list(self.retrieved_contexts),
            "response": self.response,
            "reference": self.reference,
        }


@dataclass
class AdapterReport:
    """What the adapter kept, dropped, and why."""

    samples: list[RagasSample] = field(default_factory=list)
    excluded_non_answer: list[str] = field(default_factory=list)
    excluded_no_reference: list[str] = field(default_factory=list)
    excluded_no_context: list[str] = field(default_factory=list)

    @property
    def usable(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable_samples": self.usable,
            "excluded_non_answer": self.excluded_non_answer,
            "excluded_no_reference": self.excluded_no_reference,
            "excluded_no_context": self.excluded_no_context,
            "transformations": [
                "non-answer outcomes excluded: faithfulness over an empty answer is undefined",
                "cases without a hand-written ground_truth excluded rather than auto-filled",
                "multi-policy cases retain every expected source in retrieved_contexts",
            ],
        }


def build_ragas_samples(
    records: list[dict[str, Any]],
    include_non_answers: bool = False,
) -> AdapterReport:
    """Convert evaluation records into RAGAS samples.

    Each record needs: `case_id`, `question`, `contexts`, `answer`, `outcome`,
    and `ground_truth`. Anything missing a reference or a context is dropped and
    named in the report.
    """
    report = AdapterReport()

    for record in records:
        case_id = str(record.get("case_id") or record.get("id") or "")
        outcome = str(record.get("outcome") or "")

        if not include_non_answers and outcome not in ANSWERED_OUTCOMES:
            report.excluded_non_answer.append(case_id)
            continue

        contexts = [c for c in (record.get("contexts") or []) if str(c).strip()]
        if not contexts:
            report.excluded_no_context.append(case_id)
            continue

        reference = str(record.get("ground_truth") or "").strip()
        if not reference:
            # Never substitute the model's own answer: that turns faithfulness
            # into a self-comparison and always scores well.
            report.excluded_no_reference.append(case_id)
            continue

        report.samples.append(
            RagasSample(
                user_input=str(record.get("question") or ""),
                retrieved_contexts=[str(c) for c in contexts],
                response=str(record.get("answer") or ""),
                reference=reference,
                case_id=case_id,
            )
        )

    return report
