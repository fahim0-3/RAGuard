"""Calibration-only confidence evaluation for Voyage reranking.

Nothing in this module is imported by the serving path. It keeps Voyage
relevance values out of BGE's ``rerank_score`` and ``normalised_rerank_score``
fields until a held-out safety evaluation explicitly approves a profile.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.evaluation.reranking_eval import aggregate, score_case
from src.retrieval.types import RetrievedChunk

ConfidenceLevel = Literal["high", "insufficient"]
SplitName = Literal["calibration", "development", "held_out"]


@dataclass(frozen=True, slots=True)
class VoyageConfidenceProfile:
    """Versioned, provider-specific thresholds for *evaluation only*.

    ``permit_score_min`` is the only binary eligibility gate. Support and
    margin settings are retained for diagnostics and are not eligibility
    predicates. Raw Voyage values never share BGE's score fields or settings.
    """

    profile_id: str
    status: Literal["draft", "approved", "retired"]
    provider: str
    model: str
    dataset_version: str
    corpus_fingerprint: str
    chunking_fingerprint: str
    candidate_top_k: int
    final_top_k: int
    permit_score_min: float
    support_score_min: float
    min_supporting_chunks: int
    ambiguous_margin_max: float
    minimum_answer_recall: float
    maximum_abstention_false_permit_rate: float
    calibration_split: str
    development_split: str
    held_out_split: str
    notes: str = ""

    def compatible_with(self, metadata: dict[str, Any]) -> bool:
        """Require an exact environment match before a future serving use."""
        return all(
            metadata.get(name) == getattr(self, name)
            for name in (
                "provider",
                "model",
                "dataset_version",
                "corpus_fingerprint",
                "chunking_fingerprint",
                "candidate_top_k",
                "final_top_k",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VoyageConfidence:
    level: ConfidenceLevel
    top_score: float
    second_score: float
    margin: float
    mean_score: float
    supporting_chunks: int
    has_scores: bool
    top_score_passes: bool
    support_count_diagnostic_passes: bool
    margin_diagnostic_passes: bool
    deterministic_eligible: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confidence_stratum(case: dict[str, Any]) -> str:
    """Return a deterministic outcome/category stratum for dataset splitting."""
    outcome = str(case["expected_outcome"])
    case_type = str(case.get("case_type") or "unspecified")
    category = str(case.get("category") or case_type)
    if outcome == "answer":
        return f"answer:{case_type}"
    if case_type == "prompt_injection" or category == "security":
        category = "injection_adversarial"
    elif category == "conflicting_evidence":
        category = "conflict"
    return f"abstain:{category}"


def stratified_split_assignments(cases: list[dict[str, Any]]) -> dict[str, SplitName]:
    """Create deterministic 60/20/20-like strata with a locked held-out set.

    Small strata retain at least one calibration example. Development and
    held-out examples are added where the stratum size permits them.
    """
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        strata[confidence_stratum(case)].append(case)

    assignments: dict[str, SplitName] = {}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda case: hashlib.sha256(
                f"voyage-confidence-stratified-v2:{stratum}:{case['id']}".encode()
            ).hexdigest(),
        )
        count = len(ordered)
        held_out_count = 1 if count >= 2 else 0
        development_count = 1 if count >= 3 else 0
        if count >= 8:
            held_out_count = max(1, round(count * 0.2))
            development_count = max(1, round(count * 0.2))
        calibration_count = count - development_count - held_out_count
        for index, case in enumerate(ordered):
            if index < calibration_count:
                split: SplitName = "calibration"
            elif index < calibration_count + development_count:
                split = "development"
            else:
                split = "held_out"
            assignments[str(case["id"])] = split
    return assignments


def split_composition(
    cases: list[dict[str, Any]], assignments: dict[str, SplitName]
) -> dict[str, Any]:
    """Summarize split balance without exposing held-out values to selection."""
    result: dict[str, Any] = {}
    for split in ("calibration", "development", "held_out"):
        selected = [case for case in cases if assignments[case["id"]] == split]
        result[split] = {
            "total": len(selected),
            "answer": sum(case["expected_outcome"] == "answer" for case in selected),
            "abstain": sum(case["expected_outcome"] == "abstain" for case in selected),
            "strata": dict(
                sorted(
                    {
                        stratum: sum(confidence_stratum(case) == stratum for case in selected)
                        for stratum in {confidence_stratum(case) for case in selected}
                    }.items()
                )
            ),
        }
    return result


def voyage_confidence(scores: list[float], profile: VoyageConfidenceProfile) -> VoyageConfidence:
    """Apply the explicit top-score gate and calculate diagnostic-only signals."""
    ordered = [float(score) for score in scores]
    if not ordered:
        return VoyageConfidence(
            level="insufficient",
            top_score=0.0,
            second_score=0.0,
            margin=0.0,
            mean_score=0.0,
            supporting_chunks=0,
            has_scores=False,
            top_score_passes=False,
            support_count_diagnostic_passes=False,
            margin_diagnostic_passes=False,
            deterministic_eligible=False,
            rejection_reason="no_voyage_scores",
        )

    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    margin = top - second
    supporting = sum(score >= profile.support_score_min for score in ordered)
    top_passes = top >= profile.permit_score_min
    return VoyageConfidence(
        level="high" if top_passes else "insufficient",
        top_score=top,
        second_score=second,
        margin=margin,
        mean_score=sum(ordered) / len(ordered),
        supporting_chunks=supporting,
        has_scores=True,
        top_score_passes=top_passes,
        support_count_diagnostic_passes=supporting >= profile.min_supporting_chunks,
        margin_diagnostic_passes=margin > profile.ambiguous_margin_max,
        deterministic_eligible=top_passes,
        rejection_reason=None if top_passes else "top_score_below_permit_score_min",
    )


def top_score_threshold_candidates(records: list[dict[str, Any]]) -> list[float]:
    """Return observed Voyage rank-one values only; lower ranks never tune the gate."""
    return sorted(
        {
            float(scores[0])
            for record in records
            if (scores := record.get("scores", []))
        }
    )


def profile_candidates(
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    minimum_answer_recall: float = 0.80,
    maximum_abstention_false_permit_rate: float = 0.0,
) -> list[VoyageConfidenceProfile]:
    """Build one explicit top-score-only profile per observed rank-one score."""
    return [
        VoyageConfidenceProfile(
            profile_id="voyage-rerank-2.5-lite-calibration-candidate",
            status="draft",
            permit_score_min=threshold,
            # These fixed values are reported but never used for eligibility.
            support_score_min=0.5,
            min_supporting_chunks=2,
            ambiguous_margin_max=0.1,
            minimum_answer_recall=minimum_answer_recall,
            maximum_abstention_false_permit_rate=(
                maximum_abstention_false_permit_rate
            ),
            calibration_split="stratified-v2",
            development_split="stratified-v2",
            held_out_split="stratified-v2-locked",
            notes=(
                "Generated calibration candidate; top score is the sole eligibility gate. "
                "Support and margin are diagnostic-only; never serving-enabled."
            ),
            **metadata,
        )
        for threshold in top_score_threshold_candidates(records)
    ]


def decision_metrics(
    records: list[dict[str, Any]], profile: VoyageConfidenceProfile
) -> dict[str, Any]:
    """Measure pre-grader permits strictly against declared ground truth."""
    true_positive = false_positive = true_negative = false_negative = agreement = 0
    for record in records:
        permits = voyage_confidence(record["scores"], profile).deterministic_eligible
        expected_outcome = record["expected_outcome"]
        if expected_outcome not in {"answer", "abstain"}:
            raise ValueError(
                "Voyage confidence calibration accepts only answer/abstain outcomes"
            )
        expected_answer = expected_outcome == "answer"
        expected_abstain = expected_outcome == "abstain"
        true_positive += int(permits and expected_answer)
        false_positive += int(permits and expected_abstain)
        true_negative += int(not permits and expected_abstain)
        false_negative += int(not permits and expected_answer)
        agreement += int(permits == bool(record["bge_permits"]))

    answer_total = true_positive + false_negative
    abstain_total = false_positive + true_negative
    answer_recall = true_positive / answer_total if answer_total else 0.0
    abstention_false_permit_rate = false_positive / abstain_total if abstain_total else 0.0
    abstention_recall = true_negative / abstain_total if abstain_total else 0.0
    return {
        "abstention_false_permit_rate": abstention_false_permit_rate,
        "answer_recall": answer_recall,
        "answerable_permit_rate": answer_recall,
        "balanced_answer_abstain_accuracy": (answer_recall + abstention_recall) / 2,
        "bge_decision_agreement": agreement / len(records) if records else 0.0,
        "confusion_matrix": {
            "TP": true_positive,
            "FP": false_positive,
            "TN": true_negative,
            "FN": false_negative,
        },
        "answer_cases": answer_total,
        "abstain_cases": abstain_total,
    }


def case_diagnostic(
    record: dict[str, Any], profile: VoyageConfidenceProfile
) -> dict[str, Any]:
    """Persist the complete provider-aware confidence view for one case."""
    confidence = voyage_confidence(record["scores"], profile)
    return {
        "id": record["id"],
        "split": record["split"],
        "expected_outcome": record["expected_outcome"],
        "category": record.get("confidence_category", record.get("category", "unspecified")),
        "case_type": record.get("case_type", "unspecified"),
        "voyage_top_5_scores": list(record["scores"][:5]),
        "top_score": confidence.top_score,
        "second_score": confidence.second_score,
        "margin": confidence.margin,
        "support_count": confidence.supporting_chunks,
        "confidence_predicates": {
            "has_scores": confidence.has_scores,
            "top_score_at_least_permit_min": confidence.top_score_passes,
            "support_count_at_least_diagnostic_min": (
                confidence.support_count_diagnostic_passes
            ),
            "margin_above_diagnostic_ambiguity_max": (
                confidence.margin_diagnostic_passes
            ),
            "binary_eligibility_is_top_score_only": True,
        },
        "voyage_permits": confidence.deterministic_eligible,
        "bge_permits": bool(record["bge_permits"]),
        "rejection_reason": confidence.rejection_reason,
    }


def choose_safety_first_profile(
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    minimum_answer_recall: float = 0.80,
    maximum_abstention_false_permit_rate: float = 0.0,
) -> tuple[VoyageConfidenceProfile | None, dict[str, Any]]:
    """Select only profiles satisfying independent safety and utility gates.

    Among valid profiles, maximize balanced accuracy; ties prefer a lower
    abstention false-permit rate, higher answer recall, and then the higher
    (more conservative) top-score threshold. Records explicitly marked as
    development or held-out are excluded defensively. BGE agreement is
    diagnostic-only.
    """
    if not 0.0 <= minimum_answer_recall <= 1.0:
        raise ValueError("minimum_answer_recall must be between zero and one")
    if not 0.0 <= maximum_abstention_false_permit_rate <= 1.0:
        raise ValueError(
            "maximum_abstention_false_permit_rate must be between zero and one"
        )

    selection_records = [
        record for record in records if record.get("split", "calibration") == "calibration"
    ]
    candidates = profile_candidates(
        selection_records,
        metadata=metadata,
        minimum_answer_recall=minimum_answer_recall,
        maximum_abstention_false_permit_rate=maximum_abstention_false_permit_rate,
    )
    valid: list[tuple[VoyageConfidenceProfile, dict[str, Any]]] = []
    evaluated: list[tuple[VoyageConfidenceProfile, dict[str, Any]]] = []
    for candidate in candidates:
        metrics = decision_metrics(selection_records, candidate)
        evaluated.append((candidate, metrics))
        if (
            metrics["answer_recall"] >= minimum_answer_recall
            and metrics["abstention_false_permit_rate"]
            <= maximum_abstention_false_permit_rate
        ):
            valid.append((candidate, metrics))

    def selection_key(item: tuple[VoyageConfidenceProfile, dict[str, Any]]) -> tuple[float, ...]:
        candidate, metrics = item
        return (
            -metrics["balanced_answer_abstain_accuracy"],
            metrics["abstention_false_permit_rate"],
            -metrics["answer_recall"],
            -candidate.permit_score_min,
        )

    selected = min(valid, key=selection_key) if valid else None
    diagnostic = (
        min(
            evaluated,
            key=lambda item: (
                max(
                    0.0,
                    item[1]["abstention_false_permit_rate"]
                    - maximum_abstention_false_permit_rate,
                )
                + max(0.0, minimum_answer_recall - item[1]["answer_recall"]),
                *selection_key(item),
            ),
        )
        if evaluated
        else None
    )
    return (selected[0] if selected else None), {
        "status": "valid_profile" if selected else "no_valid_profile",
        "objective": (
            "filter answer_recall >= configured minimum and abstention_false_permit_rate "
            "<= configured ceiling; then maximize balanced answer/abstain accuracy, "
            "tie-break by lower abstention false-permit rate, higher answer recall, "
            "and higher top-score threshold; BGE agreement is diagnostic-only"
        ),
        "constraints": {
            "minimum_answer_recall": minimum_answer_recall,
            "maximum_abstention_false_permit_rate": (
                maximum_abstention_false_permit_rate
            ),
        },
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "selection_case_ids": [record.get("id") for record in selection_records],
        "selected_metrics": selected[1] if selected else None,
        "diagnostic_candidate": diagnostic[0].to_dict() if diagnostic else None,
        "diagnostic_candidate_metrics": diagnostic[1] if diagnostic else None,
        "reason": None if selected else "no candidate satisfies both configured constraints",
    }


def ranking_metrics(
    cases: list[dict[str, Any]], results: dict[str, list[RetrievedChunk]]
) -> dict[str, float]:
    """Use the established Hit/Recall/MRR implementation at the top-five cut."""
    scored = [
        score_case("voyage", case, results[case["id"]])
        for case in cases
        if case["id"] in results
    ]
    return aggregate(scored, {case["id"]: case for case in cases})
