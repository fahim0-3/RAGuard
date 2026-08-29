"""Live, calibration-only evaluation of Voyage confidence thresholds.

This script never changes ``RERANKER_CONFIDENCE_PROFILE`` or serving code. It
requires an explicit Voyage opt-in because each request sends the query and the
top-20 policy chunks to Voyage.

Example:
    $env:RERANKER_PROVIDER='voyage'
    $env:RERANKER_REMOTE_ALLOWED='true'
    $env:VOYAGE_API_KEY='...'
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_voyage_confidence.py `
      --allow-remote --output reports\voyage_confidence_calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.evaluation.metrics import golden_dataset_version, load_golden_dataset  # noqa: E402
from src.evaluation.voyage_confidence import (  # noqa: E402
    VoyageConfidenceProfile,
    case_diagnostic,
    choose_safety_first_profile,
    confidence_stratum,
    decision_metrics,
    ranking_metrics,
    split_composition,
    stratified_split_assignments,
    voyage_confidence,
)


def _corpus_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(get_settings().absolute_data_dir.glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()[:16]}"


def _metadata(settings: Any) -> dict[str, Any]:
    return {
        "provider": "voyage",
        "model": settings.voyage_rerank_model,
        "dataset_version": golden_dataset_version(),
        "corpus_fingerprint": _corpus_fingerprint(),
        "chunking_fingerprint": (
            f"chunk_size={settings.chunk_size};chunk_overlap={settings.chunk_overlap}"
        ),
        "candidate_top_k": settings.rerank_candidate_top_k,
        "final_top_k": settings.rerank_top_k,
    }


def select_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the stable dataset order; zero deliberately means every case."""
    if limit < 0:
        raise ValueError("limit must be zero or greater")
    return cases if limit == 0 else cases[:limit]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(statistics.mean(values), 1),
        "p50": round(statistics.median(values), 1),
        "p95": round(ordered[round((len(ordered) - 1) * 0.95)], 1),
    }


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse either form allowed by the HTTP ``Retry-After`` header."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - (now or datetime.now(UTC))).total_seconds())


class RetryAfterTrackingClient:
    """Calibration-only HTTP wrapper; it changes no shared provider policy."""

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        import httpx

        self.retry_after: float | None = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def post(self, *args: Any, **kwargs: Any) -> Any:
        self.retry_after = None
        response = self._client.post(*args, **kwargs)
        if getattr(response, "status_code", None) == 429:
            self.retry_after = retry_after_seconds(response.headers.get("Retry-After"))
        return response


def rerank_voyage_with_rate_limit_retry(
    reranker: Any,
    query: str,
    chunks: list[Any],
    *,
    top_k: int,
    candidate_top_k: int,
    max_retries: int,
    retry_base_seconds: float,
    retry_after: Callable[[], float | None],
    progress: Callable[[str], None],
    prefix: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, int, list[dict[str, float | int | str]]]:
    """Retry only a 429 in this live benchmark; never replace Voyage data."""
    retries = 0
    events: list[dict[str, float | int | str]] = []
    while True:
        result = reranker.rerank_with_diagnostics(
            query, chunks, top_k=top_k, candidate_top_k=candidate_top_k
        )
        if result.reranker_used:
            return result, retries, events
        if result.failure != "voyage_rate_limited":
            raise RuntimeError(str(result.failure or "voyage_failed"))
        if retries >= max_retries:
            raise RuntimeError("voyage_rate_limited")

        retries += 1
        delay = retry_after()
        delay_source = "retry_after" if delay is not None else "exponential_backoff"
        if delay is None:
            delay = min(retry_base_seconds * (2 ** (retries - 1)), 60.0)
        events.append({"retry": retries, "delay_seconds": delay, "source": delay_source})
        progress(f"{prefix} Voyage rate limited - retry {retries}/{max_retries} in {delay:.1f}s")
        sleep(delay)


def _bge_permits(query: str, chunks: list[Any]) -> tuple[bool, dict[str, Any]]:
    """Record both current BGE guard families without changing either one."""
    # Import only for a real opt-in run. This keeps ``--help`` and the remote
    # confirmation guard usable in a minimal developer environment.
    from src.self_healing.confidence import score_retrieval
    from src.self_healing.evidence_grader import deterministic_signals

    confidence = score_retrieval(chunks)
    signals = deterministic_signals(query, chunks)
    # This is exactly the graph's numerical branch after its earlier empty,
    # exact-ID, and minimum-chunk checks. The structured evidence grader is
    # intentionally not invoked by calibration: it is a separate live LLM
    # evaluation that must be run with a pinned provider and prompt.
    evidence_ok = bool(
        signals["policy_id_exact_match"]
        or (
            not signals["policy_id_requested_but_missing"]
            and signals["chunk_count"] >= get_settings().evidence_min_relevant_chunks
            and signals["top_score"] >= get_settings().evidence_top_score_threshold
        )
    )
    return bool(confidence.level != "insufficient" and evidence_ok), {
        "legacy_confidence": confidence.to_dict(),
        "evidence_signals": signals,
        "bge_evidence_permits": evidence_ok,
    }


def _evaluate(
    records: list[dict[str, Any]], profile: VoyageConfidenceProfile
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disagreements: list[dict[str, Any]] = []
    for record in records:
        voyage = voyage_confidence(record["scores"], profile)
        voyage_permits = voyage.deterministic_eligible
        if voyage_permits != record["bge_permits"]:
            disagreements.append(
                {
                    "id": record["id"],
                    "expected_outcome": record["expected_outcome"],
                    "bge_permits": record["bge_permits"],
                    "voyage_permits": voyage_permits,
                    "voyage_confidence": voyage.to_dict(),
                    "voyage_chunk_ids": record["voyage_chunk_ids"],
                    "voyage_raw_scores": record["scores"],
                }
            )
    return decision_metrics(records, profile), disagreements


def _citation_evaluation(
    records: list[dict[str, Any]],
    rankings: dict[str, list[Any]],
    profile: Any,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Optional live answer/citation check over the Voyage-selected passages.

    It intentionally does not alter the graph.  This is a calibration report
    only; a future provider-aware graph gate still requires separate approval.
    """
    from src.generation.answer_chain import generate_grounded_answer
    from src.self_healing.verification import get_default_verifier

    verifier = get_default_verifier()
    rows: list[dict[str, Any]] = []
    answers = correct = false_positive = supported = 0
    total = len(records)
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        prefix = f"[{index}/{total}]"
        progress(f"{prefix} LLM verification...")
        confidence = voyage_confidence(record["scores"], profile)
        permitted = confidence.deterministic_eligible
        expected_answer = not record["should_abstain"]
        if not permitted:
            actual_answer = False
            citation_supported = False
        else:
            response = generate_grounded_answer(record["question"], rankings[record["id"]])
            verification = verifier.verify(
                response.answer,
                response.citation_ids,
                rankings[record["id"]],
                response.claim_citations,
            )
            actual_answer = bool(response.outcome == "answered" and verification.supported)
            citation_supported = bool(verification.supported)
        answers += int(actual_answer)
        correct += int(actual_answer == expected_answer)
        false_positive += int(actual_answer and not expected_answer)
        supported += int(citation_supported)
        rows.append(
            {
                "id": record["id"],
                "expected_outcome": record["expected_outcome"],
                "voyage_permitted": permitted,
                "actual_answer": actual_answer,
                "citation_supported": citation_supported,
            }
        )
        progress(f"{prefix} done - {(time.perf_counter() - started):.1f}s")
    count = max(1, len(records))
    return {
        "status": "completed",
        "metrics": {
            "answer_abstain_accuracy": correct / count,
            "false_positive_answer_rate": false_positive / count,
            "citation_support_rate": supported / max(1, answers),
        },
        "cases": rows,
    }


def run(
    *,
    with_llm: bool = False,
    limit: int = 0,
    voyage_max_retries: int = 4,
    voyage_retry_base_seconds: float = 5.0,
    voyage_request_delay_seconds: float = 1.0,
    minimum_calibration_answer_recall: float = 0.80,
    maximum_calibration_abstention_false_permit_rate: float = 0.0,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.reranking.provider import VoyageReranker
    from src.retrieval.hybrid import get_hybrid_retriever

    settings = get_settings()
    if settings.reranker_provider != "voyage" or not settings.reranker_remote_allowed:
        raise RuntimeError("set RERANKER_PROVIDER=voyage and RERANKER_REMOTE_ALLOWED=true")
    if not settings.voyage_api_key:
        raise RuntimeError("set VOYAGE_API_KEY")
    if voyage_max_retries < 0 or voyage_retry_base_seconds < 0 or voyage_request_delay_seconds < 0:
        raise ValueError("Voyage retry and pacing values must be zero or greater")
    if not 0.0 <= minimum_calibration_answer_recall <= 1.0:
        raise ValueError("minimum calibration answer recall must be between zero and one")
    if not 0.0 <= maximum_calibration_abstention_false_permit_rate <= 1.0:
        raise ValueError(
            "maximum calibration abstention false-permit rate must be between zero and one"
        )

    # Ambiguous and high-risk questions route before retrieval, so they are
    # reported but excluded from reranker confidence threshold fitting.
    cases = select_cases([
        case
        for case in load_golden_dataset()
        if case["expected_outcome"] in {"answer", "abstain"}
    ], limit)
    if not cases:
        raise RuntimeError("no calibration/evaluation cases selected")
    split_assignments = stratified_split_assignments(cases)
    progress = progress or (lambda message: print(message, flush=True))
    retriever = get_hybrid_retriever()
    local = CrossEncoderReranker(device=settings.resolved_reranker_device)
    voyage_client = RetryAfterTrackingClient(
        api_key=settings.voyage_api_key,
        timeout_seconds=settings.hosted_rerank_timeout_seconds,
    )
    voyage = VoyageReranker(
        api_key=settings.voyage_api_key,
        model_name=settings.voyage_rerank_model,
        timeout_seconds=settings.hosted_rerank_timeout_seconds,
        # The calibration-only wrapper below owns 429 pacing. Serving retains
        # the configured provider retries and fallback behaviour unchanged.
        max_retries=0,
        client=voyage_client,
    )
    records: list[dict[str, Any]] = []
    local_rankings: dict[str, list[Any]] = {}
    voyage_rankings: dict[str, list[Any]] = {}
    voyage_latencies: list[float] = []
    bge_latencies: list[float] = []
    total_latencies: list[float] = []
    rate_limit_events: list[dict[str, float | int | str]] = []
    previous_voyage_success = False

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        prefix = f"[{index}/{len(cases)}]"
        progress(f"{prefix} retrieval...")
        candidates = retriever.retrieve(case["question"])
        if previous_voyage_success and voyage_request_delay_seconds:
            progress(f"{prefix} Voyage pacing - waiting {voyage_request_delay_seconds:.1f}s")
            sleep(voyage_request_delay_seconds)
        progress(f"{prefix} Voyage...")
        try:
            voyage_result, voyage_retry_count, case_rate_limit_events = (
                rerank_voyage_with_rate_limit_retry(
                    voyage,
                    case["question"],
                    candidates,
                    top_k=min(settings.rerank_top_k, settings.hosted_rerank_top_k),
                    candidate_top_k=min(
                        settings.rerank_candidate_top_k, settings.hosted_rerank_max_candidates
                    ),
                    max_retries=voyage_max_retries,
                    retry_base_seconds=voyage_retry_base_seconds,
                    retry_after=lambda: voyage_client.retry_after,
                    progress=progress,
                    prefix=prefix,
                    sleep=sleep,
                )
            )
        except RuntimeError as exc:
            raise RuntimeError(f"reranker unavailable for {case['id']}: {exc}") from exc
        previous_voyage_success = True
        progress(f"{prefix} BGE...")
        bge_started = time.perf_counter()
        local_result = local.rerank_with_diagnostics(case["question"], candidates)
        bge_latency_ms = (time.perf_counter() - bge_started) * 1000.0
        if not local_result.reranker_used:
            raise RuntimeError(f"reranker unavailable for {case['id']}: {local_result.failure}")
        scores = [voyage_result.provider_raw_scores[chunk_id] for chunk_id in voyage_result.provider_order]
        bge_permits, bge_detail = _bge_permits(case["question"], local_result.chunks)
        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_outcome": case["expected_outcome"],
                "should_abstain": bool(case["should_abstain"]),
                "case_type": case.get("case_type", "unspecified"),
                "category": case.get("category", "unspecified"),
                "confidence_category": confidence_stratum(case),
                "split": split_assignments[case["id"]],
                "scores": scores,
                "voyage_chunk_ids": voyage_result.provider_order,
                "voyage_retry_count": voyage_retry_count,
                "voyage_rate_limit_events": case_rate_limit_events,
                "bge_permits": bge_permits,
                **bge_detail,
            }
        )
        local_rankings[case["id"]] = local_result.chunks
        voyage_rankings[case["id"]] = voyage_result.chunks
        voyage_latencies.append(voyage_result.hosted_latency_ms)
        bge_latencies.append(bge_latency_ms)
        total_latency_ms = (time.perf_counter() - started) * 1000.0
        total_latencies.append(total_latency_ms)
        rate_limit_events.extend(
            [{"case_id": case["id"], **event} for event in case_rate_limit_events]
        )
        progress(
            f"{prefix} latency Voyage={voyage_result.hosted_latency_ms:.1f}ms "
            f"BGE={bge_latency_ms:.1f}ms total={total_latency_ms:.1f}ms | "
            f"running mean Voyage={statistics.mean(voyage_latencies):.1f}ms "
            f"BGE={statistics.mean(bge_latencies):.1f}ms total={statistics.mean(total_latencies):.1f}ms"
        )
        if not with_llm:
            progress(f"{prefix} done - {total_latency_ms / 1000.0:.1f}s")

    calibration = [record for record in records if record["split"] == "calibration"]
    development = [record for record in records if record["split"] == "development"]
    held_out = [record for record in records if record["split"] == "held_out"]
    profile, fit_metrics = choose_safety_first_profile(
        calibration,
        metadata=_metadata(settings),
        minimum_answer_recall=minimum_calibration_answer_recall,
        maximum_abstention_false_permit_rate=(
            maximum_calibration_abstention_false_permit_rate
        ),
    )
    if profile is not None:
        profile = replace(
            profile,
            profile_id="voyage-rerank-2.5-lite-v1",
            notes="Draft; held-out evaluation required before any serving use.",
        )
        evaluation_profile = profile
    else:
        diagnostic_candidate = fit_metrics["diagnostic_candidate"]
        if diagnostic_candidate is None:
            raise RuntimeError("no Voyage top scores available for calibration")
        evaluation_profile = VoyageConfidenceProfile(**diagnostic_candidate)

    calibration_metrics, calibration_disagreements = _evaluate(
        calibration, evaluation_profile
    )
    development_metrics, development_disagreements = _evaluate(
        development, evaluation_profile
    )
    held_out_metrics, held_out_disagreements = _evaluate(held_out, evaluation_profile)
    all_diagnostics = [case_diagnostic(record, evaluation_profile) for record in records]
    # No automatic GO is permitted: the full-graph citation evaluation remains
    # a separate required evidence set.
    recommendation = (
        "NO-GO: NO VALID PROFILE satisfies the calibration constraints; "
        "production BGE-5 remains required"
        if profile is None
        else "NO-GO: calibration report only; production BGE-5 remains required"
    )
    citation_support = (
        _citation_evaluation(records, voyage_rankings, evaluation_profile, progress)
        if with_llm and profile is not None
        else {
            "status": "not_run",
            "reason": (
                "no valid calibration profile; LLM evaluation was not run"
                if with_llm
                else "pass --with-llm for the pinned live answer/citation evaluation"
            ),
        }
    )
    return {
        "report": "voyage_confidence_calibration_v1",
        "production_changed": False,
        "recommendation": recommendation,
        "profile": profile.to_dict() if profile is not None else None,
        "diagnostic_profile": (
            evaluation_profile.to_dict() if profile is None else None
        ),
        "fit_metrics": fit_metrics,
        "splits": split_composition(cases, split_assignments),
        "latency_ms": {
            "voyage": _latency_summary(voyage_latencies),
            "bge": _latency_summary(bge_latencies),
            "total": _latency_summary(total_latencies),
        },
        "voyage_rate_limits": {
            "max_retries": voyage_max_retries,
            "retry_base_seconds": voyage_retry_base_seconds,
            "request_delay_seconds": voyage_request_delay_seconds,
            "event_count": len(rate_limit_events),
            "events": rate_limit_events,
            "total_retries": sum(record["voyage_retry_count"] for record in records),
        },
        "ranking": {
            "local_bge": ranking_metrics(cases, local_rankings),
            "voyage": ranking_metrics(cases, voyage_rankings),
        },
        "calibration": {"metrics": calibration_metrics, "disagreements": calibration_disagreements},
        "development": {
            "metrics": development_metrics,
            "disagreements": development_disagreements,
        },
        "held_out": {"metrics": held_out_metrics, "disagreements": held_out_disagreements},
        "case_diagnostics": all_diagnostics,
        "citation_support": citation_support,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Voyage confidence without serving changes")
    parser.add_argument("--allow-remote", action="store_true", help="confirm live Voyage document transfer")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="also run live grounded-answer and citation-verifier checks (consumes LLM quota)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="run the first N answer/abstain calibration cases; 0 runs all",
    )
    parser.add_argument(
        "--voyage-max-retries",
        type=int,
        default=4,
        help="calibration-only retries for a Voyage 429 (default: 4)",
    )
    parser.add_argument(
        "--voyage-retry-base-seconds",
        type=float,
        default=5.0,
        help="calibration-only exponential-backoff base when Retry-After is absent (default: 5)",
    )
    parser.add_argument(
        "--voyage-request-delay-seconds",
        type=float,
        default=1.0,
        help="delay between successful Voyage calls in calibration (default: 1)",
    )
    parser.add_argument(
        "--minimum-calibration-answer-recall",
        type=float,
        default=0.80,
        help="minimum pre-grader permit recall on answerable calibration cases (default: 0.80)",
    )
    parser.add_argument(
        "--maximum-calibration-abstention-false-permit-rate",
        type=float,
        default=0.0,
        help="maximum pre-grader permit rate on abstention calibration cases (default: 0.0)",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports" / "voyage_confidence_calibration.json"
    )
    args = parser.parse_args()
    if not args.allow_remote:
        parser.error("--allow-remote is required; this script sends candidate chunks to Voyage")
    if (
        args.limit < 0
        or args.voyage_max_retries < 0
        or args.voyage_retry_base_seconds < 0
        or args.voyage_request_delay_seconds < 0
        or not 0.0 <= args.minimum_calibration_answer_recall <= 1.0
        or not 0.0
        <= args.maximum_calibration_abstention_false_permit_rate
        <= 1.0
    ):
        parser.error(
            "--limit and Voyage retry/pacing values must be non-negative; "
            "calibration constraints must be between zero and one"
        )
    try:
        report = run(
            with_llm=args.with_llm,
            limit=args.limit,
            voyage_max_retries=args.voyage_max_retries,
            voyage_retry_base_seconds=args.voyage_retry_base_seconds,
            voyage_request_delay_seconds=args.voyage_request_delay_seconds,
            minimum_calibration_answer_recall=(
                args.minimum_calibration_answer_recall
            ),
            maximum_calibration_abstention_false_permit_rate=(
                args.maximum_calibration_abstention_false_permit_rate
            ),
        )
    except RuntimeError as exc:
        print(f"CALIBRATION_BLOCKED: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"recommendation": report["recommendation"], "held_out": report["held_out"]}, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
