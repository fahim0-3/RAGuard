"""Phase I: evaluation, gates, and regression protection.

These tests are about the evaluation system itself, not about RAGuard's
answers. The question they settle is whether the machinery can be trusted to
say "worse" when things get worse — which means the interesting cases are the
ones where a gate must fail, a blocked layer must not pass, and a safety
regression must not be offset by a quality improvement.

Everything here runs without PostgreSQL, without Gemini, and without RAGAS.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.evaluation.gates import (
    RETRIEVAL_GATES,
    SAFETY_GATES,
    Gate,
    all_gates,
    evaluate_gates,
)
from src.evaluation.ragas_adapter import build_ragas_samples
from src.evaluation.ragas_eval import RAGAS_METRIC_PLAN, check_ragas_available, run_ragas_evaluation
from src.evaluation.safety_eval import evaluate_safety

# --------------------------------------------------------------------------
# 5. Threshold enforcement
# --------------------------------------------------------------------------


def test_gate_passes_when_measured_meets_target():
    gate = Gate("mrr_at_5", 0.84, "Phase D measured 0.8784")

    result = gate.check(0.8784)

    assert result.status == "PASS"
    assert result.delta == pytest.approx(0.0384, abs=1e-4)


def test_gate_fails_when_measured_falls_below_target():
    gate = Gate("mrr_at_5", 0.86, "Phase D measured 0.8784")

    result = gate.check(0.842)

    assert result.status == "FAIL"
    assert result.delta == pytest.approx(-0.018, abs=1e-4)


def test_failure_line_names_the_numbers():
    """A gate that only says "failed" teaches people to ignore it."""
    result = Gate("mrr_at_5", 0.86, "Phase D").check(0.842)

    line = result.summary_line()

    assert "0.8420" in line
    assert "0.8600" in line
    assert "-0.0180" in line
    assert "FAIL" in line


def test_max_direction_gate_bounds_a_count():
    gate = Gate("accepted_fabricated_citations", 0.0, "Phase G", "safety", direction="max")

    assert gate.check(0.0).status == "PASS"
    assert gate.check(1.0).status == "FAIL"


def test_tolerance_absorbs_noise_but_not_a_real_drop():
    gate = Gate("mrr_at_5", 0.86, "Phase D", tolerance=0.02)

    assert gate.check(0.845).status == "PASS"
    assert gate.check(0.80).status == "FAIL"


def test_gate_report_serializes_the_effective_tolerance():
    result = Gate("mrr_at_5", 0.86, "Phase D", tolerance=0.02).check(0.845)

    assert result.to_dict()["tolerance"] == 0.02


def test_safety_gates_have_zero_tolerance():
    for gate in SAFETY_GATES:
        assert gate.tolerance == 0.0, f"{gate.metric} must not tolerate any occurrence"
        assert gate.direction == "max"
        assert gate.target == 0.0


def test_every_gate_declares_provenance():
    """A threshold whose origin nobody remembers is a threshold that gets lowered."""
    for gate in all_gates():
        assert gate.provenance, f"{gate.metric} has no provenance"
        assert len(gate.provenance) > 15


# --------------------------------------------------------------------------
# 4. Regression detection
# --------------------------------------------------------------------------


def test_suite_reports_the_failing_gate():
    suite = evaluate_gates(
        RETRIEVAL_GATES, {g.metric: 1.0 for g in RETRIEVAL_GATES} | {"mrr_at_5": 0.1}
    )

    assert suite.passed is False
    assert len(suite.failures()) == 1
    assert suite.failures()[0].gate.metric == "mrr_at_5"


def test_suite_passes_when_everything_meets_its_floor():
    measured = {
        "hit_rate_at_1": 0.8182,
        "hit_rate_at_3": 0.8864,
        "hit_rate_at_5": 1.0,
        "recall_at_5": 0.9621,
        "recall_at_10": 1.0,
        "mrr_at_5": 0.8784,
        "keyword_recall": 1.0,
        "citation_id_validity": 1.0,
    }

    assert evaluate_gates(RETRIEVAL_GATES, measured).passed is True


def test_phase_d_measured_values_satisfy_the_current_floors():
    """The committed floors must not have drifted above the measured baseline."""
    report = json.loads(
        (PROJECT_ROOT / "reports" / "golden_baseline.json").read_text(encoding="utf-8")
    )

    suite = evaluate_gates(RETRIEVAL_GATES, report["measured_metrics"])

    assert suite.passed, [r.summary_line() for r in suite.failures()]


# --------------------------------------------------------------------------
# 9. Safety independence
# --------------------------------------------------------------------------


def test_missing_metric_is_not_measured_and_does_not_pass():
    """A gate with no measurement must never read as green."""
    suite = evaluate_gates([Gate("mrr_at_5", 0.84, "Phase D")], {})

    assert suite.results[0].status == "NOT_MEASURED"
    assert suite.passed is False


def test_safety_failure_is_not_offset_by_retrieval_success():
    """The scenario the whole design guards against."""
    measured = {
        "hit_rate_at_1": 1.0,
        "hit_rate_at_3": 1.0,
        "hit_rate_at_5": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr_at_5": 1.0,
        "keyword_recall": 1.0,
        "citation_id_validity": 1.0,
        "accepted_fabricated_citations": 3.0,
        "accepted_invalid_policy_ids": 0.0,
        "accepted_unsupported_claims": 0.0,
        "prompt_injection_failures": 0.0,
        "unanswerable_answered": 0.0,
        "escalation_failures": 0.0,
        "clarification_failures": 0.0,
    }

    suite = evaluate_gates([*RETRIEVAL_GATES, *SAFETY_GATES], measured)

    assert suite.category_status("retrieval") == "PASS"
    assert suite.category_status("safety") == "FAIL"
    assert suite.passed is False, "a perfect retrieval score must not buy a safety failure"


def test_safety_status_is_reported_separately():
    suite = evaluate_gates(SAFETY_GATES, {g.metric: 0.0 for g in SAFETY_GATES})

    assert suite.to_dict()["safety_status"] == "PASS"


# --------------------------------------------------------------------------
# 11-13. Safety evaluation from generation results
# --------------------------------------------------------------------------


def generation_report(cases: list[dict]) -> dict:
    return {"report": "generation_evaluation", "cases": cases}


def case(**overrides) -> dict:
    base = {
        "case_id": "GC-001",
        "case_type": "normal",
        "actual_outcome": "answer",
        "expected_outcome": "answer",
        "invalid_citation_ids": [],
        "cited_policy_ids": ["REF-001"],
        "expected_policy_ids": ["REF-001"],
        "verification_status": "supported",
        "citation_complete": True,
        "answer_preview": "Refunds take 5 to 7 business days.",
        "infrastructure_failure": False,
    }
    base.update(overrides)
    return base


def test_fabricated_citation_is_counted_as_a_safety_failure():
    report = evaluate_safety(generation_report([case(invalid_citation_ids=["ghost.txt#9"])]))

    assert report["metrics"]["accepted_fabricated_citations"] == 1.0
    assert report["affected_cases"]["accepted_fabricated_citations"] == ["GC-001"]


def test_unsupported_answer_is_counted():
    report = evaluate_safety(generation_report([case(verification_status="unsupported")]))

    assert report["metrics"]["accepted_unsupported_claims"] == 1.0


def test_clean_run_reports_zero_safety_failures():
    report = evaluate_safety(generation_report([case(), case(case_id="GC-002")]))

    assert report["metrics"]["accepted_fabricated_citations"] == 0.0
    assert report["metrics"]["accepted_unsupported_claims"] == 0.0


def test_answered_injection_case_is_a_failure():
    report = evaluate_safety(
        generation_report([case(case_id="GC-047", case_type="prompt_injection")])
    )

    assert report["metrics"]["prompt_injection_failures"] == 1.0


def test_refused_injection_case_is_not_a_failure():
    report = evaluate_safety(
        generation_report(
            [case(case_id="GC-047", case_type="prompt_injection", actual_outcome="abstain")]
        )
    )

    assert report["metrics"]["prompt_injection_failures"] == 0.0


def test_prompt_leak_in_an_answer_is_an_injection_failure():
    report = evaluate_safety(
        generation_report(
            [
                case(
                    case_id="GC-048",
                    case_type="prompt_injection",
                    actual_outcome="abstain",
                    answer_preview="You are RAGuard, a customer-support assistant...",
                )
            ]
        )
    )

    assert report["metrics"]["prompt_injection_failures"] == 1.0


def test_answered_unanswerable_case_is_a_failure():
    report = evaluate_safety(generation_report([case(case_id="GC-009", case_type="unanswerable")]))

    assert report["metrics"]["unanswerable_answered"] == 1.0


def test_high_risk_case_must_escalate():
    answered = evaluate_safety(generation_report([case(case_id="GC-050", case_type="high_risk")]))
    escalated = evaluate_safety(
        generation_report(
            [case(case_id="GC-050", case_type="high_risk", actual_outcome="escalate")]
        )
    )

    assert answered["metrics"]["escalation_failures"] == 1.0
    assert escalated["metrics"]["escalation_failures"] == 0.0


def test_missed_high_risk_escalation_fails_the_safety_gate():
    report = evaluate_safety(generation_report([case(case_id="GC-050", case_type="high_risk")]))

    suite = evaluate_gates(SAFETY_GATES, report["metrics"])

    assert suite.category_status("safety") == "FAIL"
    result = next(result for result in suite.results if result.gate.metric == "escalation_failures")
    assert result.status == "FAIL"


def test_missed_clarification_fails_the_safety_gate():
    report = evaluate_safety(generation_report([case(case_id="GC-AMB", case_type="ambiguous")]))

    suite = evaluate_gates(SAFETY_GATES, report["metrics"])

    assert suite.category_status("safety") == "FAIL"
    result = next(
        result for result in suite.results if result.gate.metric == "clarification_failures"
    )
    assert result.status == "FAIL"


def test_infrastructure_failures_are_excluded_from_safety_counts():
    """A provider outage must not be scored as a safety breach."""
    report = evaluate_safety(
        generation_report([case(actual_outcome="abstain", infrastructure_failure=True)])
    )

    assert report["dataset"]["scored_cases"] == 0


def test_generation_evaluation_exception_messages_never_enter_reports():
    from src.evaluation.generation_eval import evaluate_case

    class FailingService:
        def invoke(self, question, request_id):
            raise RuntimeError("postgresql://raguard:secret@example:5432/raguard")

    result = evaluate_case(
        {
            "id": "GC-FAIL",
            "question": "How long do refunds take?",
            "expected_outcome": "answer",
        },
        FailingService(),
        set(),
    )

    report = result.to_dict()
    assert report["failure_reason"] == "evaluation_exception:RuntimeError"
    assert "secret" not in str(report).lower()
    assert "postgresql" not in str(report).lower()


def test_safety_metrics_are_counts_not_rates():
    report = evaluate_safety(
        generation_report(
            [
                case(invalid_citation_ids=["a#1"]),
                case(case_id="GC-002", invalid_citation_ids=["b#1"]),
            ]
        )
    )

    assert report["metrics"]["accepted_fabricated_citations"] == 2.0


# --------------------------------------------------------------------------
# 8-10. RAGAS handling
# --------------------------------------------------------------------------


def test_ragas_availability_probe_never_raises():
    availability = check_ragas_available()

    assert isinstance(availability.available, bool)
    if not availability.available:
        assert availability.reason


def test_unavailable_ragas_reports_not_available_not_a_score():
    payload = run_ragas_evaluation([])

    assert payload["status"] in {"RAGAS_NOT_AVAILABLE", "NO_USABLE_SAMPLES", "MEASURED"}
    if payload["status"] == "RAGAS_NOT_AVAILABLE":
        assert payload["metrics"] == {}
        assert all(v == "UNVALIDATED" for v in payload["metric_status"].values())
        assert payload["remediation"]


def test_ragas_metric_plan_documents_purpose_and_limits():
    for entry in RAGAS_METRIC_PLAN:
        assert entry["metric"]
        assert entry["purpose"]
        assert entry["meaningful_for_raguard"]


def test_adapter_builds_samples_from_answered_cases():
    report = build_ragas_samples(
        [
            {
                "case_id": "GC-001",
                "question": "How long do refunds take?",
                "contexts": ["Refunds take 5 to 7 business days."],
                "answer": "5 to 7 business days.",
                "outcome": "answer",
                "ground_truth": "Card refunds take 5 to 7 business days.",
            }
        ]
    )

    assert report.usable == 1
    assert report.samples[0].reference == "Card refunds take 5 to 7 business days."


def test_adapter_excludes_non_answers_by_default():
    report = build_ragas_samples(
        [{"case_id": "GC-009", "outcome": "abstain", "contexts": ["x"], "ground_truth": "y"}]
    )

    assert report.usable == 0
    assert report.excluded_non_answer == ["GC-009"]


def test_adapter_never_substitutes_the_answer_for_a_missing_reference():
    """Using the model's answer as its own reference makes faithfulness meaningless."""
    report = build_ragas_samples(
        [
            {
                "case_id": "GC-002",
                "outcome": "answer",
                "contexts": ["some context"],
                "answer": "some answer",
                "ground_truth": "",
            }
        ]
    )

    assert report.usable == 0
    assert report.excluded_no_reference == ["GC-002"]


def test_adapter_documents_its_transformations():
    report = build_ragas_samples([])

    assert report.to_dict()["transformations"]


# --------------------------------------------------------------------------
# 1-2. CLI and report schema
# --------------------------------------------------------------------------


def test_cli_runs_and_reports_blocked_layers(monkeypatch, tmp_path):
    from src.evaluation import run_eval

    monkeypatch.setattr(run_eval, "check_database", lambda: (False, "DATABASE_UNAVAILABLE: test"))
    output = tmp_path / "eval.json"

    exit_code = run_eval.main(["--retrieval", "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["layers"]["retrieval"]["status"] == "BLOCKED"
    assert exit_code == 0, "without --fail-on-regression a blocked layer still reports"


def test_cli_exits_nonzero_when_a_layer_is_blocked_and_gating(monkeypatch, tmp_path):
    """A blocked evaluation is not a passed evaluation."""
    from src.evaluation import run_eval

    monkeypatch.setattr(run_eval, "check_database", lambda: (False, "DATABASE_UNAVAILABLE: test"))

    exit_code = run_eval.main(
        ["--retrieval", "--fail-on-regression", "--output", str(tmp_path / "e.json")]
    )

    assert exit_code == 1


def test_cli_reports_missing_credentials_clearly(monkeypatch, tmp_path):
    from src.evaluation import run_eval

    monkeypatch.setattr(run_eval, "check_database", lambda: (True, "22 chunks"))
    monkeypatch.setattr(run_eval, "check_llm", lambda: (False, "LLM_NOT_CONFIGURED: no key"))
    output = tmp_path / "eval.json"

    run_eval.main(["--generation", "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["layers"]["generation"]["status"] == "BLOCKED"
    assert "LLM_NOT_CONFIGURED" in payload["layers"]["generation"]["reason"]


def test_report_contains_reproducibility_metadata(monkeypatch, tmp_path):
    from src.evaluation import run_eval

    monkeypatch.setattr(run_eval, "check_database", lambda: (False, "blocked for test"))
    output = tmp_path / "eval.json"

    run_eval.main(["--retrieval", "--output", str(output)])

    repro = json.loads(output.read_text(encoding="utf-8"))["reproducibility"]
    for field in (
        "dataset_version",
        "prompt_version",
        "llm_provider",
        "generation_model",
        "judge_model",
        "embedding_model",
        "reranker_model",
        "retrieval",
        "graph",
    ):
        assert field in repro


def test_report_never_contains_a_credential(monkeypatch, tmp_path):
    from src.evaluation import run_eval

    monkeypatch.setattr(run_eval, "check_database", lambda: (False, "blocked"))
    output = tmp_path / "eval.json"

    run_eval.main(["--retrieval", "--output", str(output)])

    blob = output.read_text(encoding="utf-8").lower()
    for leak in ("google_api_key", "api_key", "password", "postgresql://"):
        assert leak not in blob


def test_reranking_layer_is_not_executed_by_the_cli():
    """A forty-minute benchmark must never run as a side effect of --all."""
    from src.evaluation.run_eval import run_reranking_layer

    result = run_reranking_layer()

    assert result["status"] == "NOT_RUN"
    assert "reranking_eval" in result["reason"]


# --------------------------------------------------------------------------
# 6, 16. Dataset version and historical immutability
# --------------------------------------------------------------------------


def test_dataset_version_is_recorded_and_current():
    from src.evaluation.metrics import golden_dataset_version

    assert golden_dataset_version() == "2026-08-29_golden_v3"


def test_historical_measured_reports_are_present_and_unmodified():
    """Phase B, C and D measurements are history, not editable state."""
    expectations = {
        "retrieval_baseline.json": ("B", "measured_metrics"),
        "reranking_comparison.json": ("C", "metrics"),
        "golden_baseline.json": ("D", "measured_metrics"),
    }

    for filename, (phase, metrics_key) in expectations.items():
        path = PROJECT_ROOT / "reports" / filename
        assert path.exists(), f"{filename} is missing"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("phase") == phase
        assert payload.get(metrics_key), f"{filename} has no measured metrics"
        assert payload.get("timestamp")


def test_phase_b_baseline_metrics_are_the_known_values():
    """Pins the historical numbers so an accidental overwrite is caught."""
    payload = json.loads(
        (PROJECT_ROOT / "reports" / "retrieval_baseline.json").read_text(encoding="utf-8")
    )

    assert payload["measured_metrics"]["mrr_at_5"] == 0.8611
    assert payload["measured_metrics"]["hit_rate_at_5"] == 1.0


def test_baseline_targets_file_holds_no_measurements():
    from src.evaluation.metrics import load_baseline

    baseline = load_baseline()

    assert "TARGET THRESHOLD" in baseline["_semantics"]
    assert "mean_top_confidence" not in baseline["retrieval"]


# --------------------------------------------------------------------------
# 14. Workflow configuration
# --------------------------------------------------------------------------


WORKFLOW_DIR = PROJECT_ROOT / ".github" / "workflows"


def workflows() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml"))


def test_workflows_exist_and_parse():
    yaml = pytest.importorskip("yaml")

    assert workflows(), "no workflows found"
    for path in workflows():
        assert yaml.safe_load(path.read_text(encoding="utf-8")), f"{path.name} is empty"


def test_no_workflow_contains_a_literal_credential():
    for path in workflows():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "GOOGLE_API_KEY" in line:
                assert "secrets." in line or "${{" in line or line.strip().startswith("#"), (
                    f"{path.name} may hard-code a key: {line.strip()}"
                )


def test_pr_ci_does_not_run_the_heavy_reranker_benchmark():
    yaml = pytest.importorskip("yaml")

    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    assert "reranking_eval" not in text, "the 40-minute benchmark must not run on every PR"
    assert "-m slow" not in text
    assert config is not None


def test_pr_ci_installs_what_the_fast_tier_actually_imports():
    """The fast tier's locked dependency group must cover its import surface."""
    workflow = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(project["dependency-groups"]["dev"])

    assert "--only-group dev" in workflow
    for package in ("fastapi", "langgraph", "jsonschema", "httpx"):
        assert package in dependencies, f"fast tier imports {package} but dev does not install it"


def test_integration_ci_provisions_postgres_with_pgvector():
    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert "pgvector/pgvector" in text
    assert "pg_isready" in text, "the database service must be health-checked"


def test_integration_ci_explicitly_enables_model_backed_tests():
    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert 'RAGUARD_ALLOW_HEAVY_TESTS: "1"' in text, (
        "integration tests marked heavy otherwise collect successfully but silently skip"
    )


def test_llm_evaluation_is_a_separate_workflow_from_the_merge_gate():
    ci = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")

    assert "GOOGLE_API_KEY" not in ci or "# " in ci, (
        "the deterministic merge gate must not depend on a provider key"
    )


# --------------------------------------------------------------------------
# A blocked layer is not a pass
# --------------------------------------------------------------------------


def _suite(*results):
    from src.evaluation.gates import GateSuite

    return GateSuite(results=list(results))


def test_blocked_layer_never_reports_pass():
    """Zero failing gates is not the same thing as a passing evaluation.

    Regression guard: the CLI previously derived `overall` from the gate status
    alone, so a run whose only layer was BLOCKED applied no gates and printed
    PASS. A missing database must never read as a green build.
    """
    from src.evaluation.run_eval import overall_status

    layers = {"safety": {"status": "BLOCKED", "reason": "SAFETY_REQUIRES_GENERATION"}}

    assert overall_status(layers, _suite()) == "BLOCKED"


def test_blocked_layer_outranks_passing_gates():
    """One measured layer passing does not excuse another that never ran."""
    from src.evaluation.gates import Gate, GateResult
    from src.evaluation.run_eval import overall_status

    passing = GateResult(
        gate=Gate(
            metric="mrr_at_5", category="retrieval", target=0.84, provenance="Phase D measured"
        ),
        measured=0.88,
        status="PASS",
    )
    layers = {
        "retrieval": {"status": "MEASURED"},
        "generation": {"status": "BLOCKED", "reason": "LLM_NOT_CONFIGURED"},
    }

    assert overall_status(layers, _suite(passing)) == "BLOCKED"


def test_no_gates_at_all_is_not_measured():
    from src.evaluation.run_eval import overall_status

    assert overall_status({}, _suite()) == "NOT_MEASURED"


def test_failing_gate_reports_fail():
    from src.evaluation.gates import Gate, GateResult
    from src.evaluation.run_eval import overall_status

    failing = GateResult(
        gate=Gate(
            metric="mrr_at_5", category="retrieval", target=0.90, provenance="Phase D measured"
        ),
        measured=0.80,
        status="FAIL",
    )

    assert overall_status({"retrieval": {"status": "MEASURED"}}, _suite(failing)) == "FAIL"


def test_report_records_overall_status_and_blocked_layers():
    from src.evaluation.run_eval import build_report

    layers = {"safety": {"status": "BLOCKED", "reason": "SAFETY_REQUIRES_GENERATION"}}
    payload = build_report(layers, _suite())

    assert payload["overall_status"] == "BLOCKED"
    assert payload["blocked_layers"] == ["safety"]


# --------------------------------------------------------------------------
# Generation records must satisfy the RAGAS adapter's contract
# --------------------------------------------------------------------------


def test_generation_record_carries_every_field_the_adapter_reads():
    """Regression guard for a silent Phase I integration bug.

    The evaluator emitted `actual_outcome` / `answer_preview` and no
    `ground_truth` or `contexts`, while the adapter read `outcome`, `answer`,
    `ground_truth`, and `contexts`. Nothing raised: the adapter simply excluded
    every case, so RAGAS would have reported on an empty dataset.
    """
    from src.evaluation.generation_eval import GenerationCaseResult

    record = GenerationCaseResult(
        case_id="GC-001",
        question="How long do refunds take?",
        expected_outcome="answer",
        actual_outcome="answer",
        outcome_matched=True,
        answer="Refunds take 5 to 7 business days.",
        ground_truth="Card refunds are released in 5 to 7 business days.",
        contexts=["Refunds to credit and debit cards take 5 to 7 business days."],
    ).to_dict()

    for key in ("outcome", "answer", "contexts", "ground_truth"):
        assert key in record, f"the RAGAS adapter reads {key!r} and it is absent"


def test_adapter_accepts_a_real_generation_record():
    from src.evaluation.generation_eval import GenerationCaseResult

    record = GenerationCaseResult(
        case_id="GC-001",
        question="How long do refunds take?",
        expected_outcome="answer",
        actual_outcome="answer",
        outcome_matched=True,
        answer="Refunds take 5 to 7 business days.",
        ground_truth="Card refunds are released in 5 to 7 business days.",
        contexts=["Refunds to credit and debit cards take 5 to 7 business days."],
    ).to_dict()

    report = build_ragas_samples([record])

    assert len(report.samples) == 1, report.to_dict()
    assert report.excluded_no_reference == []
    assert report.samples[0].reference


def test_adapter_still_refuses_to_invent_a_reference():
    """A case with no hand-written ground truth is excluded, never auto-filled."""
    from src.evaluation.generation_eval import GenerationCaseResult

    record = GenerationCaseResult(
        case_id="GC-XXX",
        question="q",
        expected_outcome="answer",
        actual_outcome="answer",
        outcome_matched=True,
        answer="Some generated answer.",
        ground_truth="",
        contexts=["some passage"],
    ).to_dict()

    report = build_ragas_samples([record])

    assert len(report.samples) == 0
    assert report.excluded_no_reference == ["GC-XXX"]


def test_ragas_layer_passes_full_answers_and_retrieved_passages(monkeypatch):
    """RAGAS must never score a preview or policy identifier as evidence."""
    from src.evaluation import ragas_eval, run_eval

    captured: dict[str, list[dict]] = {}

    def fake_ragas(records):
        captured["records"] = records
        return {"status": "MEASURED", "metrics": {"faithfulness": 1.0}}

    monkeypatch.setattr(ragas_eval, "run_ragas_evaluation", fake_ragas)
    run_eval.run_ragas_layer(
        {
            "cases": [
                {
                    "case_id": "GC-001",
                    "question": "How long do refunds take?",
                    "answer": "Full answer: refunds take 5 to 7 business days.",
                    "answer_preview": "Full answer: refunds take",
                    "actual_outcome": "answer",
                    "contexts": ["Card refunds take 5 to 7 business days."],
                    "cited_policy_ids": ["REF-001"],
                    "ground_truth": "Card refunds take 5 to 7 business days.",
                }
            ]
        }
    )

    assert captured["records"] == [
        {
            "case_id": "GC-001",
            "question": "How long do refunds take?",
            "answer": "Full answer: refunds take 5 to 7 business days.",
            "outcome": "answer",
            "contexts": ["Card refunds take 5 to 7 business days."],
            "ground_truth": "Card refunds take 5 to 7 business days.",
        }
    ]


@pytest.mark.evaluation
def test_ragas_execution_failures_keep_exception_messages_out_of_reports(monkeypatch):
    from src.evaluation import ragas_eval

    monkeypatch.setattr(
        ragas_eval,
        "check_ragas_available",
        lambda: ragas_eval.RagasAvailability(True, "0.4.3", None),
    )
    monkeypatch.setattr(
        ragas_eval,
        "_build_ragas_embeddings",
        lambda: (_ for _ in ()).throw(RuntimeError("postgresql://secret@example")),
    )

    report = ragas_eval.run_ragas_evaluation(
        [
            {
                "case_id": "GC-001",
                "question": "How long do refunds take?",
                "answer": "Refunds take 5 to 7 business days.",
                "outcome": "answer",
                "contexts": ["Card refunds take 5 to 7 business days."],
                "ground_truth": "Card refunds take 5 to 7 business days.",
            }
        ]
    )

    assert report["status"] == "RAGAS_EXECUTION_FAILED"
    assert report["error"] == "RuntimeError"
    assert "secret" not in str(report)


@pytest.mark.evaluation
def test_ragas_uses_the_configured_gemini_embedding_backend(monkeypatch):
    from src.config.settings import Settings
    from src.evaluation import ragas_eval

    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="not-a-real-key",
        gemini_embedding_model="gemini-embedding-001",
        vector_dimension=1024,
    )
    monkeypatch.setattr(ragas_eval, "get_settings", lambda: settings)

    embeddings = ragas_eval._build_ragas_embeddings()

    assert embeddings.embeddings.model == "gemini-embedding-001"
    assert embeddings.embeddings.task_type == "SEMANTIC_SIMILARITY"
    assert embeddings.embeddings.output_dimensionality == 1024


@pytest.mark.evaluation
def test_ragas_passes_explicit_embeddings_to_answer_relevancy(monkeypatch):
    import ragas
    import ragas.llms

    from src.evaluation import ragas_eval
    from src.generation import llm_factory

    captured: dict[str, object] = {}
    embeddings = object()
    judge = object()

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"faithfulness": 1.0, "answer_relevancy": 1.0}

    monkeypatch.setattr(
        ragas_eval,
        "check_ragas_available",
        lambda: ragas_eval.RagasAvailability(True, "0.4.3", None),
    )
    monkeypatch.setattr(ragas_eval, "_build_ragas_embeddings", lambda: embeddings)
    monkeypatch.setattr(llm_factory, "get_chat_model", lambda role: object())
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda model: judge)
    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)

    report = ragas_eval.run_ragas_evaluation(
        [
            {
                "case_id": "GC-001",
                "question": "How long do refunds take?",
                "answer": "Refunds take 5 to 7 business days.",
                "outcome": "answer",
                "contexts": ["Card refunds take 5 to 7 business days."],
                "ground_truth": "Card refunds take 5 to 7 business days.",
            }
        ]
    )

    assert report["status"] == "MEASURED"
    assert captured["embeddings"] is embeddings
    assert captured["llm"] is judge
    assert captured["metrics"][1].embeddings is embeddings
