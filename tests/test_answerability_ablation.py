from __future__ import annotations

import pytest

from src.evaluation.answerability_ablation import (
    ANSWERABILITY_DECISION_SCHEMA,
    ANSWERABILITY_SYSTEM_PROMPT,
    AnswerabilityDecision,
    _exception_category,
    _serialized_answerability_schema,
    grade_answerability,
)
from src.evaluation.voyage_evidence_ablation import SanityGateResult, evaluate_path
from src.generation.schemas import AnswerResponse, Citation
from src.retrieval.types import RetrievedChunk
from src.self_healing.state import VerificationResult


class StaticChain:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, _inputs):
        self.calls += 1
        return self.payload


class RaisingChain:
    def __init__(self, error):
        self.error = error

    def invoke(self, _inputs):
        raise self.error


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        content="[Policy]\nPolicy evidence.",
        source="policy.txt",
        doc_id="POL-001",
        chunk_index=0,
    )


def payload(**overrides):
    base = {
        "relevant": True,
        "sufficient": True,
        "requested_outcome": "the requested policy outcome",
        "request_scope": "personalized",
        "material_conditions": [
            {
                "type": "eligibility",
                "value": "condition",
                "required": True,
                "condition_role": "informational_condition",
            }
        ],
        "evidence_conditions": [
            {
                "condition": "condition is satisfied",
                "status": "satisfied",
                "required": True,
                "branch_coverage": "not_applicable",
                "material_condition_types": ["eligibility"],
                "supporting_citations": ["policy.txt#0"],
            }
        ],
        "proposition_status": "entailed",
        "question_resolution": "affirmative",
        "evidence_conflict": False,
        "policy_instruction_conflict": False,
        "missing_information": [],
        "rationale": "policy directly supports the requested outcome",
        "confidence": 0.95,
    }
    base.update(overrides)
    return base


def grade(raw):
    return grade_answerability("question", [chunk()], chain=StaticChain(raw), signals={"chunk_count": 1})


def test_cross_route_transfer_is_insufficient_without_case_specific_logic():
    decision = grade(
        payload(
            material_conditions=[
                {
                    "type": "route",
                    "value": "normal return",
                    "required": True,
                    "condition_role": "informational_condition",
                },
                {
                    "type": "exception",
                    "value": "damaged route exception",
                    "required": True,
                    "condition_role": "informational_condition",
                },
            ],
            evidence_conditions=[
                {
                    "condition": "exception applies only to damaged route",
                    "status": "failed",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["route", "exception"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
            proposition_status="unsupported",
            missing_information=["evidence does not connect exception to normal return"],
        )
    )

    assert decision.sufficient is False
    assert decision.proposition_status == "unsupported"


def test_exclusive_policy_choice_contradicts_requested_conjunction():
    decision = grade(
        payload(
            material_conditions=[
                {
                    "type": "logical_operator",
                    "value": "both",
                    "required": True,
                    "condition_role": "informational_condition",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "policy permits exactly one option",
                    "status": "failed",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["logical_operator"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
            proposition_status="contradicted",
            missing_information=["policy offers an exclusive choice"],
        )
    )

    assert decision.sufficient is False
    assert decision.proposition_status == "contradicted"


def test_numeric_threshold_passed_can_be_sufficient():
    decision = grade(
        payload(
            material_conditions=[
                {
                    "type": "amount",
                    "value": "under threshold",
                    "required": True,
                    "condition_role": "informational_condition",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "customer amount is below threshold",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["amount"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is True


def test_numeric_threshold_failed_is_insufficient():
    decision = grade(
        payload(
            evidence_conditions=[
                {
                    "condition": "customer amount exceeds threshold",
                    "status": "failed",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["eligibility"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
            proposition_status="contradicted",
            missing_information=["amount fails policy threshold"],
        )
    )

    assert decision.sufficient is False


def test_policy_override_cannot_make_an_answerable_decision_sufficient():
    decision = grade(
        payload(
            policy_instruction_conflict=True,
            missing_information=["policy override request cannot change eligibility"],
        )
    )

    assert decision.sufficient is False
    assert decision.policy_instruction_conflict is True


def test_missing_eligibility_condition_is_insufficient():
    decision = grade(
        payload(
            evidence_conditions=[
                {
                    "condition": "eligibility evidence is absent",
                    "status": "unknown",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["eligibility"],
                    "supporting_citations": [],
                }
            ],
            proposition_status="unsupported",
            missing_information=["eligibility status is missing"],
        )
    )

    assert decision.sufficient is False


def test_clearly_supported_normal_answer_is_sufficient():
    assert grade(payload()).sufficient is True


def test_general_policy_with_complete_branch_coverage_is_sufficient():
    decision = grade(
        payload(
            sufficient=False,
            request_scope="general_policy",
            requested_outcome="when a parcel is formally declared lost",
            material_conditions=[
                {
                    "type": "order_type",
                    "value": "domestic and international branches",
                    "required": True,
                    "condition_role": "branch_selector",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "domestic is 10 business days and international is 21 business days",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "complete",
                    "material_condition_types": ["order_type"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is True
    assert decision.sufficiency_consistency == "normalized_true"


def test_general_policy_criteria_can_be_answered_with_all_branches_despite_possessive_wording():
    decision = grade(
        payload(
            request_scope="general_policy",
            requested_outcome="when a customer's parcel is formally declared lost",
            material_conditions=[
                {
                    "type": "delivery_destination",
                    "value": "domestic and international policy branches",
                    "required": True,
                    "condition_role": "branch_selector",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "domestic is 10 business days and international is 21 business days",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "complete",
                    "material_condition_types": ["delivery_destination"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is True


def test_complete_contradiction_can_safely_resolve_a_question_negatively():
    decision = grade(
        payload(
            sufficient=True,
            requested_outcome="a 21-day electronics return is allowed",
            material_conditions=[
                {
                    "type": "item_type",
                    "value": "electronics",
                    "required": True,
                    "condition_role": "user_specific",
                },
                {
                    "type": "elapsed_time",
                    "value": "21 days",
                    "required": True,
                    "condition_role": "user_specific",
                },
            ],
            evidence_conditions=[
                {
                    "condition": "the known 21-day purchase age exceeds the 14-day electronics window",
                    "status": "failed",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["item_type", "elapsed_time"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
            proposition_status="contradicted",
            question_resolution="negative",
        )
    )

    assert decision.sufficient is True


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "evidence_conditions": [
                {
                    "condition": "the elapsed time is unknown",
                    "status": "unknown",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["eligibility"],
                    "supporting_citations": [],
                }
            ],
            "proposition_status": "contradicted",
            "question_resolution": "negative",
            "missing_information": ["elapsed time"],
        },
        {
            "relevant": False,
            "proposition_status": "contradicted",
            "question_resolution": "negative",
        },
        {
            "policy_instruction_conflict": True,
            "proposition_status": "contradicted",
            "question_resolution": "negative",
        },
        {
            "evidence_conflict": True,
            "proposition_status": "contradicted",
            "question_resolution": "negative",
        },
    ],
)
def test_negative_resolution_stays_insufficient_when_safety_invariants_fail(overrides):
    assert grade(payload(**overrides)).sufficient is False


def test_informational_rule_reference_is_not_a_branch_selector():
    decision = grade(
        payload(
            request_scope="general_policy",
            requested_outcome="meaning of a policy reason code",
            material_conditions=[
                {
                    "type": "policy_reference",
                    "value": "a reason code",
                    "required": True,
                    "condition_role": "informational_condition",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "the reason code identifies an expired policy window",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["policy_reference"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is True


def test_unrelated_procedural_condition_does_not_block_an_independently_resolved_outcome():
    decision = grade(
        payload(
            requested_outcome="whether a damaged-on-arrival return has a label fee",
            material_conditions=[
                {
                    "type": "return_reason",
                    "value": "damaged on arrival",
                    "required": True,
                    "condition_role": "user_specific",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "damaged-on-arrival returns use the free returns path",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["return_reason"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is True


def test_personalized_question_with_unknown_branch_selector_is_insufficient():
    decision = grade(
        payload(
            request_scope="personalized",
            material_conditions=[
                {
                    "type": "order_type",
                    "value": "customer order type",
                    "required": True,
                    "condition_role": "branch_selector",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "customer order type is not known",
                    "status": "unknown",
                    "required": True,
                    "branch_coverage": "unknown",
                    "material_condition_types": ["order_type"],
                    "supporting_citations": [],
                }
            ],
            proposition_status="unsupported",
            missing_information=["domestic or international status"],
        )
    )

    assert decision.sufficient is False


def test_general_policy_with_only_one_branch_is_insufficient():
    decision = grade(
        payload(
            request_scope="general_policy",
            material_conditions=[
                {
                    "type": "order_type",
                    "value": "domestic and international branches",
                    "required": True,
                    "condition_role": "branch_selector",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "only domestic branch is present",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "partial",
                    "material_condition_types": ["order_type"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is False


def test_general_policy_with_one_unsupported_branch_is_insufficient():
    decision = grade(
        payload(
            request_scope="general_policy",
            material_conditions=[
                {
                    "type": "order_type",
                    "value": "domestic and international branches",
                    "required": True,
                    "condition_role": "branch_selector",
                }
            ],
            evidence_conditions=[
                {
                    "condition": "international branch has no evidence",
                    "status": "unknown",
                    "required": True,
                    "branch_coverage": "partial",
                    "material_condition_types": ["order_type"],
                    "supporting_citations": [],
                }
            ],
            proposition_status="unsupported",
            missing_information=["international branch evidence"],
        )
    )

    assert decision.sufficient is False


def test_model_false_with_positive_invariants_is_normalized_true():
    decision = grade(payload(sufficient=False))

    assert decision.sufficient is True
    assert decision.model_sufficient is False
    assert decision.sufficiency_consistency == "normalized_true"


def test_model_true_with_missing_required_coverage_is_forced_false():
    decision = grade(
        payload(
            material_conditions=[
                {
                    "type": "route",
                    "value": "route",
                    "required": True,
                    "condition_role": "informational_condition",
                },
                {
                    "type": "timing",
                    "value": "deadline",
                    "required": True,
                    "condition_role": "informational_condition",
                },
            ],
            evidence_conditions=[
                {
                    "condition": "route is supported",
                    "status": "satisfied",
                    "required": True,
                    "branch_coverage": "not_applicable",
                    "material_condition_types": ["route"],
                    "supporting_citations": ["policy.txt#0"],
                }
            ],
        )
    )

    assert decision.sufficient is False
    assert decision.model_sufficient is True
    assert decision.sufficiency_consistency == "normalized_false"


def test_strict_schema_declares_enum_types_and_closed_nested_objects():
    evidence = ANSWERABILITY_DECISION_SCHEMA["properties"]["evidence_conditions"]["items"]

    assert evidence["additionalProperties"] is False
    assert evidence["properties"]["status"]["type"] == "string"
    assert ANSWERABILITY_DECISION_SCHEMA["properties"]["proposition_status"]["type"] == "string"
    assert "material_condition_types" in evidence["required"]
    assert "branch_coverage" in evidence["required"]
    assert "request_scope" in ANSWERABILITY_DECISION_SCHEMA["required"]
    assert "question_resolution" in ANSWERABILITY_DECISION_SCHEMA["required"]
    assert "evidence_conflict" in ANSWERABILITY_DECISION_SCHEMA["required"]


def test_prompt_preserves_general_condition_role_and_negative_resolution_contracts():
    assert "not merely possessive wording" in ANSWERABILITY_SYSTEM_PROMPT
    assert "outcome-determinative material conditions" in ANSWERABILITY_SYSTEM_PROMPT
    assert "fact whose value selects between mutually exclusive policy branches" in (
        ANSWERABILITY_SYSTEM_PROMPT
    )
    assert "question_resolution=`negative`" in ANSWERABILITY_SYSTEM_PROMPT
    assert "evidence_conflict=true" in ANSWERABILITY_SYSTEM_PROMPT


def test_prompt_schema_is_valid_json_not_python_dict_syntax():
    import json

    rendered = _serialized_answerability_schema()

    assert json.loads(rendered) == ANSWERABILITY_DECISION_SCHEMA
    assert '"request_scope"' in rendered
    assert "'request_scope'" not in rendered


def test_strict_schema_has_no_optional_or_open_objects():
    def inspect(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["properties"]) == set(node["required"])
            if node.get("type") == "array":
                assert "items" in node
            if "enum" in node:
                assert node.get("type") == "string"
            assert "default" not in node
            assert node.get("type") != ["string", "null"]
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)

    inspect(ANSWERABILITY_DECISION_SCHEMA)


def test_answerability_failure_categories_are_sanitized_and_fail_closed():
    class ProviderError(Exception):
        def __init__(self, *, status_code=None, body=None, message="provider failure"):
            super().__init__(message)
            self.status_code = status_code
            self.body = body

    cases = [
        (ProviderError(status_code=429), "rate_limited"),
        (TimeoutError("deadline"), "timeout"),
        (ProviderError(body={"error": {"code": "json_validate_failed"}}), "structured_output_failure"),
        (ProviderError(message="connection unavailable"), "provider_unavailable"),
        (RuntimeError("request budget exhausted"), "budget_exhausted"),
    ]
    for error, category in cases:
        decision = grade_answerability(
            "question", [chunk()], chain=RaisingChain(error), signals={"chunk_count": 1}
        )
        assert decision.sufficient is False
        assert decision.failure_category == category
        assert "provider failure" not in decision.failure_reason


def test_malformed_and_schema_invalid_output_have_distinct_categories():
    malformed = grade_answerability("q", [chunk()], chain=StaticChain("not-an-object"), signals={})
    invalid = grade_answerability("q", [chunk()], chain=StaticChain({"relevant": True}), signals={})

    assert malformed.failure_category == "malformed_output"
    assert invalid.failure_category == "schema_validation_failure"
    assert malformed.failure_phase == "output_parsing"
    assert invalid.failure_phase == "schema_validation"
    assert invalid.failure_exception_type == "ValidationError"
    assert "request_scope" in invalid.failure_reason


def test_new_required_fields_report_safe_field_level_validation_failures():
    missing_scope = payload()
    missing_scope.pop("request_scope")
    missing_role = payload()
    missing_role["material_conditions"][0].pop("condition_role")
    missing_coverage = payload()
    missing_coverage["evidence_conditions"][0].pop("branch_coverage")
    invalid_enum = payload(request_scope="organization_wide")

    cases = [
        (missing_scope, "request_scope", "missing_required_field"),
        (missing_role, "material_conditions.0.condition_role", "missing_required_field"),
        (missing_coverage, "evidence_conditions.0.branch_coverage", "missing_required_field"),
        (invalid_enum, "request_scope", "invalid_enum"),
    ]
    for raw, field_path, failure_kind in cases:
        decision = grade(raw)
        assert decision.sufficient is False
        assert decision.failure_category == "schema_validation_failure"
        assert field_path in decision.failure_reason
        assert failure_kind in decision.failure_reason


def test_evaluation_path_persists_sanitized_answerability_failure():
    decision = grade_answerability(
        "q", [chunk()], chain=StaticChain({"relevant": True}), signals={}
    )
    result = evaluate_path(
        case_id="GC-test",
        question="q",
        expected_outcome="abstain",
        path="enhanced_candidate",
        chunks=[chunk()],
        gate=SanityGateResult(True, "minimal_sanity_passed", {}),
        grade_evidence=lambda: decision,
        generate_answer=lambda: (_ for _ in ()).throw(AssertionError("must not generate")),
        verify=lambda _response: VerificationResult(),
        shared_latency_ms=0.0,
        component_latency_ms={},
    )

    assert result.provider_failures == ["evidence_grader_schema_validation_failure"]
    assert len(result.stage_failures) == 1
    failure = result.stage_failures[0]
    assert failure["stage"] == "answerability_grader"
    assert failure["category"] == "schema_validation_failure"
    assert "missing_required_field" in failure["reason"]
    assert "request_scope" in failure["reason"]
    assert failure["phase"] == "schema_validation"
    assert failure["exception_type"] == "ValidationError"


def test_other_failure_category_is_sanitized():
    category, reason = _exception_category(RuntimeError("opaque failure"))

    assert category == "other"
    assert reason == "unclassified answerability grader failure"

    decision = grade_answerability(
        "question", [chunk()], chain=RaisingChain(RuntimeError("opaque failure")), signals={}
    )
    assert decision.failure_phase == "chain_invocation_unknown"
    assert decision.failure_exception_type == "RuntimeError"


def test_output_parser_failure_is_not_classified_as_other():
    class OutputParserException(RuntimeError):
        pass

    category, reason = _exception_category(OutputParserException("opaque provider payload"))

    assert category == "malformed_output"
    assert reason == "structured response parser failed"


def test_clearly_unsupported_question_is_insufficient():
    decision = grade(
        payload(
            relevant=False,
            sufficient=False,
            proposition_status="unsupported",
            missing_information=["no passage addresses the requested outcome"],
        )
    )

    assert decision.sufficient is False


def test_citation_support_remains_independent_from_answerability():
    item = chunk()
    decision = AnswerabilityDecision.model_validate(payload())
    response = AnswerResponse(
        question="question",
        answer="A grounded-looking answer.",
        outcome="answered",
        citations=[Citation.from_chunk(item)],
    )
    result = evaluate_path(
        case_id="GC-test",
        question="question",
        expected_outcome="answer",
        path="enhanced_candidate",
        chunks=[item],
        gate=SanityGateResult(True, "minimal_sanity_passed", {}),
        grade_evidence=lambda: decision,
        generate_answer=lambda: response,
        verify=lambda _response: VerificationResult(supported=False, checked=True),
        shared_latency_ms=0.0,
        component_latency_ms={},
    )

    assert result.answerability["proposition_status"] == "entailed"
    assert result.citation_supported is False
    assert result.final_outcome == "abstain"
