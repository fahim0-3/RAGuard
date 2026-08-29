"""Request-wide deadline and LLM-call budget behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.retrieval.types import RetrievedChunk
from src.self_healing.execution_budget import (
    ExecutionBudget,
    ExecutionBudgetExceeded,
    ensure_time_remaining,
    request_budget,
    reserve_llm_call,
)
from src.self_healing.verification import EntailmentVerifier


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_budget_shares_one_llm_call_limit_across_stages():
    clock = Clock()
    budget = ExecutionBudget(timeout_s=120, max_llm_calls=2, clock=clock)

    with request_budget(budget):
        first = reserve_llm_call("evidence_grader", default_timeout_s=60)
        second = reserve_llm_call("generate_answer", default_timeout_s=60)
        with pytest.raises(ExecutionBudgetExceeded, match="llm_call_limit"):
            reserve_llm_call("verify_citations", default_timeout_s=60)

    assert first.timeout_s == 60
    assert second.timeout_s == 60
    assert budget.llm_calls_used == 2
    assert budget.exhausted_reason == "llm_call_limit"
    assert budget.exhausted_stage == "verify_citations"


def test_budget_caps_each_provider_timeout_to_the_remaining_deadline():
    clock = Clock()
    budget = ExecutionBudget(timeout_s=20, max_llm_calls=3, clock=clock)

    with request_budget(budget):
        clock.now += 13.5
        permit = reserve_llm_call("generate_answer", default_timeout_s=60)

    assert permit.timeout_s == pytest.approx(6.5)
    assert permit.max_retries == 0


def test_expired_deadline_refuses_a_provider_call_without_consuming_it():
    clock = Clock()
    budget = ExecutionBudget(timeout_s=10, max_llm_calls=3, clock=clock)

    with request_budget(budget):
        clock.now += 10.1
        with pytest.raises(ExecutionBudgetExceeded, match="deadline"):
            reserve_llm_call("evidence_grader", default_timeout_s=60)

    assert budget.llm_calls_used == 0
    assert budget.exhausted_reason == "deadline"


def test_expired_deadline_stops_non_llm_work_at_the_next_graph_boundary():
    clock = Clock()
    budget = ExecutionBudget(timeout_s=10, max_llm_calls=3, clock=clock)

    with request_budget(budget):
        clock.now += 10.1
        with pytest.raises(ExecutionBudgetExceeded, match="hybrid_retrieve"):
            ensure_time_remaining("hybrid_retrieve")

    assert budget.llm_calls_used == 0
    assert budget.exhausted_stage == "hybrid_retrieve"


def test_request_timeout_must_finish_before_the_admission_lease():
    with pytest.raises(ValidationError, match="graph_request_timeout_s"):
        Settings(
            _env_file=None,
            graph_request_timeout_s=300,
            admission_lease_seconds=300,
        )


def test_entailment_reserves_one_budgeted_call_per_claim():
    class Judge:
        def invoke(self, _payload):
            return {"supported": True, "confidence": 1.0, "reason": "supported"}

    evidence = [
        RetrievedChunk(
            chunk_id=1,
            content=(
                "Approved card refunds are released in 5 to 7 business days. "
                "The original payment method receives the refund."
            ),
            source="refund_policy.txt",
            chunk_index=0,
            doc_id="REF-001",
        )
    ]
    label = evidence[0].citation_label
    claims = [
        "A card refund should arrive in 5 to 7 business days.",
        "The money is returned through the payment method originally used.",
    ]
    budget = ExecutionBudget(timeout_s=120, max_llm_calls=1)

    with request_budget(budget), pytest.raises(ExecutionBudgetExceeded, match="llm_call_limit"):
        EntailmentVerifier(chain=Judge(), use_llm=True).verify(
            " ".join(claims),
            [label],
            evidence,
            [{"claim": claim, "citations": [label]} for claim in claims],
        )

    assert budget.llm_calls_used == 1
    assert budget.exhausted_stage == "verify_citations"
