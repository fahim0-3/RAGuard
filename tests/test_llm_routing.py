"""Deterministic provider selection and graph-scoped fallback behavior."""

from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.generation.llm_routing import (
    advance_route,
    current_provider,
    is_retryable_provider_error,
    route_context,
    select_route,
    workload_context,
)
from src.self_healing.execution_budget import ExecutionBudget, request_budget, reserve_llm_call


class MockGroqJsonValidateFailed(RuntimeError):
    """Minimal Groq-like 400 response; no network or SDK dependency."""

    status_code = 400
    body = {"error": {"code": "json_validate_failed", "message": "Failed to validate JSON"}}


class MockGroqArbitraryBadRequest(RuntimeError):
    status_code = 400
    body = {"error": {"code": "invalid_request_error", "message": "Bad request"}}


def routed_settings(**overrides: object) -> Settings:
    """Use explicit credentials so tests never depend on a developer `.env`."""
    base = Settings(
        _env_file=None,
        llm_routing_mode="dynamic",
        google_api_key="g" * 32,
        groq_api_key="r" * 32,
    )
    return base.model_copy(update=overrides)


def test_static_mode_preserves_manual_llm_provider_selection():
    settings = routed_settings(llm_routing_mode="static", llm_provider="groq")

    route = select_route(settings)

    assert route.provider == "groq"
    assert route.candidates == ("groq",)
    assert route.can_fallback is False


def test_dynamic_normal_rag_prefers_gemini_then_groq_then_ollama():
    route = select_route(routed_settings())

    assert route.candidates == ("gemini", "groq", "ollama")


def test_dynamic_strict_workload_prefers_groq_then_gemini_then_ollama():
    route = select_route(routed_settings(llm_routing_strict_structured_output=True))

    assert route.candidates == ("groq", "gemini", "ollama")


def test_dynamic_evaluation_workload_prefers_groq_without_query_inspection():
    settings = routed_settings()

    with workload_context("evaluation"):
        route = select_route(settings)

    assert route.workload == "evaluation"
    assert route.candidates == ("groq", "gemini", "ollama")


def test_dynamic_local_only_uses_ollama_without_hosted_fallback():
    route = select_route(routed_settings(llm_routing_local_only=True))

    assert route.candidates == ("ollama",)
    assert route.can_fallback is False


def test_dynamic_route_skips_hosted_provider_without_a_key():
    route = select_route(routed_settings(google_api_key=None))

    assert route.candidates == ("groq", "ollama")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("deadline exceeded"), "timeout"),
        (RuntimeError("429 rate limit"), "rate_limited"),
        (ConnectionError("provider unavailable"), "provider_unavailable"),
        (MockGroqJsonValidateFailed("Failed to validate JSON"), "structured_output_failure"),
        (MockGroqArbitraryBadRequest("Bad request"), None),
    ],
)
def test_retryable_failures_are_classified_for_fallback(error, expected):
    assert is_retryable_provider_error(error) == expected


def test_dynamic_route_changes_once_after_a_retryable_failure():
    settings = routed_settings()

    with route_context(settings) as route:
        assert current_provider(settings) == "gemini"
        assert advance_route(TimeoutError("deadline exceeded")) is True
        assert current_provider(settings) == "groq"
        assert route.fallback_reasons == ["gemini:timeout"]


def test_static_route_does_not_automatically_fallback():
    settings = routed_settings(llm_routing_mode="static", llm_provider="gemini")

    with route_context(settings) as route:
        assert advance_route(TimeoutError("deadline exceeded")) is False
        assert route.provider == "gemini"


def test_static_groq_json_validate_failure_does_not_advance_route():
    settings = routed_settings(llm_routing_mode="static", llm_provider="groq")

    with route_context(settings) as route:
        assert advance_route(MockGroqJsonValidateFailed("Failed to validate JSON")) is False
        assert route.provider == "groq"
        assert route.fallback_reasons == []


def test_factory_failover_rebuilds_the_chain_and_consumes_a_second_budgeted_call(monkeypatch):
    """A provider switch is a new inference attempt, never a hidden retry."""
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    settings = routed_settings()
    attempts: list[str] = []

    def fake_structured_model(*_args, **_kwargs):
        provider = current_provider(settings)
        attempts.append(provider)
        if provider == "gemini":
            def fail(_value):
                raise TimeoutError("hosted timeout")

            return RunnableLambda(fail)
        return RunnableLambda(lambda _value: {"answer": "fallback response"})

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=2)
    with route_context(settings) as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        result = llm_factory.build_json_chain(
            RunnableLambda(lambda value: value),
            "generator",
            {"type": "object"},
        ).invoke({"question": "test"})

    assert result == {"answer": "fallback response"}
    assert attempts == ["gemini", "groq"]
    assert budget.llm_calls_used == 2
    assert route.fallback_reasons == ["gemini:timeout"]


def test_factory_falls_back_after_groq_json_validate_failure_with_second_permit(monkeypatch):
    """Only the exact nested Groq structured-output code advances the route."""
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    settings = routed_settings()
    attempts: list[str] = []

    def fake_structured_model(*_args, **_kwargs):
        provider = current_provider(settings)
        attempts.append(provider)
        if provider == "groq":
            def fail(_value):
                raise MockGroqJsonValidateFailed("Failed to validate JSON")

            return RunnableLambda(fail)
        return RunnableLambda(lambda _value: '{"answer": "fallback response"}')

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=2)
    with route_context(settings, "evaluation") as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        result = llm_factory.build_json_chain(
            RunnableLambda(lambda value: value), "generator", {"type": "object"}
        ).invoke({"question": "test"})

    assert result == {"answer": "fallback response"}
    assert attempts == ["groq", "gemini"]
    assert budget.llm_calls_used == 2
    assert route.fallback_reasons == ["groq:structured_output_failure"]


def test_arbitrary_groq_http_400_does_not_fallback(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    settings = routed_settings()
    attempts: list[str] = []

    def fake_structured_model(*_args, **_kwargs):
        attempts.append(current_provider(settings))

        def fail(_value):
            raise MockGroqArbitraryBadRequest("Bad request")

        return RunnableLambda(fail)

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=3)
    with route_context(settings, "evaluation") as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        with pytest.raises(MockGroqArbitraryBadRequest):
            llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ).invoke({"question": "test"})

    assert attempts == ["groq"]
    assert budget.llm_calls_used == 1
    assert route.fallback_reasons == []


def test_structured_output_failover_stops_after_the_final_provider(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    settings = routed_settings()
    attempts: list[str] = []

    def fake_structured_model(*_args, **_kwargs):
        attempts.append(current_provider(settings))

        def fail(_value):
            raise MockGroqJsonValidateFailed("Failed to validate JSON")

        return RunnableLambda(fail)

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=3)
    with route_context(settings, "evaluation") as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        with pytest.raises(MockGroqJsonValidateFailed):
            llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ).invoke({"question": "test"})

    assert attempts == ["groq", "gemini", "ollama"]
    assert budget.llm_calls_used == 3
    assert route.fallback_reasons == [
        "groq:structured_output_failure",
        "gemini:structured_output_failure",
    ]


def test_static_groq_structured_output_failure_becomes_a_closed_generation_failure(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory
    from src.generation.answer_chain import generate_grounded_answer
    from src.retrieval.types import RetrievedChunk

    settings = routed_settings(llm_routing_mode="static", llm_provider="groq")

    def fake_structured_model(*_args, **_kwargs):
        def fail(_value):
            raise MockGroqJsonValidateFailed("Failed to validate JSON")

        return RunnableLambda(fail)

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)
    evidence = [RetrievedChunk(1, "Policy text", "policy.txt", 0, doc_id="POL-001")]

    with route_context(settings) as route:
        response = generate_grounded_answer(
            "q",
            evidence,
            chain=llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ),
        )

    assert response.outcome == "provider_error"
    assert response.answer == ""
    assert route.fallback_reasons == []


def test_malformed_fallback_payload_is_rejected_not_accepted(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory
    from src.generation.answer_chain import generate_grounded_answer
    from src.retrieval.types import RetrievedChunk

    settings = routed_settings()

    def fake_structured_model(*_args, **_kwargs):
        if current_provider(settings) == "groq":
            def fail(_value):
                raise MockGroqJsonValidateFailed("Failed to validate JSON")

            return RunnableLambda(fail)
        return RunnableLambda(
            lambda _value: json.dumps(
                {
                    "answer": "An unsupported answer.",
                    "claim_citations": [],
                    "sufficient_context": True,
                    "confidence": 0.8,
                }
            )
        )

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)
    evidence = [RetrievedChunk(1, "Policy text", "policy.txt", 0, doc_id="POL-001")]
    budget = ExecutionBudget(timeout_s=60, max_llm_calls=2)
    with route_context(settings, "evaluation") as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        response = generate_grounded_answer(
            "q",
            evidence,
            chain=llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ),
        )

    assert response.outcome == "rejected_invalid_citation"
    assert response.answer == ""
    assert budget.llm_calls_used == 2
    assert route.fallback_reasons == ["groq:structured_output_failure"]
