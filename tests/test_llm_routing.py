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


class MockProviderAuthenticationError(RuntimeError):
    status_code = 401


def routed_settings(**overrides: object) -> Settings:
    """Use explicit credentials so tests never depend on a developer `.env`."""
    base = Settings(
        _env_file=None,
        llm_routing_mode="dynamic",
        google_api_key="g" * 32,
        groq_api_key="r" * 32,
        openrouter_api_key="o" * 32,
    )
    return base.model_copy(update=overrides)


def test_static_mode_preserves_manual_llm_provider_selection():
    settings = routed_settings(llm_routing_mode="static", llm_provider="groq")

    route = select_route(settings)

    assert route.provider == "groq"
    assert route.candidates == ("groq",)
    assert route.can_fallback is False


def test_dynamic_normal_rag_prefers_groq_then_gemini_then_openrouter():
    """Serving never falls into CPU Ollama unless a deployment asks for it."""
    route = select_route(routed_settings())

    assert route.candidates == ("groq", "gemini", "openrouter")


def test_an_explicit_local_fallback_puts_ollama_last():
    route = select_route(routed_settings(llm_routing_local_fallback=True))

    assert route.candidates == ("groq", "gemini", "openrouter", "ollama")


def test_a_deployment_without_hosted_credentials_routes_to_ollama():
    """No hosted key at all is local operation by construction."""
    route = select_route(
        routed_settings(google_api_key=None, groq_api_key=None, openrouter_api_key=None)
    )

    assert route.candidates == ("ollama",)


def test_dynamic_strict_workload_prefers_groq_then_gemini_then_openrouter():
    route = select_route(routed_settings(llm_routing_strict_structured_output=True))

    assert route.candidates == ("groq", "gemini", "openrouter")


def test_dynamic_evaluation_workload_prefers_groq_without_query_inspection():
    settings = routed_settings()

    with workload_context("evaluation"):
        route = select_route(settings)

    assert route.workload == "evaluation"
    assert route.candidates == ("groq", "gemini", "openrouter")


def test_dynamic_local_only_uses_ollama_without_hosted_fallback():
    route = select_route(routed_settings(llm_routing_local_only=True))

    assert route.candidates == ("ollama",)
    assert route.can_fallback is False


def test_dynamic_route_skips_hosted_provider_without_a_key():
    route = select_route(routed_settings(google_api_key=None))

    assert route.candidates == ("groq", "openrouter")


def test_dynamic_route_skips_openrouter_without_a_key():
    route = select_route(routed_settings(openrouter_api_key=None))

    assert route.candidates == ("groq", "gemini")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("deadline exceeded"), "timeout"),
        (RuntimeError("429 rate limit"), "rate_limited"),
        (ConnectionError("provider unavailable"), "provider_unavailable"),
        (MockGroqJsonValidateFailed("Failed to validate JSON"), "structured_output_failure"),
        (MockGroqArbitraryBadRequest("Bad request"), None),
        (MockProviderAuthenticationError("invalid API key"), "unauthorized"),
    ],
)
def test_retryable_failures_are_classified_for_fallback(error, expected):
    assert is_retryable_provider_error(error) == expected


def test_dynamic_route_changes_once_after_a_retryable_failure():
    settings = routed_settings()

    with route_context(settings) as route:
        assert current_provider(settings) == "groq"
        assert advance_route(TimeoutError("deadline exceeded")) is True
        assert current_provider(settings) == "gemini"
        assert route.fallback_reasons == ["groq:timeout"]


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
        if provider == "groq":

            def fail(_value):
                raise TimeoutError("hosted timeout")

            return RunnableLambda(fail)
        return RunnableLambda(lambda _value: '{"answer": "fallback response"}')

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
    assert attempts == ["groq", "gemini"]
    assert budget.llm_calls_used == 2
    assert route.fallback_reasons == ["groq:timeout"]


def test_dynamic_route_can_advance_from_groq_to_openrouter():
    settings = routed_settings(google_api_key=None)

    with route_context(settings, "evaluation") as route:
        assert route.provider == "groq"
        assert advance_route(TimeoutError("deadline exceeded")) is True
        assert route.provider == "openrouter"


def test_dynamic_route_can_advance_from_openrouter_to_ollama_when_allowed():
    settings = routed_settings(
        google_api_key=None, groq_api_key=None, llm_routing_local_fallback=True
    )

    with route_context(settings) as route:
        assert route.provider == "openrouter"
        assert advance_route(TimeoutError("deadline exceeded")) is True
        assert route.provider == "ollama"


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

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=4)
    with route_context(settings, "evaluation") as route, request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        with pytest.raises(MockGroqJsonValidateFailed):
            llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ).invoke({"question": "test"})

    assert attempts == ["groq", "gemini", "openrouter"], "the hosted route ends at OpenRouter"
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


# --------------------------------------------------------------------------
# Circuit breaker: recently failed providers are skipped, not retried
# --------------------------------------------------------------------------


class MockRateLimited(RuntimeError):
    status_code = 429


class MockServiceUnavailable(RuntimeError):
    """A 503 whose message says nothing useful; the status must classify it."""

    status_code = 503


def _fake_clock(monkeypatch, start: float = 1_000.0) -> list[float]:
    from src.generation.llm_routing import PROVIDER_HEALTH

    now = [start]
    monkeypatch.setattr(PROVIDER_HEALTH, "_clock", lambda: now[0])
    return now


def test_a_5xx_is_classified_by_status_not_by_message_wording():
    assert is_retryable_provider_error(MockServiceUnavailable("oops")) == "provider_unavailable"


def test_a_rate_limited_provider_is_skipped_by_the_next_request(monkeypatch):
    _fake_clock(monkeypatch)
    settings = routed_settings()

    with route_context(settings):
        assert advance_route(MockRateLimited("slow down")) is True

    next_route = select_route(settings)

    assert next_route.provider == "gemini", "the next request must not re-pay Groq's 429"
    assert next_route.skipped == ["groq:rate_limited"]
    assert next_route.fallback_reasons == [], "a skip is not a failed attempt"


def test_a_provider_returns_once_its_cooldown_expires(monkeypatch):
    now = _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings):
        advance_route(MockRateLimited("slow down"))

    now[0] += 601.0

    assert select_route(settings).provider == "groq"


def test_a_structured_output_failure_does_not_bench_the_provider(monkeypatch):
    """Malformed JSON depends on the prompt, not on provider health."""
    _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings):
        advance_route(MockGroqJsonValidateFailed("Failed to validate JSON"))

    route = select_route(settings)

    assert route.provider == "groq"
    assert route.skipped == []


def test_an_unauthorized_provider_is_benched_for_longer_than_a_rate_limit(monkeypatch):
    now = _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings):
        advance_route(MockProviderAuthenticationError("invalid API key"))

    now[0] += 120.0

    assert select_route(settings).provider == "gemini", "a bad key does not fix itself"


def test_every_provider_cooling_probes_only_the_one_due_back_soonest(monkeypatch):
    """An outage costs one attempt per request, not a walk through every failure."""
    from src.generation.llm_routing import PROVIDER_HEALTH

    now = _fake_clock(monkeypatch)
    settings = routed_settings()
    PROVIDER_HEALTH.record_failure("groq", "unauthorized")  # 600 s
    PROVIDER_HEALTH.record_failure("gemini", "provider_unavailable")  # 120 s
    now[0] += 30.0
    PROVIDER_HEALTH.record_failure("openrouter", "timeout")  # 30 s, from here

    route = select_route(settings)

    assert route.candidates == ("openrouter",)
    assert route.can_fallback is False
    assert route.skipped == ["groq:unauthorized", "gemini:provider_unavailable"]


def test_evaluation_routes_ignore_the_breaker(monkeypatch):
    """Evaluation results are only comparable across runs on a fixed route."""
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    PROVIDER_HEALTH.record_failure("groq", "rate_limited")

    assert select_route(routed_settings(), "evaluation").provider == "groq"


def test_the_breaker_can_be_disabled(monkeypatch):
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    PROVIDER_HEALTH.record_failure("groq", "rate_limited")

    route = select_route(routed_settings(llm_provider_cooldown_enabled=False))

    assert route.provider == "groq"


def test_a_mid_request_failover_skips_a_provider_benched_by_another_request(monkeypatch):
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings) as route:
        # A concurrent request benches Gemini after this route was chosen.
        PROVIDER_HEALTH.record_failure("gemini", "provider_unavailable")
        assert advance_route(TimeoutError("deadline exceeded")) is True

        assert route.provider == "openrouter"
        assert route.skipped == ["gemini:provider_unavailable"]


def test_the_final_candidate_is_attempted_even_while_cooling(monkeypatch):
    """A last attempt beats refusing outright; the skip loop stops before it."""
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    settings = routed_settings(openrouter_api_key=None)
    with route_context(settings) as route:
        PROVIDER_HEALTH.record_failure("gemini", "provider_unavailable")
        advance_route(TimeoutError("deadline exceeded"))

        assert route.provider == "gemini"


def test_a_successful_call_clears_the_providers_cooldown(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    settings = routed_settings()
    PROVIDER_HEALTH.record_failure("groq", "rate_limited")
    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(
        llm_factory,
        "get_structured_chat_model",
        lambda *_a, **_k: RunnableLambda(lambda _v: '{"ok": true}'),
    )

    with route_context(settings, "evaluation"):  # evaluation: Groq still selected
        llm_factory.build_json_chain(
            RunnableLambda(lambda value: value), "generator", {"type": "object"}
        ).invoke({"question": "q"})

    assert PROVIDER_HEALTH.cooling("groq") is None


def test_the_second_request_no_longer_pays_for_the_failed_provider(monkeypatch):
    """The measured cascade, end to end: one 429 costs one request, not every one."""
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    _fake_clock(monkeypatch)
    settings = routed_settings()
    attempts: list[str] = []

    def fake_structured_model(*_args, **_kwargs):
        provider = current_provider(settings)
        attempts.append(provider)
        if provider == "groq":

            def fail(_value):
                raise MockRateLimited("rate limit")

            return RunnableLambda(fail)
        return RunnableLambda(lambda _value: '{"answer": "ok"}')

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    for _ in range(2):
        budget = ExecutionBudget(timeout_s=60, max_llm_calls=4)
        with route_context(settings), request_budget(budget):
            reserve_llm_call("generate_answer", default_timeout_s=60)
            llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "generator", {"type": "object"}
            ).invoke({"question": "q"})

    assert attempts == ["groq", "gemini", "gemini"]


class _Headers(dict):
    """Case-insensitive enough for the one header the breaker reads."""


class MockDailyQuotaExhausted(RuntimeError):
    """Groq's per-day token quota: a 429 whose retry hint is minutes, not seconds."""

    status_code = 429

    def __init__(self, retry_after: str) -> None:
        super().__init__("rate limit")
        self.response = type("Response", (), {"headers": _Headers({"retry-after": retry_after})})()


def test_a_providers_retry_after_hint_extends_the_cooldown(monkeypatch):
    now = _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings):
        advance_route(MockDailyQuotaExhausted("276"))

    now[0] += 120.0  # well past the 60 s rate-limit floor

    assert select_route(settings).provider == "gemini", "the provider said four minutes"


def test_a_retry_after_hint_is_capped(monkeypatch):
    from src.generation.llm_routing import PROVIDER_HEALTH

    now = _fake_clock(monkeypatch)
    PROVIDER_HEALTH.record_failure("groq", "rate_limited", retry_after_s=86_400.0)

    now[0] += 1_801.0

    assert PROVIDER_HEALTH.cooling("groq") is None


def test_a_short_retry_after_hint_never_shortens_the_floor(monkeypatch):
    from src.generation.llm_routing import PROVIDER_HEALTH

    now = _fake_clock(monkeypatch)
    PROVIDER_HEALTH.record_failure("groq", "rate_limited", retry_after_s=1.0)

    now[0] += 30.0

    assert PROVIDER_HEALTH.cooling("groq") == "rate_limited"


def test_a_malformed_retry_after_header_is_ignored():
    from src.generation.llm_routing import retry_after_seconds

    assert retry_after_seconds(MockDailyQuotaExhausted("soon")) is None
    assert retry_after_seconds(RuntimeError("no response attached")) is None


def test_a_disabled_breaker_neither_records_nor_skips_mid_request(monkeypatch):
    """Off means off: the flag must govern the failover path, not only selection."""
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    PROVIDER_HEALTH.record_failure("gemini", "provider_unavailable")
    settings = routed_settings(llm_provider_cooldown_enabled=False)

    with route_context(settings) as route:
        advance_route(MockRateLimited("slow down"))

        assert route.provider == "gemini", "a disabled breaker must not skip Gemini"
        assert route.skipped == []
    assert PROVIDER_HEALTH.cooling("groq") is None, "nor record Groq's failure"


def test_a_concurrent_failure_is_charged_to_the_provider_that_failed(monkeypatch):
    """Grading and the speculative draft both fail on Groq; Gemini must stay clean."""
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    settings = routed_settings()
    with route_context(settings) as route:
        # First call (grader) on Groq fails and moves the route to Gemini.
        assert advance_route(MockProviderAuthenticationError("bad key"), failed_provider="groq")
        assert route.provider == "gemini"
        # Second call (speculative draft), also started on Groq, fails a moment later.
        assert advance_route(MockProviderAuthenticationError("bad key"), failed_provider="groq")

        assert route.provider == "gemini", "no second advance for the same failure"
        assert route.fallback_reasons == ["groq:unauthorized"]
    assert PROVIDER_HEALTH.cooling("gemini") is None, "Gemini never failed"
    assert PROVIDER_HEALTH.cooling("groq") == "unauthorized"


def test_success_is_credited_to_the_provider_that_answered(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory
    from src.generation.llm_routing import PROVIDER_HEALTH

    _fake_clock(monkeypatch)
    settings = routed_settings()
    PROVIDER_HEALTH.record_failure("gemini", "provider_unavailable")

    def fake_structured_model(*_args, **_kwargs):
        def answer(_value):
            # A concurrent call moves the route while this Groq call is in flight.
            from src.generation.llm_routing import current_route

            current_route().index = 1
            return '{"ok": true}'

        return RunnableLambda(answer)

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)

    with route_context(settings):
        llm_factory.build_json_chain(
            RunnableLambda(lambda value: value), "generator", {"type": "object"}
        ).invoke({"q": 1})

    assert PROVIDER_HEALTH.cooling("gemini") == "provider_unavailable", "Groq answered, not Gemini"


# --------------------------------------------------------------------------
# Hosted outage: a fast controlled failure, never a CPU Ollama detour
# --------------------------------------------------------------------------


def _failing_hosted_factory(monkeypatch, settings, attempts: list[str]):
    from langchain_core.runnables import RunnableLambda

    from src.generation import llm_factory

    def fake_structured_model(*_args, **_kwargs):
        provider = current_provider(settings)
        attempts.append(provider)
        if provider == "ollama":
            return RunnableLambda(lambda _value: '{"answer": "local"}')

        def fail(_value):
            raise MockServiceUnavailable("unavailable")

        return RunnableLambda(fail)

    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_factory, "get_structured_chat_model", fake_structured_model)
    return llm_factory


def test_every_hosted_provider_failing_raises_without_reaching_ollama(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    _fake_clock(monkeypatch)
    settings = routed_settings()
    attempts: list[str] = []
    llm_factory = _failing_hosted_factory(monkeypatch, settings, attempts)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=8)
    with route_context(settings), request_budget(budget):
        reserve_llm_call("evidence_grader", default_timeout_s=60)
        with pytest.raises(MockServiceUnavailable):
            llm_factory.build_json_chain(
                RunnableLambda(lambda value: value), "judge", {"type": "object"}
            ).invoke({"q": 1})

    assert attempts == ["groq", "gemini", "openrouter"]
    assert "ollama" not in attempts


def test_the_next_request_during_an_outage_makes_a_single_probe(monkeypatch):
    """Each request after the first costs one attempt, not the whole route."""
    from langchain_core.runnables import RunnableLambda

    _fake_clock(monkeypatch)
    settings = routed_settings()
    attempts: list[str] = []
    llm_factory = _failing_hosted_factory(monkeypatch, settings, attempts)

    for _ in range(2):
        budget = ExecutionBudget(timeout_s=60, max_llm_calls=8)
        with route_context(settings), request_budget(budget):
            reserve_llm_call("evidence_grader", default_timeout_s=60)
            with pytest.raises(MockServiceUnavailable):
                llm_factory.build_json_chain(
                    RunnableLambda(lambda value: value), "judge", {"type": "object"}
                ).invoke({"q": 1})

    assert attempts == ["groq", "gemini", "openrouter", "groq"]


def test_explicit_local_mode_still_answers_through_ollama(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    settings = routed_settings(llm_routing_local_only=True)
    attempts: list[str] = []
    llm_factory = _failing_hosted_factory(monkeypatch, settings, attempts)

    with route_context(settings):
        result = llm_factory.build_json_chain(
            RunnableLambda(lambda value: value), "generator", {"type": "object"}
        ).invoke({"q": 1})

    assert attempts == ["ollama"]
    assert result == {"answer": "local"}


def test_an_explicit_local_fallback_reaches_ollama_after_hosted_failures(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    _fake_clock(monkeypatch)
    settings = routed_settings(llm_routing_local_fallback=True)
    attempts: list[str] = []
    llm_factory = _failing_hosted_factory(monkeypatch, settings, attempts)

    budget = ExecutionBudget(timeout_s=60, max_llm_calls=8)
    with route_context(settings), request_budget(budget):
        reserve_llm_call("generate_answer", default_timeout_s=60)
        result = llm_factory.build_json_chain(
            RunnableLambda(lambda value: value), "generator", {"type": "object"}
        ).invoke({"q": 1})

    assert attempts == ["groq", "gemini", "openrouter", "ollama"]
    assert result == {"answer": "local"}


def test_a_rejected_gemini_key_is_unauthorized_despite_http_400():
    """Google answers a bad key with 400 INVALID_ARGUMENT, not 401."""

    class GoogleBadKey(RuntimeError):
        code = 400

    error = GoogleBadKey("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.")

    assert is_retryable_provider_error(error) == "unauthorized"
