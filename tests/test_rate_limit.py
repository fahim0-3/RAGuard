"""Token budgets: route away from a provider before it refuses the request."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.generation.rate_limit import (
    BudgetLimits,
    TokenBudget,
    budget_status,
    record_usage,
    reset_budgets,
)


@pytest.fixture(autouse=True)
def _clean_budgets():
    reset_budgets()
    yield
    reset_budgets()


def budget(per_minute=8_000, per_day=200_000, now=None):
    clock = now or [1_000.0]
    return TokenBudget(BudgetLimits(per_minute, per_day), clock=lambda: clock[0]), clock


# --------------------------------------------------------------------------
# The tally itself
# --------------------------------------------------------------------------


def test_a_call_that_fits_is_allowed():
    tally, _now = budget()
    tally.record(3_000)

    assert tally.exceeded_by(2_000) is None


def test_the_call_that_would_exceed_the_minute_is_refused():
    """The last request that fits is served; the next is routed elsewhere."""
    tally, _now = budget()
    tally.record(7_000)

    assert tally.exceeded_by(2_000) == "token_budget_minute"


def test_the_minute_window_rolls_forward():
    tally, now = budget()
    tally.record(8_000)
    assert tally.exceeded_by(1_000) == "token_budget_minute"

    now[0] += 61.0

    assert tally.exceeded_by(1_000) is None


def test_the_daily_window_outlasts_the_minute_window():
    tally, now = budget(per_minute=8_000, per_day=10_000)
    tally.record(8_000)
    now[0] += 61.0

    assert tally.exceeded_by(4_000) == "token_budget_day", "a minute passed, the day did not"


def test_the_daily_window_rolls_forward():
    tally, now = budget(per_minute=0, per_day=10_000)
    tally.record(10_000)
    assert tally.exceeded_by(1) == "token_budget_day"

    now[0] += 86_401.0

    assert tally.exceeded_by(1) is None


def test_an_unknown_limit_never_blocks():
    tally, _now = budget(per_minute=0, per_day=0)
    tally.record(10_000_000)

    assert tally.exceeded_by(10_000) is None


def test_the_snapshot_reports_headroom_without_content():
    tally, _now = budget()
    tally.record(1_500)

    snapshot = tally.snapshot()

    assert snapshot["tokens_used_last_minute"] == 1_500
    assert snapshot["tokens_remaining_this_minute"] == 6_500
    assert snapshot["tokens_remaining_today"] == 198_500


# --------------------------------------------------------------------------
# Routing consults the budget
# --------------------------------------------------------------------------


def settings_with(**overrides) -> Settings:
    base = {
        "_env_file": None,
        "llm_routing_mode": "dynamic",
        "google_api_key": "g" * 32,
        "groq_api_key": "r" * 32,
        "openrouter_api_key": "o" * 32,
    }
    base.update(overrides)
    return Settings(**base)


def test_groq_is_skipped_once_its_minute_budget_is_spent():
    settings = settings_with(groq_tokens_per_minute=8_000)
    assert budget_status("groq", settings, 2_000) is None

    record_usage("groq", 7_500)

    assert budget_status("groq", settings, 2_000) == "token_budget_minute"


def test_a_provider_without_published_limits_is_never_skipped():
    settings = settings_with()
    record_usage("gemini", 1_000_000)

    assert budget_status("gemini", settings, 2_000) is None


def test_the_budget_can_be_disabled():
    settings = settings_with(llm_token_budget_enabled=False, groq_tokens_per_minute=10)
    record_usage("groq", 1_000)

    assert budget_status("groq", settings, 2_000) is None


def test_an_exhausted_groq_budget_routes_the_request_to_gemini():
    """The behaviour this exists for: a question still gets answered."""
    from src.generation.llm_routing import select_route

    settings = settings_with(groq_tokens_per_minute=8_000)
    record_usage("groq", 7_999)

    route = select_route(settings)

    assert route.provider == "gemini"
    assert route.skipped == ["groq:token_budget_minute"]
    assert route.fallback_reasons == [], "a skip is not a failed attempt"


def test_groq_returns_once_the_minute_has_passed(monkeypatch):
    from src.generation import rate_limit
    from src.generation.llm_routing import select_route

    now = [1_000.0]
    settings = settings_with(groq_tokens_per_minute=8_000)
    budget_status("groq", settings, 1)  # builds the registry
    monkeypatch.setattr(rate_limit.PROVIDER_BUDGETS["groq"], "_clock", lambda: now[0])
    record_usage("groq", 8_000)
    assert select_route(settings).provider == "gemini"

    now[0] += 61.0

    assert select_route(settings).provider == "groq"
