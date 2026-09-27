"""Shared fixtures.

Tests are layered by cost:

- unmarked           pure logic, no models, no database, no network
- @pytest.mark.heavy loads BGE-M3 or the reranker
- @pytest.mark.integration requires a running pgvector instance with the corpus
- @pytest.mark.llm    consumes provider quota

Run the fast tier with:  pytest -m "not heavy and not integration and not llm and not evaluation"
"""

from __future__ import annotations

import os

import pytest

from src.evaluation.metrics import load_golden_dataset
from src.retrieval.types import RetrievedChunk


@pytest.fixture(scope="session", autouse=True)
def _ignore_developer_env_file():
    """Assert against declared defaults, never a developer's local `.env`.

    `Settings` reads `PROJECT_ROOT/.env` so the application picks up local
    configuration. That file is machine-specific: a value such as
    `GRAPH_MAX_RETRIES=0`, set while debugging latency, silently turns the
    retry-loop tests red on one machine and green in CI. Tests describe the
    committed defaults, so the file is ignored for the session. Real
    environment variables still apply, and `RAGUARD_TESTS_USE_ENV_FILE=1`
    restores the file for deliberate configuration checks.
    """
    from src.config.settings import Settings, get_settings

    if os.getenv("RAGUARD_TESTS_USE_ENV_FILE") == "1":
        yield
        return

    original = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_provider_health():
    """Process-wide provider state must not leak between tests: one test's
    failure must not bench a provider, nor its token spend exhaust a budget,
    for the next."""
    from src.generation.llm_routing import PROVIDER_HEALTH
    from src.generation.rate_limit import reset_budgets

    PROVIDER_HEALTH.reset()
    reset_budgets()
    yield
    PROVIDER_HEALTH.reset()
    reset_budgets()


def pytest_collection_modifyitems(items):
    """Require a second, explicit opt-in before any real model test can run."""
    if os.getenv("RAGUARD_ALLOW_HEAVY_TESTS") == "1":
        return
    blocked = pytest.mark.skip(
        reason="set RAGUARD_ALLOW_HEAVY_TESTS=1 to permit transformer downloads"
    )
    for item in items:
        if item.get_closest_marker("heavy") is not None:
            item.add_marker(blocked)


@pytest.fixture(scope="session")
def golden_cases() -> list[dict]:
    return load_golden_dataset()


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    """Deterministic stand-in for reranked retrieval output."""
    return [
        RetrievedChunk(
            chunk_id=1,
            content=(
                "[Refund Policy > Refund processing times]\n"
                "Credit and debit cards: 5 to 7 business days. The processing clock "
                "starts when the returned item is scanned at the warehouse."
            ),
            source="refund_policy.txt",
            chunk_index=2,
            normalised_rerank_score=0.91,
            rerank_score=2.3,
        ),
        RetrievedChunk(
            chunk_id=2,
            content=(
                "[Payment Failure FAQ > Gateway error codes]\n"
                "PAY-402 Insufficient funds. Retry with another card or top up the account."
            ),
            source="payment_failure_faq.txt",
            chunk_index=1,
            normalised_rerank_score=0.64,
            rerank_score=0.6,
        ),
        RetrievedChunk(
            chunk_id=3,
            content=(
                "[Delivery Policy > Failed delivery attempts]\n"
                "Carriers make two delivery attempts before returning the parcel."
            ),
            source="delivery_policy.txt",
            chunk_index=5,
            normalised_rerank_score=0.21,
            rerank_score=-1.3,
        ),
    ]


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Prevent settings cached in one test from leaking into the next."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
