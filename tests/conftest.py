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
