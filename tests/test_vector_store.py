"""Focused tests for the pgvector connection boundary."""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval import vector_store


def test_pool_checks_managed_database_connections_before_checkout(monkeypatch):
    """A provider-closed idle connection must not reach application SQL."""
    captured: dict[str, object] = {}

    class FakePool:
        @staticmethod
        def check_connection(connection):
            return connection

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vector_store, "ConnectionPool", FakePool)
    monkeypatch.setattr(vector_store, "_pool", None)
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql://example.invalid/raguard",
            db_pool_timeout_s=10.0,
            db_connect_timeout_s=10,
            db_reconnect_timeout_s=30.0,
        ),
    )

    pool = vector_store.get_pool()

    assert isinstance(pool, FakePool)
    assert captured["check"] is FakePool.check_connection
    assert captured["timeout"] == 10.0
    assert captured["reconnect_timeout"] == 30.0
    assert captured["kwargs"] == {"connect_timeout": 10}
