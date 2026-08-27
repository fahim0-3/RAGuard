"""Focused tests for the pgvector connection boundary."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from src.config import Settings
from src.retrieval import vector_store


def test_schema_database_url_defaults_to_the_runtime_connection():
    settings = Settings(
        _env_file=None,
        database_url="postgresql://runtime.invalid/raguard",
        database_admin_url="",
    )

    assert settings.schema_database_url == "postgresql://runtime.invalid/raguard"


def test_schema_database_url_prefers_the_explicit_admin_connection():
    settings = Settings(
        _env_file=None,
        database_url="postgresql://runtime.invalid/raguard",
        database_admin_url="postgresql://admin.invalid/raguard",
    )

    assert settings.schema_database_url == "postgresql://admin.invalid/raguard"


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


def test_init_schema_bootstraps_vector_before_opening_the_vector_pool(monkeypatch):
    """A fresh database cannot register the vector type before it exists."""
    events: list[tuple[str, object]] = []

    class BootstrapConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql):
            events.append(("bootstrap_sql", sql))

        def commit(self):
            events.append(("bootstrap_commit", None))

    class ApplicationConnection:
        def execute(self, sql):
            events.append(("application_sql", sql))

        def commit(self):
            events.append(("application_commit", None))

    def connect(url, *, connect_timeout):
        events.append(("bootstrap_connect", (url, connect_timeout)))
        return BootstrapConnection()

    @contextmanager
    def application_connection():
        events.append(("application_connect", None))
        yield ApplicationConnection()

    monkeypatch.setattr(vector_store.psycopg, "connect", connect)
    monkeypatch.setattr(vector_store, "get_connection", application_connection)
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql://runtime.invalid/raguard",
            schema_database_url="postgresql://admin.invalid/raguard",
            db_connect_timeout_s=7,
            vector_dimension=1024,
        ),
    )

    vector_store.init_schema()

    event_names = [name for name, _value in events]
    assert event_names.index("bootstrap_connect") < event_names.index(
        "application_connect"
    )
    assert events[0] == (
        "bootstrap_connect",
        ("postgresql://admin.invalid/raguard", 7),
    )
    bootstrap_sql = next(value for name, value in events if name == "bootstrap_sql")
    application_sql = next(value for name, value in events if name == "application_sql")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in str(bootstrap_sql)
    assert "CREATE EXTENSION" not in str(application_sql)


def test_replace_source_chunks_deletes_and_upserts_before_one_commit(monkeypatch):
    events: list[tuple[str, object]] = []

    class Cursor:
        rowcount = 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params):
            events.append(("delete", (sql, params)))

        def executemany(self, sql, rows):
            events.append(("upsert", (sql, rows)))

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            events.append(("commit", None))

    @contextmanager
    def connection():
        yield Connection()

    monkeypatch.setattr(vector_store, "get_connection", connection)
    records = [
        {
            "source": "policy.txt",
            "doc_id": "POL-001",
            "chunk_index": 0,
            "content": "Policy text",
            "metadata": {},
            "embedding": [0.1, 0.2],
        }
    ]

    removed, written = vector_store.replace_source_chunks("policy.txt", records)

    assert (removed, written) == (3, 1)
    assert [name for name, _value in events] == ["delete", "upsert", "commit"]
