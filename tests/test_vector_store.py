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
    # Checkout validation is idle-aware: `_check_if_idle` defers to the pool's
    # own check for any connection that may have gone stale.
    assert captured["check"] is vector_store._check_if_idle
    assert captured["reset"] is vector_store._mark_returned
    assert captured["timeout"] == 10.0
    assert captured["reconnect_timeout"] == 30.0
    assert captured["kwargs"]["connect_timeout"] == 10
    assert captured["kwargs"]["autocommit"] is True
    # Dead sockets must fail within the request budget, not hang for hours.
    assert captured["kwargs"]["keepalives"] == 1
    assert captured["kwargs"]["keepalives_idle"] <= 60


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
        @contextmanager
        def transaction(self):
            events.append(("application_transaction_begin", None))
            yield
            events.append(("application_transaction_commit", None))

        def execute(self, sql):
            events.append(("application_sql", sql))

    def connect(url, *, connect_timeout):
        events.append(("bootstrap_connect", (url, connect_timeout)))
        return BootstrapConnection()

    @contextmanager
    def application_connection():
        events.append(("application_connect", None))
        yield ApplicationConnection()

    class ApplicationPool:
        def wait(self, *, timeout):
            events.append(("pool_wait", timeout))

    monkeypatch.setattr(vector_store.psycopg, "connect", connect)
    monkeypatch.setattr(vector_store, "get_pool", lambda: ApplicationPool())
    monkeypatch.setattr(vector_store, "get_connection", application_connection)
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(
            database_url="postgresql://runtime.invalid/raguard",
            schema_database_url="postgresql://admin.invalid/raguard",
            db_connect_timeout_s=7,
            db_reconnect_timeout_s=30.0,
            vector_dimension=1024,
        ),
    )

    vector_store.init_schema()

    event_names = [name for name, _value in events]
    assert event_names.index("bootstrap_connect") < event_names.index("application_connect")
    assert events[event_names.index("pool_wait")] == ("pool_wait", 30.0)
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
        @contextmanager
        def transaction(self):
            events.append(("transaction_begin", None))
            yield
            events.append(("transaction_commit", None))

        def cursor(self):
            return Cursor()

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
    assert [name for name, _value in events] == [
        "transaction_begin",
        "delete",
        "upsert",
        "transaction_commit",
    ]


def test_upsert_updates_the_document_identifier(monkeypatch):
    captured: dict[str, object] = {}

    class Cursor:
        def executemany(self, sql, rows):
            captured["sql"] = sql
            captured["rows"] = rows

    written = vector_store._upsert_chunks(  # noqa: SLF001 - SQL contract test
        Cursor(),
        [
            {
                "source": "policy.txt",
                "doc_id": "NEW-002",
                "chunk_index": 0,
                "content": "updated",
                "metadata": {},
                "embedding": [0.1],
            }
        ],
    )

    assert written == 1
    assert "SET doc_id = EXCLUDED.doc_id" in str(captured["sql"])


# --------------------------------------------------------------------------
# Idle-aware checkout validation
# --------------------------------------------------------------------------


class _FakeConnection:
    """Weak-referenceable stand-in; the check never touches the socket."""


def _validation_counter(monkeypatch, window_s: float) -> list[object]:
    checked: list[object] = []

    class FakePool:
        @staticmethod
        def check_connection(connection):
            checked.append(connection)

    monkeypatch.setattr(vector_store, "ConnectionPool", FakePool)
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(db_checkout_validation_idle_s=window_s),
    )
    return checked


def test_a_never_returned_connection_is_validated(monkeypatch):
    checked = _validation_counter(monkeypatch, window_s=30.0)
    conn = _FakeConnection()

    vector_store._check_if_idle(conn)

    assert checked == [conn]


def test_a_connection_returned_moments_ago_skips_the_round_trip(monkeypatch):
    checked = _validation_counter(monkeypatch, window_s=30.0)
    conn = _FakeConnection()
    vector_store._mark_returned(conn)

    vector_store._check_if_idle(conn)

    assert checked == [], "a connection that just worked must not pay a validation RTT"


def test_an_idle_connection_is_validated_again(monkeypatch):
    """The case the check exists for: a serverless database closed it while idle."""
    checked = _validation_counter(monkeypatch, window_s=30.0)
    conn = _FakeConnection()
    vector_store._mark_returned(conn)
    vector_store._returned_at[conn] -= 31.0

    vector_store._check_if_idle(conn)

    assert checked == [conn]


def test_a_zero_window_restores_validation_on_every_checkout(monkeypatch):
    checked = _validation_counter(monkeypatch, window_s=0.0)
    conn = _FakeConnection()
    vector_store._mark_returned(conn)

    vector_store._check_if_idle(conn)

    assert checked == [conn]


# --------------------------------------------------------------------------
# A read that meets a connection the database closed while idle
# --------------------------------------------------------------------------


def _fake_pool_and_connections(monkeypatch, outcomes):
    """`outcomes` is consumed one per checkout: an exception to raise, or a value."""
    import psycopg

    events: list[str] = []

    class Pool:
        def check(self):
            events.append("pool_check")

    @contextmanager
    def connection():
        events.append("checkout")
        yield object()

    monkeypatch.setattr(vector_store, "get_pool", lambda: Pool())
    monkeypatch.setattr(vector_store, "get_connection", connection)

    remaining = list(outcomes)

    def statement(_conn):
        item = remaining.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return events, statement, psycopg


def test_a_stale_connection_is_retried_once_on_a_validated_pool(monkeypatch):
    import psycopg

    events, statement, _ = _fake_pool_and_connections(
        monkeypatch, [psycopg.OperationalError("server closed the connection"), ["row"]]
    )

    assert vector_store._read_retrying_stale_connection(statement) == ["row"]
    assert events == ["checkout", "checkout"]


def test_a_statement_timeout_is_not_retried(monkeypatch):
    """A slow query repeated is just a slower request."""
    import psycopg
    import pytest

    events, statement, _ = _fake_pool_and_connections(
        monkeypatch, [psycopg.errors.QueryCanceled("statement timeout")]
    )

    with pytest.raises(psycopg.errors.QueryCanceled):
        vector_store._read_retrying_stale_connection(statement)
    assert events == ["checkout"]


def test_a_second_failure_propagates(monkeypatch):
    import psycopg
    import pytest

    _events, statement, _ = _fake_pool_and_connections(
        monkeypatch,
        [psycopg.OperationalError("closed"), psycopg.OperationalError("still closed")],
    )

    with pytest.raises(psycopg.OperationalError):
        vector_store._read_retrying_stale_connection(statement)


def test_a_pool_timeout_is_not_retried(monkeypatch):
    """No connection at all is an outage; a retry only doubles the wait."""
    import pytest
    from psycopg_pool import PoolTimeout

    events, statement, _ = _fake_pool_and_connections(monkeypatch, [PoolTimeout("no connection")])

    with pytest.raises(PoolTimeout):
        vector_store._read_retrying_stale_connection(statement)
    assert events == ["checkout"]


def test_pgvector_vector_objects_convert_to_float32_arrays():
    """The adapter yields `Vector`, not ndarray; the live fetch once crashed on it."""
    import numpy as np
    from pgvector import Vector

    array = vector_store._as_float32(Vector([0.25, -1.0, 2.0]))

    assert array.dtype == np.float32
    assert array.tolist() == [0.25, -1.0, 2.0]
    assert vector_store._as_float32([1, 2]).tolist() == [1.0, 2.0]
