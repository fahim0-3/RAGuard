"""Embedding-model initialisation and the readiness contract it feeds.

The defect these cover: the model was loaded lazily inside the first user
request, so `/query` blocked for the whole download while `/health`, `/ready`,
and the Docker health check all reported the service fine. The loader itself was
already correct — process-wide and lock-guarded — so these tests pin that
property down rather than change it, and add the readiness gate that was
missing.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from api.main import app, graph_dependency
from src.retrieval import embeddings


@pytest.fixture
def clean_model_cache(monkeypatch):
    """Reset the module-level cache so each test starts cold.

    Touches the private globals deliberately: the whole point is to exercise the
    caching, which cannot be observed through the public surface alone.
    """
    monkeypatch.setattr(embeddings, "_model", None, raising=True)
    monkeypatch.setattr(embeddings, "_load_error", None, raising=True)
    yield
    monkeypatch.setattr(embeddings, "_model", None, raising=True)


class CountingModel:
    """Stand-in for SentenceTransformer that records construction."""

    instances = 0
    delay = 0.0

    def __init__(self, name, device=None, **kwargs):
        type(self).instances += 1
        self.name = name
        self.device = device
        # A slow constructor makes a race observable; an instant one would let a
        # broken lock pass by luck.
        time.sleep(type(self).delay)

    def get_sentence_embedding_dimension(self):
        return 1024


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Provide the patched dependency without installing the model stack.

    The fast CI tier intentionally omits Torch and sentence-transformers. The
    loader imports SentenceTransformer lazily, so these tests can exercise its
    singleton and failure behavior with a lightweight module substitute.
    """
    module = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def counting_model(fake_sentence_transformers):
    CountingModel.instances = 0
    CountingModel.delay = 0.0

    fake_sentence_transformers.SentenceTransformer = CountingModel
    return CountingModel


# --------------------------------------------------------------------------
# Loaded once, per process
# --------------------------------------------------------------------------


def test_model_is_constructed_only_once(clean_model_cache, counting_model):
    embeddings.get_embedding_model()
    embeddings.get_embedding_model()
    embeddings.get_embedding_model()

    assert counting_model.instances == 1


def test_repeated_warmup_does_not_reload(clean_model_cache, counting_model):
    assert embeddings.warmup_embedding_model() is True
    assert embeddings.warmup_embedding_model() is True

    assert counting_model.instances == 1


def test_concurrent_callers_share_one_construction(clean_model_cache, counting_model):
    """Twenty threads must not start twenty downloads."""
    counting_model.delay = 0.15
    results: list[object] = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        results.append(embeddings.get_embedding_model())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert counting_model.instances == 1, "the double-checked lock let a race through"
    assert len(results) == 20
    assert len({id(model) for model in results}) == 1, "all callers share one instance"


def test_probe_does_not_trigger_a_load(clean_model_cache, counting_model):
    """`is_model_loaded()` is a readiness probe, not an accessor."""
    assert embeddings.is_model_loaded() is False
    assert counting_model.instances == 0


def test_probe_is_true_after_warmup(clean_model_cache, counting_model):
    embeddings.warmup_embedding_model()

    assert embeddings.is_model_loaded() is True


# --------------------------------------------------------------------------
# Failure is recorded, not swallowed
# --------------------------------------------------------------------------


def test_warmup_reports_failure_without_raising(
    clean_model_cache, monkeypatch, fake_sentence_transformers
):

    def explode(*args, **kwargs):
        raise OSError("connection reset while downloading")

    fake_sentence_transformers.SentenceTransformer = explode

    assert embeddings.warmup_embedding_model() is False
    assert embeddings.is_model_loaded() is False
    assert "OSError" in (embeddings.model_load_error() or "")


def test_direct_access_still_raises_for_callers_that_can_handle_it(
    clean_model_cache, monkeypatch, fake_sentence_transformers
):
    """`warmup` swallows; `get_embedding_model` must not."""

    def explode(*args, **kwargs):
        raise OSError("no space left on device")

    fake_sentence_transformers.SentenceTransformer = explode

    with pytest.raises(OSError):
        embeddings.get_embedding_model()


# --------------------------------------------------------------------------
# Health versus readiness
# --------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """No lifespan: these tests drive the endpoints, not start-up."""
    monkeypatch.setattr("api.main.is_reranker_model_loaded", lambda: True)
    monkeypatch.setattr("api.main.loaded_reranker_model_name", lambda: "test-reranker")
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def test_health_is_ok_before_the_model_is_loaded(client, monkeypatch):
    """Liveness must not wait on a 2.2 GB download."""
    monkeypatch.setattr("api.main.is_model_loaded", lambda: False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_is_503_while_the_model_is_loading(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: False)
    monkeypatch.setattr("api.main.model_load_error", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["embedding_model"]["status"] == "loading"
    assert "still loading" in body["detail"]


def test_ready_is_503_and_says_failed_when_loading_broke(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: False)
    monkeypatch.setattr("api.main.model_load_error", lambda: "OSError: connection reset by peer")

    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["checks"]["embedding_model"]["status"] == "failed"
    assert "failed to load" in body["detail"]
    # Type name only: the full message can carry a cache path.
    assert body["checks"]["embedding_model"]["error"] == "OSError"
    assert "peer" not in response.text


def test_ready_is_200_once_the_model_is_loaded(client, monkeypatch):
    from src.config import Settings, get_settings

    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: True)
    stub = Settings(_env_file=None).model_copy(update={"llm_provider": "ollama"})
    app.dependency_overrides[get_settings] = lambda: stub

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["embedding_model"]["status"] == "loaded"
    assert body["checks"]["reranker_model"]["status"] == "loaded"


def test_ready_is_503_while_the_enabled_reranker_is_loading(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: True)
    monkeypatch.setattr("api.main.is_reranker_model_loaded", lambda: False)
    monkeypatch.setattr("api.main.reranker_model_load_error", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["reranker_model"]["status"] == "loading"


def test_ready_allows_an_explicitly_disabled_reranker(client, monkeypatch):
    from src.config import Settings, get_settings

    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: True)
    monkeypatch.setattr("api.main.is_reranker_model_loaded", lambda: False)
    stub = Settings(_env_file=None).model_copy(
        update={"llm_provider": "ollama", "reranker_enabled": False}
    )
    app.dependency_overrides[get_settings] = lambda: stub

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["reranker_model"]["status"] == "disabled"


def test_ready_reports_the_database_and_model_independently(client, monkeypatch):
    """A loaded model must not mask a dead database."""
    monkeypatch.setattr(
        "api.main.count_chunks", lambda: (_ for _ in ()).throw(ConnectionError("down"))
    )
    monkeypatch.setattr("api.main.is_model_loaded", lambda: True)

    response = client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["database"]["status"] == "unavailable"
    assert checks["embedding_model"]["status"] == "loaded"


def test_query_is_503_while_the_model_is_loading(client, monkeypatch):
    class GraphThatMustNotRun:
        def invoke(self, question, request_id=None):
            raise AssertionError("the graph ran before the embedding model was ready")

    app.dependency_overrides[graph_dependency] = lambda: GraphThatMustNotRun()
    monkeypatch.setattr("api.main.is_model_loaded", lambda: False)
    monkeypatch.setattr("api.main.model_load_error", lambda: None)

    response = client.post("/query", json={"query": "How long do refunds take?"})

    assert response.status_code == 503
    assert response.json()["error"] == "service_not_ready"
    assert response.json()["request_id"]


def test_query_reports_model_initialization_failure_safely(client, monkeypatch):
    monkeypatch.setattr("api.main.is_model_loaded", lambda: False)
    monkeypatch.setattr(
        "api.main.model_load_error",
        lambda: "OSError: C:\\models\\private-cache download failed",
    )

    response = client.post("/query", json={"query": "How long do refunds take?"})

    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"
    assert "private-cache" not in response.text


def test_startup_schedules_a_background_warmup(monkeypatch):
    """The whole point: loading happens at start-up, not in the first request."""
    calls: list[str] = []
    monkeypatch.setattr("api.main.init_schema", lambda: None)
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    monkeypatch.setattr("api.main.warmup_embedding_model", lambda: calls.append("warmed") or True)
    monkeypatch.setattr("api.main.warmup_reranker_model", lambda: calls.append("reranked") or True)

    with TestClient(app):
        # Entering the context runs the lifespan, which starts a daemon thread.
        for _ in range(50):
            if len(calls) == 2:
                break
            time.sleep(0.05)

    assert sorted(calls) == ["reranked", "warmed"]


def test_startup_skips_model_warmup_when_database_initialization_fails(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "api.main.init_schema",
        lambda: (_ for _ in ()).throw(ConnectionError("database unavailable")),
    )
    monkeypatch.setattr(
        "api.main.count_chunks",
        lambda: (_ for _ in ()).throw(AssertionError("count must not run")),
    )
    monkeypatch.setattr("api.main.warmup_embedding_model", lambda: calls.append("warmed") or True)
    monkeypatch.setattr("api.main.warmup_reranker_model", lambda: calls.append("reranked") or True)

    with TestClient(app) as started_client:
        assert started_client.get("/health").status_code == 200

    assert calls == []


def test_startup_skips_model_warmup_until_the_corpus_is_ingested(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("api.main.init_schema", lambda: None)
    monkeypatch.setattr("api.main.count_chunks", lambda: 0)
    monkeypatch.setattr("api.main.warmup_embedding_model", lambda: calls.append("warmed") or True)
    monkeypatch.setattr("api.main.warmup_reranker_model", lambda: calls.append("reranked") or True)

    with TestClient(app) as started_client:
        assert started_client.get("/health").status_code == 200

    assert calls == []
