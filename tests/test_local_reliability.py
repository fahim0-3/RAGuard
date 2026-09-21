"""Local-only admission and cancellation-cleanup contracts."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from starlette.requests import Request

from api.admission import QueryAdmission
from api.main import admit_query, app, graph_dependency
from api.observability import runtime_metrics
from src.config import Settings, get_settings


def test_admission_generator_releases_a_slot_when_request_teardown_closes_it(monkeypatch):
    """Framework dependency teardown covers cancellation before graph completion."""
    admission = QueryAdmission()
    monkeypatch.setattr("api.main.query_admission", admission)
    runtime_metrics.reset()
    settings = Settings(
        _env_file=None,
        query_max_concurrency=1,
        db_pool_max_size=1,
        query_rate_limit_per_minute=10,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/query",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )

    dependency = admit_query(request, settings)
    next(dependency)
    assert runtime_metrics.snapshot()["queries"]["in_flight"] == 1

    dependency.close()

    replacement = admission.acquire("127.0.0.1", max_concurrency=1, requests_per_minute=10)
    assert replacement.reason is None
    assert runtime_metrics.snapshot()["queries"]["in_flight"] == 0


def test_query_admission_rejects_saturation_then_recovers_after_work_completes(monkeypatch):
    """A busy request must not consume a slot permanently after the first completes."""

    class BlockingGraph:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def invoke(self, question: str, request_id: str | None = None) -> dict[str, object]:
            self.started.set()
            assert self.release.wait(timeout=2), "test failed to release the blocked graph"
            return {
                "request_id": request_id,
                "original_query": question,
                "current_query": question,
                "rewritten_queries": [],
                "final_outcome": "abstain",
                "final_answer": "Insufficient evidence to answer safely.",
                "failure_reason": "",
                "risk_level": "none",
                "citations": [],
                "answer_confidence": 0.0,
                "evidence_grade": {"sufficient": False, "confidence": 0.0},
                "verification_result": {"checked": False, "supported": False},
                "retrieved_chunks": [],
                "reranker_used": False,
                "node_sequence": [],
                "timestamps": {},
                "retry_count": 0,
                "max_retries": 0,
                "regeneration_count": 0,
                "max_regenerations": 0,
                "prompt_version": "test",
                "llm_calls_used": 0,
                "llm_call_limit": 0,
                "budget_exhausted": False,
            }

    graph = BlockingGraph()
    settings = Settings(
        _env_file=None,
        query_max_concurrency=1,
        db_pool_max_size=1,
        query_rate_limit_per_minute=10,
    )
    admission = QueryAdmission()
    monkeypatch.setattr("api.main.query_admission", admission)
    monkeypatch.setattr("api.main.is_model_loaded", lambda: True)
    runtime_metrics.reset()
    app.dependency_overrides[graph_dependency] = lambda: graph
    app.dependency_overrides[get_settings] = lambda: settings

    def post_query() -> int:
        client = TestClient(app, raise_server_exceptions=False)
        try:
            return client.post("/query", json={"query": "How long is a refund?"}).status_code
        finally:
            client.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_request = executor.submit(post_query)
            assert graph.started.wait(timeout=2), "first request never entered the graph"

            def graph_must_not_be_resolved_for_a_rejected_request() -> BlockingGraph:
                raise AssertionError("admission rejection resolved the graph dependency")

            app.dependency_overrides[graph_dependency] = (
                graph_must_not_be_resolved_for_a_rejected_request
            )

            client = TestClient(app, raise_server_exceptions=False)
            try:
                saturated = client.post("/query", json={"query": "How long is a refund?"})
            finally:
                client.close()
            assert saturated.status_code == 503
            assert saturated.json()["error"] == "service_busy"
            assert runtime_metrics.snapshot()["queries"]["in_flight"] == 1

            graph.release.set()
            assert first_request.result(timeout=2) == 200

        app.dependency_overrides[graph_dependency] = lambda: graph
        client = TestClient(app, raise_server_exceptions=False)
        try:
            recovered = client.post("/query", json={"query": "How long is a refund?"})
        finally:
            client.close()
        assert recovered.status_code == 200
        snapshot = runtime_metrics.snapshot()["queries"]
        assert snapshot["in_flight"] == 0
        assert snapshot["concurrency_limit"] == 1
        assert snapshot["admission_rejections"] == {"busy": 1}
    finally:
        app.dependency_overrides.clear()
        runtime_metrics.reset()
