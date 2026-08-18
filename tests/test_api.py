"""Phase H: the FastAPI transport layer.

The graph is stubbed. That is the point of these tests: the API's job is
validation, error translation, request-ID propagation, and projecting graph
state onto a contract, and each of those is testable without pgvector, a
cross-encoder, or a provider.

No API key and no database are required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, graph_dependency
from src.generation.llm_factory import LLMProviderError
from src.retrieval.types import RetrievedChunk


def chunk(label_index: int = 2, source: str = "refund_policy.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=40,
        content="Refunds to credit and debit cards take 5 to 7 business days.",
        source=source,
        chunk_index=label_index,
        doc_id="REF-001",
    )


def graph_state(**overrides):
    """A graph final state, shaped as the workflow really returns it."""
    state = {
        "request_id": "req-1",
        "original_query": "How long do refunds take?",
        "current_query": "How long do refunds take?",
        "rewritten_queries": [],
        "final_outcome": "answer",
        "final_answer": "Refunds take 5 to 7 business days.",
        "failure_reason": "",
        "risk_level": "none",
        "citations": ["refund_policy.txt#2"],
        "answer_confidence": 0.9,
        "evidence_grade": {"sufficient": True, "confidence": 0.95},
        "verification_result": {
            "checked": True, "supported": True,
            "supported_claim_count": 1, "unsupported_claim_count": 0,
        },
        "retrieved_chunks": [chunk()],
        "reranker_used": True,
        "node_sequence": ["sanitize_and_classify", "risk_router", "generate_answer"],
        "timestamps": {"started_at": "2026-08-16T00:00:00Z"},
        "retry_count": 0,
        "max_retries": 2,
        "regeneration_count": 0,
        "max_regenerations": 1,
        "prompt_version": "2026-08-16_prompts_v2",
    }
    state.update(overrides)
    return state


class StubGraph:
    """Stands in for SelfHealingGraph."""

    def __init__(self, state=None, error: Exception | None = None):
        self.state = state if state is not None else graph_state()
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def invoke(self, question, request_id=None):
        self.calls.append((question, request_id))
        if self.error is not None:
            raise self.error
        return {**self.state, "request_id": request_id or self.state.get("request_id")}


@pytest.fixture
def client():
    """A TestClient that does not run the application lifespan.

    Entering `TestClient`'s context manager fires FastAPI's startup handler,
    which calls `init_schema()` and then `count_chunks()`. Both open a psycopg
    connection. This tier has no database by design, and the pool does not fail
    fast on a refused connection — it waits out its 30 second timeout — so the
    context manager cost 30s of *setup* per test, 38 times over. Locally the
    database is usually up, which is why it only showed on CI.

    None of these tests need the startup handler: they stub the graph and
    exercise validation, routing, error translation, and response projection.
    The end-to-end test builds its own `with TestClient(app)` so it still runs
    the real lifespan against a real database.
    """
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


def use_graph(stub: StubGraph) -> StubGraph:
    app.dependency_overrides[graph_dependency] = lambda: stub
    return stub


# --------------------------------------------------------------------------
# 1. Health
# --------------------------------------------------------------------------


def test_health_is_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_does_not_touch_the_database(client, monkeypatch):
    """A dead database must not make the process look dead."""

    def explode():
        raise ConnectionError("pgvector unreachable")

    monkeypatch.setattr("src.retrieval.vector_store.count_chunks", explode)

    assert client.get("/health").status_code == 200


def test_health_exposes_no_configuration(client):
    body = response_body = client.get("/health").json()

    assert "google_api_key" not in str(body).lower()
    assert set(response_body) == {"status", "service", "version"}


# --------------------------------------------------------------------------
# 2. Readiness
# --------------------------------------------------------------------------


def test_ready_reports_not_ready_when_the_database_is_down(client, monkeypatch):
    monkeypatch.setattr(
        "api.main.count_chunks", lambda: (_ for _ in ()).throw(ConnectionError("down"))
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["database"]["status"] == "unavailable"


def test_ready_reports_not_ready_when_the_corpus_is_empty(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 0)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "no chunks indexed" in response.json()["detail"]


def use_settings(**overrides):
    """Inject settings through FastAPI rather than the environment.

    Environment patching is not isolation here: pydantic-settings also reads
    `.env`, so a developer machine with real credentials would silently pass a
    test asserting they are absent.
    """
    from src.config import Settings, get_settings

    base = Settings(_env_file=None)  # type: ignore[call-arg]
    stub = base.model_copy(update=overrides)
    app.dependency_overrides[get_settings] = lambda: stub
    return stub


def test_ready_is_ready_when_dependencies_are_present(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    use_settings(llm_provider="ollama")

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_flags_a_missing_api_key(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    use_settings(llm_provider="gemini", google_api_key=None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_ready_never_echoes_the_key_value(client, monkeypatch):
    monkeypatch.setattr("api.main.count_chunks", lambda: 22)
    use_settings(llm_provider="gemini", google_api_key="secret-value-xyz")

    assert "secret-value-xyz" not in client.get("/ready").text


# --------------------------------------------------------------------------
# 3-5. Query validation
# --------------------------------------------------------------------------


def test_valid_query_returns_an_answer(client):
    use_graph(StubGraph())

    response = client.post("/query", json={"query": "How long do refunds take?"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answer"
    assert body["answer"] == "Refunds take 5 to 7 business days."
    assert body["more_info_required"] is False


def test_question_alias_still_works(client):
    """The earlier API and UI used `question`; breaking it buys nothing."""
    stub = use_graph(StubGraph())

    response = client.post("/query", json={"question": "How long do refunds take?"})

    assert response.status_code == 200
    assert stub.calls[0][0] == "How long do refunds take?"


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "  "}, {"query": "ab"}, {}],
    ids=["empty", "whitespace", "too_short", "missing"],
)
def test_invalid_query_is_rejected(client, payload):
    response = client.post("/query", json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_overlong_query_is_rejected(client):
    response = client.post("/query", json={"query": "x" * 5000})

    assert response.status_code == 422


def test_unknown_field_is_rejected(client):
    response = client.post("/query", json={"query": "hello there", "admin": True})

    assert response.status_code == 422
    assert "admin" in response.json()["detail"]


def test_malformed_body_is_rejected(client):
    response = client.post(
        "/query", content="not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_query_never_reaches_the_graph_when_invalid(client):
    stub = use_graph(StubGraph())

    client.post("/query", json={"query": "ab"})

    assert stub.calls == []


# --------------------------------------------------------------------------
# 6. Request ID
# --------------------------------------------------------------------------


def test_client_request_id_is_propagated(client):
    stub = use_graph(StubGraph())

    response = client.post(
        "/query", json={"query": "How long do refunds take?", "request_id": "req-abc-123"}
    )

    assert stub.calls[0][1] == "req-abc-123"
    assert response.json()["request_id"] == "req-abc-123"


def test_request_id_is_generated_when_absent(client):
    stub = use_graph(StubGraph())

    response = client.post("/query", json={"query": "How long do refunds take?"})

    assert stub.calls[0][1]
    assert response.json()["request_id"]


def test_unsafe_request_id_is_rejected(client):
    response = client.post(
        "/query", json={"query": "How long do refunds take?", "request_id": "a b\nc"}
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# 7-10. Outcomes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "answer"),
    [
        ("answer", "Refunds take 5 to 7 business days."),
        ("clarify", "Is this about a refund or a return?"),
        ("abstain", "I don't have enough information in the available policies."),
        ("escalate", "I am passing this to a member of our team."),
    ],
)
def test_every_outcome_is_returned(client, outcome, answer):
    use_graph(StubGraph(graph_state(final_outcome=outcome, final_answer=answer)))

    body = client.post("/query", json={"query": "some question here"}).json()

    assert body["outcome"] == outcome
    assert body["answer"] == answer


def test_non_answer_outcomes_require_more_information(client):
    use_graph(StubGraph(graph_state(final_outcome="abstain", final_answer="No answer.")))

    assert client.post("/query", json={"query": "a question"}).json()["more_info_required"]


def test_retry_information_is_exposed(client):
    use_graph(
        StubGraph(graph_state(retry_count=2, rewritten_queries=["refund processing time"]))
    )

    body = client.post("/query", json={"query": "a question"}).json()

    assert body["retry_count"] == 2
    assert body["max_retries"] == 2
    assert body["rewritten_queries"] == ["refund processing time"]


def test_escalation_reports_the_risk_level(client):
    use_graph(StubGraph(graph_state(final_outcome="escalate", risk_level="high")))

    assert client.post("/query", json={"query": "a question"}).json()["risk_level"] == "high"


# --------------------------------------------------------------------------
# 11-12, 14. Failures
# --------------------------------------------------------------------------


def test_provider_failure_is_503(client):
    use_graph(StubGraph(error=LLMProviderError("GOOGLE_API_KEY is not set")))

    response = client.post("/query", json={"query": "a question here"})

    assert response.status_code == 503
    assert response.json()["error"] == "provider_unavailable"


def test_provider_failure_does_not_leak_the_message(client):
    use_graph(StubGraph(error=LLMProviderError("key sk-secret-123 rejected")))

    assert "sk-secret-123" not in client.post("/query", json={"query": "a question"}).text


def test_database_failure_is_503(client):
    use_graph(StubGraph(error=ConnectionError("could not connect to postgresql://u:p@host")))

    response = client.post("/query", json={"query": "a question here"})

    assert response.status_code == 503


def test_database_failure_does_not_leak_the_connection_string(client):
    use_graph(StubGraph(error=ConnectionError("postgresql://raguard:raguard@db:5432")))

    assert "raguard:raguard" not in client.post("/query", json={"query": "a question"}).text


def test_unexpected_graph_error_is_handled(client):
    use_graph(StubGraph(error=RuntimeError("something exploded internally")))

    response = client.post("/query", json={"query": "a question here"})

    assert response.status_code == 503
    assert "exploded" not in response.text


def test_no_stack_trace_reaches_the_client(client):
    use_graph(StubGraph(error=RuntimeError("boom")))

    body = client.post("/query", json={"query": "a question here"}).text

    for leak in ("Traceback", "File \"", "line ", ".py"):
        assert leak not in body


# --------------------------------------------------------------------------
# 13. Citations
# --------------------------------------------------------------------------


def test_citations_carry_validated_metadata(client):
    use_graph(StubGraph())

    citations = client.post("/query", json={"query": "a question here"}).json()["citations"]

    assert len(citations) == 1
    assert citations[0]["policy_id"] == "REF-001"
    assert citations[0]["source"] == "refund_policy.txt"
    assert citations[0]["chunk_id"] == 40
    assert citations[0]["excerpt"]


def test_citation_without_a_retrieved_chunk_is_dropped(client):
    """The API never invents metadata for a label it cannot resolve."""
    use_graph(StubGraph(graph_state(citations=["ghost.txt#9"], retrieved_chunks=[chunk()])))

    assert client.post("/query", json={"query": "a question"}).json()["citations"] == []


def test_verification_status_is_reported(client):
    use_graph(StubGraph())

    body = client.post("/query", json={"query": "a question here"}).json()

    assert body["verification_status"] == "supported"
    assert body["verified_claim_count"] == 1


def test_unsupported_verification_is_reported(client):
    use_graph(
        StubGraph(
            graph_state(
                final_outcome="abstain",
                verification_result={
                    "checked": True, "supported": False,
                    "supported_claim_count": 0, "unsupported_claim_count": 2,
                },
            )
        )
    )

    body = client.post("/query", json={"query": "a question here"}).json()

    assert body["verification_status"] == "unsupported"
    assert body["unsupported_claim_count"] == 2


# --------------------------------------------------------------------------
# Trace and leakage
# --------------------------------------------------------------------------


def test_trace_exposes_the_node_path(client):
    use_graph(StubGraph())

    trace = client.post("/query", json={"query": "a question here"}).json()["trace"]

    assert [step["node"] for step in trace] == [
        "sanitize_and_classify", "risk_router", "generate_answer"
    ]
    assert trace[0]["step"] == 1


def test_response_carries_no_prompts_or_chunk_bodies_beyond_excerpts(client):
    use_graph(StubGraph())

    body = client.post("/query", json={"query": "a question here"}).json()

    assert "retrieved_chunks" not in body
    assert "evidence_grade" not in body
    for leak in ("you are raguard", "output_schema", "api_key", "system prompt"):
        assert leak not in str(body).lower()


def test_response_schema_is_closed(client):
    """Extra graph state cannot leak through: the model forbids unknown fields."""
    use_graph(StubGraph(graph_state(secret_internal_field="should not appear")))

    assert "secret_internal_field" not in client.post(
        "/query", json={"query": "a question here"}
    ).text


# --------------------------------------------------------------------------
# CORS and config
# --------------------------------------------------------------------------


def test_cors_allows_the_configured_origin(client):
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_cors_does_not_allow_an_arbitrary_origin(client):
    response = client.options(
        "/query",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"


def test_config_endpoint_exposes_no_secret(client):
    body = client.get("/config").json()

    assert "google_api_key" not in {k.lower() for k in body}
    assert body["api_version"]


# --------------------------------------------------------------------------
# 15. End-to-end, through the real stack
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.heavy
@pytest.mark.llm
def test_end_to_end_query_through_the_real_stack():
    """HTTP -> FastAPI -> LangGraph -> retrieval -> generation -> verification.

    Requires PostgreSQL with the corpus ingested and a configured provider, so
    it is excluded from every default tier. Run with:
        pytest -m "integration and llm"
    """
    from api.main import reset_graph_service

    app.dependency_overrides.clear()
    reset_graph_service()

    with TestClient(app) as live_client:
        response = live_client.post(
            "/query",
            json={
                "query": "How long does a refund take to reach my credit card?",
                "request_id": "e2e-live-1",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["request_id"] == "e2e-live-1"
    assert body["outcome"] in {"answer", "abstain", "clarify", "escalate"}
    assert body["trace"], "the decision path must be reported"
    assert body["retrieved_chunk_count"] > 0

    if body["outcome"] == "answer":
        assert body["citations"], "an answer must carry citations"
        assert body["verification_status"] == "supported"
        for citation in body["citations"]:
            assert citation["policy_id"]
            assert citation["source"]
