"""Regression tests for the Streamlit question input.

These drive the *real* app script through Streamlit's own `AppTest` harness, so
they exercise widget state and rerun behaviour rather than asserting on source
text. That matters here: the original defect was invisible to unit tests
because it lived entirely in the interaction between `st.text_area` (which does
not rerun on keystrokes) and a `disabled=` flag computed from the previous run.

Only the socket is faked. `httpx.get`/`httpx.post` are replaced, so the app's
own `check_readiness`, `call_api`, and presenter logic all execute for real —
patching `frontend.app.call_api` would not work anyway, because `AppTest` execs
the script in its own namespace rather than importing the module.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402

APP = str(PROJECT_ROOT / "frontend" / "app.py")
EXAMPLE = "How long does a refund take to reach my credit card?"
CUSTOM = "What resolution options are available for a validated damaged product claim?"

ANSWER_PAYLOAD = {
    "request_id": "req-1",
    "outcome": "answer",
    "answer": "Free replacement, full refund, or a repair credit of up to 40 percent.",
    "citations": [
        {
            "citation_label": "damaged_product_policy.txt#2",
            "policy_id": "DMG-003",
            "source": "damaged_product_policy.txt",
            "chunk_index": 2,
            "chunk_id": 25,
            "excerpt": "Free replacement, shipped within 2 business days...",
        }
    ],
    "confidence": 0.9,
    "more_info_required": False,
    "retry_count": 0,
    "max_retries": 2,
    "rewritten_queries": [],
    "risk_level": "none",
    "verification_status": "supported",
    "verified_claim_count": 1,
    "unsupported_claim_count": 0,
    "retrieved_chunk_count": 5,
    "reranker_used": True,
    "failure_reason": None,
    "trace": [{"step": 1, "node": "sanitize_and_classify"}],
    "latency_ms": 1200.0,
}

READY_OK = {"status": "ready", "checks": {}, "detail": None}


class FakeResponse:
    """Minimal stand-in for `httpx.Response`."""

    def __init__(self, status_code: int, payload: object, json_ok: bool = True) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_ok = json_ok

    def json(self):
        if not self._json_ok:
            raise ValueError("not json")
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload) if self._json_ok else "<html>"


@pytest.fixture
def api(monkeypatch):
    """Fake the HTTP boundary and record what the app sent."""
    import httpx

    state = {
        "sent": [],
        "query_response": FakeResponse(200, ANSWER_PAYLOAD),
        "ready_response": FakeResponse(200, READY_OK),
        "query_raises": None,
    }

    def fake_get(url, *args, **kwargs):
        if url.endswith("/ready"):
            return state["ready_response"]
        if url.endswith("/health"):
            return FakeResponse(200, {"status": "ok", "version": "0.2.0"})
        return FakeResponse(200, {})

    def fake_post(url, *args, **kwargs):
        state["sent"].append((kwargs.get("json") or {}).get("query"))
        if state["query_raises"] is not None:
            raise state["query_raises"]
        return state["query_response"]

    monkeypatch.setattr(httpx, "get", fake_get, raising=True)
    monkeypatch.setattr(httpx, "post", fake_post, raising=True)
    return state


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    return app


# --------------------------------------------------------------------------
# The original defect
# --------------------------------------------------------------------------


def test_ask_button_is_never_disabled(api):
    """The regression that made custom questions unsubmittable.

    `st.text_area` does not rerun the script while the user types, so a button
    guarded by `disabled=not question.strip()` was still disabled at the moment
    the user clicked it. Typing a question and pressing Ask did nothing.
    """
    app = run_app()

    ask = [b for b in app.button if b.label == "Ask"]
    assert ask, "Ask button missing"
    assert ask[0].disabled is False, "Ask must not be disabled by stale state"


def test_typing_a_custom_question_and_asking_sends_it(api):
    """A question typed into an empty field must reach the API verbatim."""
    app = run_app()

    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    assert api["sent"] == [CUSTOM]


def test_custom_question_survives_a_rerun(api):
    """Typed text must not be discarded when the script reruns."""
    app = run_app()

    app.text_area(key="question_text").input(CUSTOM).run()
    app.run()

    assert app.session_state["question_text"] == CUSTOM


# --------------------------------------------------------------------------
# Examples are suggestions, not a cage
# --------------------------------------------------------------------------


def test_selecting_an_example_populates_the_question_field(api):
    app = run_app()

    app.selectbox(key="example_choice").select(EXAMPLE).run()

    assert app.session_state["question_text"] == EXAMPLE


def test_selecting_an_example_clears_a_stale_validation_message(api):
    app = run_app()

    app.button[0].click().run()
    assert app.session_state["view"].heading == "Check your question"

    app.selectbox(key="example_choice").select(EXAMPLE).run()

    assert app.session_state["question_text"] == EXAMPLE
    assert "view" not in app.session_state


def test_an_edited_example_is_not_overwritten_by_reruns(api):
    """Selecting, then editing, then rerunning must keep the edit.

    Previously the text area received `value=` recomputed from the selector, so
    Streamlit re-initialised the widget and the edit vanished.
    """
    app = run_app()
    app.selectbox(key="example_choice").select(EXAMPLE).run()

    edited = EXAMPLE + " And what about PayPal?"
    app.text_area(key="question_text").input(edited).run()
    app.run()

    assert app.session_state["question_text"] == edited

    app.button[0].click().run()
    assert api["sent"] == [edited]


def test_choosing_type_your_own_clears_the_field(api):
    app = run_app()
    app.selectbox(key="example_choice").select(EXAMPLE).run()

    app.selectbox(key="example_choice").select("(type your own)").run()

    assert app.session_state["question_text"] == ""


# --------------------------------------------------------------------------
# Validation is preserved, not removed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "ab"], ids=["empty", "whitespace", "too_short"])
def test_short_questions_are_rejected_without_calling_the_api(api, bad):
    app = run_app()

    app.text_area(key="question_text").input(bad).run()
    app.button[0].click().run()

    assert api["sent"] == [], "must not send an invalid question"
    assert app.session_state["view"].is_error is True
    assert app.session_state["view"].kind == "info"


def test_overlong_question_is_rejected(api):
    app = run_app()

    app.text_area(key="question_text").input("x" * 2500).run()
    app.button[0].click().run()

    assert api["sent"] == []
    assert "2000" in app.session_state["view"].body


def test_question_is_stripped_before_sending(api):
    app = run_app()

    app.text_area(key="question_text").input(f"   {CUSTOM}   ").run()
    app.button[0].click().run()

    assert api["sent"] == [CUSTOM]


# --------------------------------------------------------------------------
# Readiness and failure surfaces
# --------------------------------------------------------------------------


def test_query_is_not_sent_while_the_service_is_not_ready(api):
    """A query must never be parked inside a model download."""
    api["ready_response"] = FakeResponse(
        503,
        {
            "status": "not_ready",
            "checks": {},
            "detail": "embedding model still loading (first start downloads ~2.2 GB)",
        },
    )

    app = run_app()
    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    assert api["sent"] == []
    view = app.session_state["view"]
    assert view.kind == "warning", "not-ready is not a crash"
    assert "still loading" in view.body


def test_timeout_leaves_a_visible_explanation(api):
    """The frontend must never end on a spinner with no message."""
    import httpx

    api["query_raises"] = httpx.ReadTimeout("too slow")

    app = run_app()
    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    view = app.session_state["view"]
    assert view.is_error is True
    assert view.heading == "Timed out"
    assert view.body


def test_unreachable_api_is_reported_without_leaking_detail(api):
    import httpx

    api["query_raises"] = httpx.ConnectError("connection refused to secret-host:5432")

    app = run_app()
    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    view = app.session_state["view"]
    assert view.heading == "API unreachable"
    assert "secret-host" not in view.body, "raw transport detail must not surface"


def test_sidebar_never_renders_a_raw_connection_exception(api, monkeypatch):
    import httpx

    original_get = httpx.get

    def fail_health(url, *args, **kwargs):
        if url.endswith("/health"):
            raise httpx.ConnectError("connection refused to private-host:8000")
        return original_get(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "get", fail_health)

    app = run_app()

    rendered = " ".join(error.value for error in app.error)
    assert "API unreachable." in rendered
    assert "private-host" not in rendered


def test_non_json_response_is_reported(api):
    api["query_response"] = FakeResponse(502, None, json_ok=False)

    app = run_app()
    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    assert app.session_state["view"].heading == "Unexpected response"


def test_successful_answer_is_rendered_with_citations(api):
    app = run_app()

    app.text_area(key="question_text").input(CUSTOM).run()
    app.button[0].click().run()

    view = app.session_state["view"]
    assert view.outcome == "answer"
    assert view.kind == "success"
    assert view.citations[0]["policy_id"] == "DMG-003"


def test_abstention_is_rendered_as_a_refusal_not_an_error(api):
    api["query_response"] = FakeResponse(
        200,
        {
            **ANSWER_PAYLOAD,
            "outcome": "abstain",
            "answer": "I don't have enough information in the available policies.",
            "citations": [],
            "verification_status": "not_checked",
            "more_info_required": True,
        },
    )

    app = run_app()
    app.text_area(key="question_text").input("Can I get a mortgage here?").run()
    app.button[0].click().run()

    view = app.session_state["view"]
    assert view.outcome == "abstain"
    assert view.kind == "warning"
    assert view.is_error is False
