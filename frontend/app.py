"""Streamlit demonstration UI.

Built around the guard rails rather than the chat bubble. A plain RAG demo shows
an answer; this one shows why the answer was allowed through, or why it was
refused: the evidence grade, the rewrites the healing loop tried, the citation
verdict, and the decision path through the workflow.

No RAG logic lives here. The UI posts to FastAPI and renders what comes back;
every display decision is a pure function in `presenter.py`, which is where the
tests are. That split is deliberate — the interesting behaviour is "how should
an abstention look", and that should not require a browser to verify.

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# `streamlit run frontend/app.py` puts only `frontend/` on sys.path, so the
# package import below needs the repository root added explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.presenter import present  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "180"))

st.set_page_config(page_title="RAGuard", page_icon="🛡️", layout="wide")

EXAMPLES = [
    "How long does a refund take to reach my credit card?",
    "What does error PAY-402 mean at checkout?",
    "I have a problem with my order",
    "Can I get a mortgage or a personal loan through your store?",
    "I was charged twice for the same order",
]


#: Mirrors the API contract in `api/schemas.py`. Checked here only to give
#: immediate feedback; the server remains the authority and still rejects
#: anything out of range with 422.
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2000

#: Session-state key for the question. The text area is driven entirely through
#: this key and is never passed a `value=`, so a rerun cannot overwrite what the
#: user typed.
QUESTION_KEY = "question_text"
EXAMPLE_KEY = "example_choice"
NO_EXAMPLE = "(type your own)"


def check_readiness(base_url: str) -> tuple[bool, str]:
    """Ask the API whether it can actually serve a query.

    Called before sending, so a request is never parked inside a model download.
    Returns (ready, human-readable reason).
    """
    try:
        response = httpx.get(f"{base_url}/ready", timeout=15.0)
    except httpx.HTTPError as exc:
        return False, f"Could not reach the API at {base_url} ({type(exc).__name__})."

    if response.status_code == 200:
        return True, ""

    try:
        detail = response.json().get("detail") or "The service is not ready."
    except ValueError:
        detail = f"The service is not ready (HTTP {response.status_code})."
    return False, detail


def call_api(base_url: str, question: str) -> dict:
    """Post the question, mapping each failure mode to a distinct outcome.

    Every branch returns something renderable. The frontend must never be left
    on a spinner with no explanation, which is what a bare `raise` here would
    produce.
    """
    try:
        response = httpx.post(
            f"{base_url}/query", json={"query": question}, timeout=REQUEST_TIMEOUT
        )
    except httpx.TimeoutException:
        return {
            "error": "timeout",
            "detail": (
                f"The API did not respond within {REQUEST_TIMEOUT:.0f}s. It may be "
                "loading models or under load. Check /ready, then try again."
            ),
        }
    except httpx.HTTPError as exc:
        # Connection refused, DNS failure, TLS problem. The type name is safe;
        # the full string can contain internal hostnames.
        return {
            "error": "unreachable",
            "detail": f"Could not reach the API at {base_url} ({type(exc).__name__}).",
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "error": "bad_response",
            "detail": f"The API returned a non-JSON response (HTTP {response.status_code}).",
        }

    if not isinstance(payload, dict):
        return {"error": "bad_response", "detail": "The API returned an unexpected payload."}

    if response.status_code >= 400 and "error" not in payload:
        payload = {
            "error": "http_error",
            "detail": f"The API returned HTTP {response.status_code}.",
        }
    return payload


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("RAGuard")
    st.caption("Self-healing hybrid RAG with citation verification")

    api_url = st.text_input("API base URL", API_BASE_URL)

    st.divider()
    st.subheader("Service status")
    try:
        health = httpx.get(f"{api_url}/health", timeout=10.0).json()
        st.success(f"API {health.get('status', 'unknown')} · v{health.get('version', '?')}")
    except Exception:
        # The raw exception can include an internal hostname or credentials
        # embedded in a user-edited URL. Keep the sidebar actionable and safe.
        st.error("API unreachable.")

    try:
        readiness = httpx.get(f"{api_url}/ready", timeout=15.0)
        if readiness.status_code == 200:
            st.success("Dependencies ready")
        else:
            st.warning(readiness.json().get("detail", "Not ready"))
        with st.expander("Readiness detail"):
            st.json(readiness.json().get("checks", {}))
    except Exception:
        st.caption("Readiness unavailable")

    with st.expander("Active configuration"):
        try:
            st.json(httpx.get(f"{api_url}/config", timeout=10.0).json())
        except Exception:
            st.caption("Unavailable")


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

st.header("Ask the support assistant")

# The question lives in session state, and the text area is bound to it by key
# with no `value=`. Passing `value=` recomputed from the example selector is
# what previously discarded typed text whenever the selector changed.
st.session_state.setdefault(QUESTION_KEY, "")


def _clear_view() -> None:
    st.session_state.pop("view", None)


def _apply_example() -> None:
    """Copy the chosen example into the editable question field.

    Runs only on an actual change of the selector, so it cannot overwrite text
    the user is typing. Choosing "(type your own)" clears the field, which is
    the only way to get an empty box back once an example has been used.
    """
    choice = st.session_state.get(EXAMPLE_KEY, NO_EXAMPLE)
    st.session_state[QUESTION_KEY] = "" if choice == NO_EXAMPLE else choice
    _clear_view()


st.selectbox(
    "Example questions",
    [NO_EXAMPLE, *EXAMPLES],
    key=EXAMPLE_KEY,
    on_change=_apply_example,
    help="A starting point. The question below stays fully editable.",
)
st.text_area(
    "Question",
    key=QUESTION_KEY,
    height=90,
    placeholder="Ask anything about refunds, returns, delivery, damage, or payments…",
)

# Never disabled. `st.text_area` does not rerun the script on keystrokes, so a
# `disabled=not question` button is still disabled at the moment the user
# clicks it after typing — which is what made custom questions unsubmittable.
# Validation happens on submit instead, against the same bounds as the API.
if st.button("Ask", type="primary"):
    question = (st.session_state.get(QUESTION_KEY) or "").strip()

    if len(question) < MIN_QUESTION_CHARS:
        st.session_state["view"] = present(
            {
                "error": "invalid_question",
                "detail": (
                    f"Please enter a question of at least {MIN_QUESTION_CHARS} "
                    "characters."
                ),
            }
        )
    elif len(question) > MAX_QUESTION_CHARS:
        st.session_state["view"] = present(
            {
                "error": "invalid_question",
                "detail": (
                    f"That question is {len(question)} characters; the limit is "
                    f"{MAX_QUESTION_CHARS}."
                ),
            }
        )
    else:
        # Check readiness first so a query is never parked inside a model
        # download. This is also what makes the spinner text below truthful.
        with st.spinner("Checking service readiness…"):
            ready, reason = check_readiness(api_url)

        if not ready:
            st.session_state["view"] = present({"error": "not_ready", "detail": reason})
        else:
            with st.spinner("Processing your question…"):
                payload = call_api(api_url, question)
            st.session_state["view"] = present(payload)

view = st.session_state.get("view")

if view is None:
    st.info("Ask a question to see the decision path.")
else:
    render = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }[view.kind]
    render(f"**{view.heading}** — {view.explanation}")

    if view.body:
        st.markdown(view.body)

    if view.is_error:
        st.stop()

    columns = st.columns(len(view.metrics))
    for column, (label, value) in zip(columns, view.metrics.items(), strict=True):
        column.metric(label, value)

    st.caption(view.verification)
    if view.failure_reason:
        st.caption(f"Reason: {view.failure_reason}")

    # -- Citations, exactly as validated server-side --------------------
    if view.citations:
        st.subheader("Citations")
        for citation in view.citations:
            header = f"{citation['policy_id']} · {citation['label']}"
            with st.expander(header):
                st.caption(
                    f"source: {citation['source']} · chunk_index: {citation['chunk_index']} "
                    f"· chunk_id: {citation['chunk_id']}"
                )
                st.write(citation["excerpt"])
    elif view.outcome == "answer":
        st.warning("This answer carries no citations, which should not happen.")

    # -- Self-healing surface --------------------------------------------
    if view.rewritten_queries:
        st.subheader("Query rewrites tried")
        for index, rewritten in enumerate(view.rewritten_queries, start=1):
            st.code(f"{index}. {rewritten}", language=None)

    # -- Decision path (operational state only, never reasoning) ----------
    if view.trace:
        st.subheader("Decision path")
        st.markdown(
            "  \n".join(f"{row['step']}. {row['label']}" for row in view.trace)
        )

    if view.request_id:
        st.caption(f"request_id: {view.request_id}")
