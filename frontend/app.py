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

st.markdown(
    """
    <style>
      .stApp { background: #f7f8f6; color: #1b2927; }
      .stApp p, .stApp li, .stApp label,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stCaptionContainer"] { color: #1b2927; }
      [data-testid="stSidebar"] { background: #173a3a; }
      [data-testid="stSidebar"] * { color: #f4f7f4; }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stTextInput"] label,
      [data-testid="stSidebar"] summary,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f4f7f4 !important;
      }
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] textarea {
        background: #224b4b;
        border-color: #5b8580;
      }
      [data-testid="stSidebar"] [data-baseweb="input"] input,
      [data-testid="stSidebar"] [data-baseweb="input"] textarea,
      [data-testid="stSidebar"] [data-baseweb="select"] *,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] .stButton > button {
        color: #f4f7f4 !important;
      }
      [data-testid="stSidebar"] input::placeholder,
      [data-testid="stSidebar"] textarea::placeholder { color: #b9d4cf !important; opacity: 1; }
      [data-testid="stSidebar"] .stButton > button {
        background: #224b4b;
        border-color: #5b8580;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        background: #2c5d5c;
        border-color: #82aaa4;
      }
      .stApp h1, .stApp h2, .stApp h3 { color: #1b2927; }
      .stApp [data-testid="stAlert"] p,
      .stApp [data-testid="stAlert"] div { color: #1b2927; }
      .stApp [data-testid="stMetricLabel"],
      .stApp [data-testid="stMetricLabel"] *,
      .stApp [data-testid="stMetricValue"],
      .stApp [data-testid="stMetricValue"] * { color: #1b2927; }
      .stApp button[data-baseweb="tab"] { color: #1b2927; }
      .stApp button[data-baseweb="tab"][aria-selected="true"] { color: #0f635a; }
      .stApp [data-testid="stExpander"] summary,
      .stApp [data-testid="stExpander"] p,
      .stApp code, .stApp pre { color: #1b2927; }
      .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
      .stApp [data-testid="stTextArea"] [data-baseweb="input"] {
        background: #224b4b;
        border-color: #5b8580;
      }
      .stApp [data-testid="stSelectbox"] [data-baseweb="select"] *,
      .stApp [data-testid="stTextArea"] textarea {
        color: #f4f7f4 !important;
      }
      .stApp [data-testid="stTextArea"] textarea::placeholder {
        color: #b9d4cf !important;
        opacity: 1;
      }
      .stApp [data-testid="stTextArea"] [data-baseweb="input"]:focus-within {
        border-color: #82aaa4 !important;
        box-shadow: 0 0 0 1px #82aaa4;
      }
      .stApp [data-testid="stTextArea"] [data-baseweb="input"]:has(textarea:invalid),
      .stApp [data-testid="stTextArea"] [data-baseweb="input"]:has(textarea[aria-invalid="true"]) {
        border-color: #c95a5a !important;
        box-shadow: 0 0 0 1px #c95a5a;
      }
      .stApp [data-testid="stButton"] > button:not([kind="primary"]) {
        background: #224b4b;
        border-color: #5b8580;
        color: #f4f7f4 !important;
      }
      .stApp [data-testid="stButton"] > button:not([kind="primary"]):hover {
        background: #2c5d5c;
        border-color: #82aaa4;
      }
      [data-testid="stSidebar"] [data-testid="stAlert"] p,
      [data-testid="stSidebar"] [data-testid="stAlert"] div,
      [data-testid="stSidebar"] [data-testid="stExpander"] summary,
      [data-testid="stSidebar"] [data-testid="stExpander"] p,
      [data-testid="stSidebar"] code,
      [data-testid="stSidebar"] pre { color: #f4f7f4 !important; }
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe3df;
        border-radius: 6px;
        padding: 0.65rem 0.8rem;
      }
      .raguard-kicker {
        color: #167a6e;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
      }
      .raguard-subtitle { color: #526460; margin-bottom: 1.25rem; }
      .stButton > button[kind="primary"] {
        background: #167a6e;
        border-color: #167a6e;
      }
      .stButton > button[kind="primary"]:hover {
        background: #0f635a;
        border-color: #0f635a;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

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
APPLIED_EXAMPLE_KEY = "applied_example_choice"
RECENT_KEY = "recent_questions"
NO_EXAMPLE = "(type your own)"
MAX_RECENT_QUESTIONS = 5


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

st.markdown(
    '<div class="raguard-kicker">Customer support policy desk</div>', unsafe_allow_html=True
)
st.title("RAGuard")
st.markdown(
    '<div class="raguard-subtitle">Ask a policy question, review the cited evidence, and see the verified outcome.</div>',
    unsafe_allow_html=True,
)

# The question lives in session state, and the text area is bound to it by key
# with no `value=`. Passing `value=` recomputed from the example selector is
# what previously discarded typed text whenever the selector changed.
st.session_state.setdefault(QUESTION_KEY, "")
st.session_state.setdefault(APPLIED_EXAMPLE_KEY, NO_EXAMPLE)
st.session_state.setdefault(RECENT_KEY, [])


def _clear_view() -> None:
    st.session_state.pop("view", None)


def _sync_selected_example(choice: str) -> None:
    """Apply a newly selected example before the Question widget is created.

    The marker distinguishes selecting an example from a later rerun caused by
    editing the textarea.  That makes the selector deterministic without
    overwriting a user's edit or mutating the textarea widget after creation.
    """
    if choice == st.session_state.get(APPLIED_EXAMPLE_KEY, NO_EXAMPLE):
        return
    st.session_state[QUESTION_KEY] = "" if choice == NO_EXAMPLE else choice
    st.session_state[APPLIED_EXAMPLE_KEY] = choice
    _clear_view()


def _clear_question() -> None:
    """Reset the composer without discarding the user's session history."""
    st.session_state[QUESTION_KEY] = ""
    st.session_state[EXAMPLE_KEY] = NO_EXAMPLE
    _clear_view()


def _remember_question(question: str) -> None:
    """Keep a small, local-only history of submitted questions."""
    history = [item for item in st.session_state.get(RECENT_KEY, []) if item != question]
    st.session_state[RECENT_KEY] = [question, *history][:MAX_RECENT_QUESTIONS]


def _reuse_question(question: str) -> None:
    """Put a session question back in the editable composer."""
    st.session_state[QUESTION_KEY] = question
    st.session_state[EXAMPLE_KEY] = NO_EXAMPLE
    _clear_view()


composer, session_panel = st.columns([3, 2], gap="large")

with composer:
    st.subheader("Ask a policy question")
    selected_example = st.selectbox(
        "Example questions",
        [NO_EXAMPLE, *EXAMPLES],
        key=EXAMPLE_KEY,
        help="Optional starting points. The question below stays fully editable.",
    )
    _sync_selected_example(selected_example)
    st.text_area(
        "Question",
        key=QUESTION_KEY,
        height=130,
        placeholder="Ask anything about refunds, returns, delivery, damage, or payments…",
    )

    submit_column, clear_column, _ = st.columns([1, 1, 5])
    with submit_column:
        # Never disabled. `st.text_area` does not rerun the script on
        # keystrokes, so a `disabled=not question` button is still disabled at
        # the moment the user clicks it after typing. Validation happens on
        # submit instead, against the same bounds as the API.
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with clear_column:
        st.button("Clear", on_click=_clear_question, use_container_width=True)

    if ask_clicked:
        question = (st.session_state.get(QUESTION_KEY) or "").strip()

        if len(question) < MIN_QUESTION_CHARS:
            st.session_state["view"] = present(
                {
                    "error": "invalid_question",
                    "detail": (
                        f"Please enter a question of at least {MIN_QUESTION_CHARS} characters."
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
            _remember_question(question)
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

with session_panel:
    st.subheader("This session")
    history = st.session_state.get(RECENT_KEY, [])
    if history:
        for index, item in enumerate(history):
            st.button(
                item,
                key=f"recent_question_{index}",
                on_click=_reuse_question,
                args=(item,),
                use_container_width=True,
            )
    else:
        st.caption("Your submitted questions will appear here.")

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

    if view.is_error:
        if view.body:
            st.markdown(view.body)
        st.stop()

    answer_tab, evidence_tab, trace_tab = st.tabs(["Answer", "Evidence", "Decision trace"])
    with answer_tab:
        if view.body:
            st.markdown(view.body)

        columns = st.columns(len(view.metrics))
        for column, (label, value) in zip(columns, view.metrics.items(), strict=True):
            column.metric(label, value)

        st.caption(view.verification)
        if view.failure_reason:
            st.caption(f"Reason: {view.failure_reason}")

    with evidence_tab:
        # Citation metadata is exactly as validated server-side.
        if view.citations:
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
        else:
            st.caption("No policy passage was cited for this outcome.")

    with trace_tab:
        if view.rewritten_queries:
            st.subheader("Query rewrites tried")
            for index, rewritten in enumerate(view.rewritten_queries, start=1):
                st.code(f"{index}. {rewritten}", language=None)

        if view.trace:
            st.markdown("  \n".join(f"{row['step']}. {row['label']}" for row in view.trace))
        if view.request_id:
            st.caption(f"request_id: {view.request_id}")
