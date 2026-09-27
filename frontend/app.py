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

# 127.0.0.1 rather than `localhost` deliberately. On Windows `localhost`
# resolves to ::1 first, the API binds IPv4 only, and each new connection
# then stalls about two seconds before falling back: measured 2.3 s per
# call against 0.23 s here.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT_S", "180"))


@st.cache_resource(show_spinner=False)
def _client() -> httpx.Client:
    """One pooled client for the whole session.

    `httpx.get(...)` opens a new connection every call, and a click made
    three or four of them before the question was even sent. Reusing one
    connection pays any connection-setup cost once per process instead of
    once per call, which matters most when the base URL is a hostname that
    resolves to an address the API does not listen on.
    """
    # A generous `keepalive_expiry` only helps within a burst: uvicorn closes an
    # idle connection after about five seconds regardless, so a connection never
    # survives the pause while someone types. The saving is real but bounded to
    # the several calls one rerun makes back to back.
    return httpx.Client(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60.0),
    )


st.set_page_config(page_title="RAGuard", page_icon="🛡️", layout="wide")

# The theme itself is pinned in `.streamlit/config.toml`. This sheet only adds
# what Streamlit has no setting for: the brand sidebar, calmer surfaces, and one
# behaviour fix — Streamlit dims every element while a rerun is in flight, which
# reads as a broken page rather than a busy one during a fifteen-second query.
st.markdown(
    """
    <style>
      :root {
        --rg-ink: #14231f;
        --rg-muted: #5c6c68;
        --rg-line: #dfe7e3;
        --rg-brand: #126f63;
        --rg-brand-dark: #0d5a50;
        --rg-sidebar: #12332f;
        --rg-sidebar-soft: #1e4a44;
        --rg-sidebar-line: #356a62;
        --rg-sidebar-ink: #eef5f2;
      }

      /* A query takes seconds, and Streamlit fades the whole page while it
         runs. Keep the previous result readable; the spinner reports progress. */
      [data-stale="true"], .element-container[data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
      }

      .stApp { color: var(--rg-ink); }
      .block-container { padding-top: 2.4rem; max-width: 1180px; }
      .stApp h1 { font-size: 1.95rem; letter-spacing: -0.015em; margin-bottom: 0.1rem; }
      .stApp h2, .stApp h3 { font-size: 1.08rem; letter-spacing: 0.005em; }

      .raguard-kicker {
        color: var(--rg-brand);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
      }
      .raguard-subtitle {
        color: var(--rg-muted);
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
      }

      /* --- Sidebar ------------------------------------------------------ */
      [data-testid="stSidebar"] { background: var(--rg-sidebar); }
      [data-testid="stSidebar"] * { color: var(--rg-sidebar-ink); }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #b7d2cc; }
      [data-testid="stSidebar"] h1 { font-size: 1.4rem; }
      /* Streamlit renamed these wrappers; the input's own background is
         transparent, so the surrounding root element is what has to be dark.
         Styling only the input leaves white text on a white field. */
      [data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
      [data-testid="stSidebar"] [data-testid="stTextAreaRootElement"],
      [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
      [data-testid="stSidebar"] .stButton > button {
        background: var(--rg-sidebar-soft) !important;
        border-color: var(--rg-sidebar-line) !important;
      }
      /* Streamlit 1.62 renders text inputs through BaseWeb's `base-input`
         and `input` containers. The older root-element selector above does
         not reach those nodes, leaving a white field behind white sidebar
         text. Style both generations of markup so the API URL stays visible. */
      [data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"],
      [data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: var(--rg-sidebar-soft) !important;
        border-color: var(--rg-sidebar-line) !important;
        color: var(--rg-sidebar-ink) !important;
        -webkit-text-fill-color: var(--rg-sidebar-ink) !important;
        caret-color: var(--rg-sidebar-ink) !important;
      }
      [data-testid="stSidebar"] input::placeholder { color: #a9c7c1 !important; opacity: 1; }
      /* Status pills are readable on the dark sidebar without relying on the
         alert background Streamlit happens to pick. */
      [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: var(--rg-sidebar-soft);
        border: 1px solid var(--rg-sidebar-line);
        border-left: 3px solid var(--rg-brand);
        border-radius: 6px;
      }
      [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: var(--rg-sidebar-ink) !important;
      }
      [data-testid="stSidebar"] [data-testid="stAlert"] svg { fill: var(--rg-sidebar-ink); }

      /* --- Main panel --------------------------------------------------- */
      [data-testid="stAlert"] { border-radius: 8px; }
      .stApp [data-testid="stAlert"] * { color: var(--rg-ink); }

      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--rg-line);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
      }
      [data-testid="stMetricLabel"] * {
        color: var(--rg-muted) !important;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }
      [data-testid="stMetricValue"] { font-size: 1.5rem; }

      .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--rg-line); }
      .stTabs [data-baseweb="tab"] { padding: 0.4rem 0; color: var(--rg-muted); }
      .stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--rg-brand); }

      /* Scoped to the main panel: the same rule in the sidebar produced a
         white panel holding white text. */
      [data-testid="stMain"] [data-testid="stExpander"] {
        border: 1px solid var(--rg-line);
        border-radius: 8px;
        background: #ffffff;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: var(--rg-sidebar-soft);
        border: 1px solid var(--rg-sidebar-line);
        border-radius: 8px;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] * {
        color: var(--rg-sidebar-ink) !important;
      }
      /* The summary keeps its own near-white background, so the light label
         sitting on it was invisible until the panel was scrolled past. */
      [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: var(--rg-sidebar-soft) !important;
        font-weight: 600;
      }
      [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: #235852 !important;
      }

      /* Status rows: readable on the dark sidebar, unlike the JSON viewer's
         own palette. */
      .rg-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.28rem 0;
        border-bottom: 1px solid rgba(238, 245, 242, 0.12);
        font-size: 0.8rem;
        line-height: 1.35;
      }
      .rg-row:last-child { border-bottom: none; }
      .rg-row__key { color: #b7d2cc; text-transform: capitalize; }
      .rg-row__value {
        color: var(--rg-sidebar-ink);
        font-weight: 600;
        text-align: right;
        word-break: break-word;
      }
      .rg-row--group {
        color: #8fb8b1;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: none;
        padding-top: 0.6rem;
      }
      .rg-row--empty { color: #b7d2cc; border-bottom: none; }

      /* Without this a narrow column breaks a label into one letter per
         line, which looks broken rather than compact. */
      .stButton > button { border-radius: 7px; font-weight: 600; white-space: nowrap; }
      .stButton > button[kind="primary"] {
        background: var(--rg-brand);
        border-color: var(--rg-brand);
      }
      .stButton > button[kind="primary"]:hover {
        background: var(--rg-brand-dark);
        border-color: var(--rg-brand-dark);
      }
      /* Session history reads as a list of links, not a wall of buttons. */
      .stButton > button:not([kind="primary"]) {
        background: #ffffff;
        border: 1px solid var(--rg-line);
        color: var(--rg-ink);
        text-align: left;
        font-weight: 500;
      }
      .stButton > button:not([kind="primary"]):hover {
        border-color: var(--rg-brand);
        color: var(--rg-brand);
      }

      .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: var(--rg-line);
      }
      .stTextArea textarea:focus { border-color: var(--rg-brand); box-shadow: none; }

      .rg-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.9rem 0 0.2rem; }
      .rg-chip {
        background: #eaf3f1;
        border: 1px solid #cfe2dd;
        border-radius: 999px;
        color: var(--rg-brand-dark);
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.18rem 0.62rem;
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


def readiness_gate(status: dict, base_url: str) -> tuple[bool, str]:
    """Decide whether to send, from the probe already taken this rerun.

    The gate exists so a question is never parked inside a model download.
    It used to re-fetch `/ready` here, duplicating the call the sidebar had
    just made and adding a whole round trip between the click and the
    request. The cached probe is at most `STATUS_TTL_S` old, which is
    ample: model loading takes minutes, and the API rejects a query itself
    if it is still initialising.
    """
    if status["ready"] is None:
        return False, f"Could not reach the API at {base_url}."
    if status["ready"]:
        return True, ""
    return False, status["ready_detail"] or "The service is not ready."


def call_api(base_url: str, question: str) -> dict:
    """Post the question, mapping each failure mode to a distinct outcome.

    Every branch returns something renderable. The frontend must never be left
    on a spinner with no explanation, which is what a bare `raise` here would
    produce.
    """
    try:
        response = _client().post(
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

#: Streamlit reruns the whole script on every interaction, so an uncached
#: sidebar re-polled /health, /ready and /config before the main panel could
#: render — three round trips added to every click, including Ask. A short TTL
#: keeps the status honest while making a rerun feel instant.
STATUS_TTL_S = 15


@st.cache_data(ttl=STATUS_TTL_S, show_spinner=False)
def _service_status(base_url: str) -> dict:
    """Health, readiness and configuration in one cached probe.

    Returns only rendered-safe values. A raw exception can carry an internal
    hostname or a credential embedded in a user-edited URL, so nothing beyond
    the exception's presence crosses this boundary.
    """
    status: dict = {"health": None, "ready": None, "ready_detail": "", "checks": {}, "config": None}
    try:
        status["health"] = _client().get(f"{base_url}/health", timeout=10.0).json()
    except Exception:
        status["health"] = None

    try:
        readiness = _client().get(f"{base_url}/ready", timeout=15.0)
        body = readiness.json()
        status["ready"] = readiness.status_code == 200
        status["ready_detail"] = body.get("detail", "Not ready")
        status["checks"] = body.get("checks", {})
    except Exception:
        status["ready"] = None

    try:
        status["config"] = _client().get(f"{base_url}/config", timeout=10.0).json()
    except Exception:
        status["config"] = None
    return status


def _render_detail_rows(data: object, prefix: str = "") -> None:
    """Render a nested mapping as flat, readable rows.

    `st.json` brings its own colour scheme, which sits unreadably on the dark
    sidebar and cannot be restyled reliably across Streamlit versions. These
    rows inherit the sidebar palette instead, and a status panel is read as
    label-and-value anyway rather than as a JSON document.
    """
    if not isinstance(data, dict) or not data:
        st.markdown(
            '<div class="rg-row rg-row--empty">No detail reported.</div>', unsafe_allow_html=True
        )
        return

    for key, value in data.items():
        label = f"{prefix}{key}".replace("_", " ")
        if isinstance(value, dict):
            st.markdown(f'<div class="rg-row rg-row--group">{label}</div>', unsafe_allow_html=True)
            _render_detail_rows(value, prefix="")
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value) or "—"
        rendered = "—" if value is None or value == "" else str(value)
        st.markdown(
            f'<div class="rg-row"><span class="rg-row__key">{label}</span>'
            f'<span class="rg-row__value">{rendered}</span></div>',
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.title("RAGuard")
    st.caption("Self-healing hybrid RAG with citation verification")

    api_url = st.text_input("API base URL", API_BASE_URL)

    st.divider()
    st.subheader("Service status")
    status = _service_status(api_url)

    health = status["health"]
    if health is None:
        st.error("API unreachable.")
    else:
        st.success(f"API {health.get('status', 'unknown')} · v{health.get('version', '?')}")

    if status["ready"] is None:
        st.caption("Readiness unavailable")
    elif status["ready"]:
        st.success("Dependencies ready")
    else:
        st.warning(status["ready_detail"])

    if status["ready"] is not None:
        with st.expander("Readiness detail"):
            _render_detail_rows(status["checks"])

    with st.expander("Active configuration"):
        if status["config"] is None:
            st.caption("Unavailable")
        else:
            _render_detail_rows(status["config"])


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
            ready, reason = readiness_gate(status, api_url)

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

        # The cited passages, named inline. The full text stays one tab away;
        # this is the at-a-glance answer to "what is this resting on?".
        if view.citations:
            chips = "".join(
                f'<span class="rg-chip">{citation["policy_id"]} · {citation["label"]}</span>'
                for citation in view.citations
            )
            st.markdown(f'<div class="rg-chips">{chips}</div>', unsafe_allow_html=True)

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
