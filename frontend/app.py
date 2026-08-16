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


def call_api(base_url: str, question: str) -> dict:
    """Post the question. Transport failures become an error view, not a crash."""
    try:
        response = httpx.post(
            f"{base_url}/query", json={"query": question}, timeout=REQUEST_TIMEOUT
        )
    except httpx.HTTPError as exc:
        return {"error": "unreachable", "detail": f"Could not reach the API: {exc}"}

    try:
        payload = response.json()
    except ValueError:
        return {"error": "bad_response", "detail": f"API returned status {response.status_code}"}

    if response.status_code >= 400 and "error" not in payload:
        payload = {"error": "http_error", "detail": f"API returned {response.status_code}"}
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
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

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

example = st.selectbox("Example questions", ["(type your own)", *EXAMPLES])
default_text = "" if example == "(type your own)" else example
question = st.text_area("Question", value=default_text, height=90)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving, grading evidence, and verifying citations…"):
        payload = call_api(api_url, question.strip())
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
