"""Streamlit demonstration UI.

The interface is built around the guard rails rather than the chat bubble. A
plain RAG demo shows an answer; this one shows why the answer was allowed
through, or why it was refused: confidence scores, the rewritten queries the
healing loop tried, per-claim citation verdicts, and the full decision trace.

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

st.set_page_config(page_title="RAGuard", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("RAGuard")
    st.caption("Self-healing hybrid RAG with citation verification")

    api_url = st.text_input("API base URL", API_BASE_URL)
    use_llm_rewrite = st.toggle("LLM query rewriting", value=True)
    use_llm_verification = st.toggle(
        "LLM citation entailment", value=False, help="Slower, non-deterministic, more thorough"
    )

    st.divider()
    st.subheader("Service status")
    try:
        health = httpx.get(f"{api_url}/health", timeout=10.0).json()
        st.success(f"{health['status']} · {health.get('chunks_indexed', 0)} chunks")
        st.caption(f"Provider: {health.get('llm_provider', 'unknown')}")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

    with st.expander("Active configuration"):
        try:
            st.json(httpx.get(f"{api_url}/config", timeout=10.0).json())
        except Exception:
            st.caption("Unavailable")


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

st.header("Ask the support assistant")

EXAMPLES = [
    "How long does a refund take to reach my credit card?",
    "What does error PAY-402 mean at checkout?",
    "My parcel never turned up. When do you count it as actually lost?",
    "My AuraBrew X200 is showing E04, what should I do?",
    "Can I get a mortgage or a personal loan through your store?",
]

example = st.selectbox("Example questions", ["—", *EXAMPLES])
question = st.text_area(
    "Question",
    value="" if example == "—" else example,
    height=90,
    placeholder="Ask about refunds, returns, damage, delivery, payments, or the product manual",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving, verifying, and deciding..."):
        try:
            response = httpx.post(
                f"{api_url}/query",
                json={
                    "question": question,
                    "use_llm_rewrite": use_llm_rewrite,
                    "use_llm_verification": use_llm_verification,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    # --- Verdict ---------------------------------------------------------
    if result["abstained"]:
        st.warning("**Abstained.** " + result["answer"])
        st.caption(f"Reason: `{result['abstain_reason']}`")
    else:
        st.success(result["answer"])

    # --- Guard-rail metrics ---------------------------------------------
    confidence = result.get("confidence", {})
    citation_report = result.get("citation_report", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top confidence", f"{confidence.get('top_score', 0):.2f}", confidence.get("level"))
    col2.metric("Score margin", f"{confidence.get('margin', 0):.2f}")
    col3.metric("Healing retries", result.get("healing_attempts", 0))
    col4.metric("Latency", f"{result.get('latency_ms', 0) / 1000:.1f}s")

    # --- Citations -------------------------------------------------------
    st.subheader("Citations")
    citations = result.get("citations", [])
    if not citations:
        st.caption("No citations. The system did not release an answer.")
    for citation in citations:
        with st.expander(f"{citation['citation_label']} · rerank {citation.get('rerank_score', 0):.2f}"):
            st.markdown(f"**Source:** `{citation['source']}`")
            st.text(citation["content"])

    # --- Verification detail --------------------------------------------
    if citation_report:
        st.subheader("Claim verification")
        claims = citation_report.get("claims", [])
        if claims:
            st.dataframe(
                [
                    {
                        "Claim": c["claim"],
                        "Supported": "yes" if c["supported"] else "no",
                        "Overlap": c["overlap"],
                        "Missing facts": ", ".join(c["missing_facts"]) or "—",
                        "Method": c["method"],
                    }
                    for c in claims
                ],
                use_container_width=True,
                hide_index=True,
            )
        if citation_report.get("invalid_labels"):
            st.error(f"Invented citation labels: {citation_report['invalid_labels']}")

    # --- Healing and trace ----------------------------------------------
    if result.get("rewritten_queries"):
        st.subheader("Self-healing rewrites")
        for variant in result["rewritten_queries"]:
            st.markdown(f"- `{variant}`")

    with st.expander("Decision trace (raw)"):
        st.code(json.dumps(result.get("trace", []), indent=2), language="json")


st.divider()
st.caption(
    "RAGuard abstains rather than guessing. An abstention is a correct outcome "
    "when the corpus does not contain the answer."
)
