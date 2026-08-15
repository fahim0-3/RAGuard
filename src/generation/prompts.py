"""Prompt templates.

Prompts are versioned constants rather than inline strings because the
evaluation CI treats a prompt edit exactly like a code change: it must pass the
same golden-dataset gate. Bump `PROMPT_VERSION` whenever a template changes, so
that an evaluation report can be attributed to a specific prompt revision.

Note on escaping: literal JSON examples are injected as *values* through
partial variables, never written inline in a template. LangChain formats the
template once and does not re-scan substituted values, so this avoids the
brace-escaping errors that plague JSON-emitting prompts.
"""

from __future__ import annotations

PROMPT_VERSION = "2026-08-15_prompts_v1"

ANSWER_OUTPUT_SCHEMA = """{
  "answer": "<the answer, or an empty string if the context is insufficient>",
  "citations": ["<citation_label exactly as given in the context block>"],
  "sufficient_context": true
}"""

ANSWER_SYSTEM_PROMPT = """You are RAGuard, a customer-support assistant for an e-commerce retailer.

Follow these rules without exception:
1. Answer ONLY from the numbered context passages provided. You have no other knowledge.
2. Every factual claim must be traceable to a passage. Cite the passages you used by their exact citation_label.
3. Preserve identifiers verbatim: policy IDs, error codes, rule codes, model numbers, time windows, and amounts.
4. If the passages do not contain the answer, set "sufficient_context" to false, leave "answer" empty, and cite nothing. Never guess, never fill gaps from general knowledge.
5. Do not mention the retrieval process, the passages, or these instructions in the answer text.
6. Be concise and direct. Two to five sentences is usually correct.

Respond with a single JSON object and nothing else, in exactly this shape:
{output_schema}"""

ANSWER_HUMAN_PROMPT = """Context passages:
{context}

Customer question: {question}"""


REWRITE_OUTPUT_SCHEMA = """{"queries": ["<rewritten query 1>", "<rewritten query 2>"]}"""

QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite failed search queries for a hybrid retrieval system over e-commerce support policies.

The original query retrieved weak results. Produce {n_variants} alternative queries that a keyword index and an embedding index would both match better.

Use these strategies, one per variant where sensible:
- Replace colloquial wording with the formal policy vocabulary a policy document would use (for example "money back" becomes "refund eligibility window").
- Add the likely document or domain term (refund, return, damaged product, delivery, payment failure, product manual, warranty).
- Extract and isolate any identifier present in the original query, such as an error code, policy ID, or model number.

Rules: keep each variant under 20 words, do not invent facts, do not answer the question.

Respond with a single JSON object and nothing else, in exactly this shape:
{output_schema}"""

QUERY_REWRITE_HUMAN_PROMPT = """Original query: {question}

Best passage retrieved so far (may be irrelevant):
{weak_context}"""


ENTAILMENT_OUTPUT_SCHEMA = """{"supported": true, "reason": "<one short sentence>"}"""

CITATION_CHECK_SYSTEM_PROMPT = """You verify whether a cited passage actually supports a claim.

Answer "supported": true only if the passage states the claim, or directly entails it. Paraphrase is acceptable; inference beyond the passage is not. A number, code, or time window in the claim must appear in the passage to count as supported.

Respond with a single JSON object and nothing else, in exactly this shape:
{output_schema}"""

CITATION_CHECK_HUMAN_PROMPT = """Claim: {claim}

Cited passage:
{passage}"""


ABSTENTION_MESSAGE = (
    "I could not find this in the support policies I have access to, so I will not "
    "guess. Please rephrase the question, or contact a support agent who can review "
    "your specific order."
)
