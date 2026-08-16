"""Citation verification boundary (Phase F), designed for Phase G to replace.

Phase F does not build a new verifier. One already exists in
`citation_verifier`: it resolves citation labels, splits the answer into
claims, and scores lexical support against the cited passage. Duplicating that
logic here would give the system two disagreeing definitions of "supported".

What Phase F adds is the *seam*. The graph depends on a `Verifier` protocol and
writes its output into `state["verification_result"]` as a `VerificationResult`.
Phase G supplies an LLM-backed entailment verifier satisfying the same
protocol, and no node, edge, or routing function changes: only the object
passed to `build_graph(verifier=...)`.

The `verifier` field on the result records which implementation produced it, so
a trace never leaves you guessing whether a run was checked deterministically
or by a model.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Protocol, runtime_checkable

from src.config import get_settings
from src.generation.answer_chain import validate_citations
from src.retrieval.types import RetrievedChunk
from src.self_healing.citation_verifier import lexical_overlap, verify_citations
from src.self_healing.claims import Claim, extract_claims
from src.self_healing.entailment import judge_claim
from src.self_healing.state import VerificationResult

logger = logging.getLogger(__name__)

__all__ = [
    "DeterministicVerifier",
    "EntailmentVerifier",
    "Verifier",
    "get_default_verifier",
]


@runtime_checkable
class Verifier(Protocol):
    """The contract Phase G must satisfy. Deliberately minimal."""

    name: str

    def verify(
        self, answer: str, citations: list[str], chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        """Decide whether `answer` is supported by the cited `chunks`."""
        ...


class DeterministicVerifier:
    """Phase F default: the existing lexical verifier, adapted to the contract.

    Reproducible and free, which is what lets the graph's verification route be
    exercised in the fast test tier. It under-detects paraphrase, and that is
    the specific weakness Phase G exists to fix.
    """

    name = "deterministic"

    def __init__(self, use_llm: bool = False) -> None:
        # `use_llm` is passed through to the existing verifier's optional
        # entailment step; Phase F leaves it off so the tier stays offline.
        self.use_llm = use_llm

    def verify(
        self, answer: str, citations: list[str], chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        if not answer.strip():
            return VerificationResult(
                supported=False,
                checked=False,
                verifier=self.name,
                reason="no answer text to verify",
            )

        report = verify_citations(answer, citations, chunks, use_llm=self.use_llm)

        reason = "all claims supported by cited passages"
        if report.invalid_labels:
            reason = f"citations not in the supplied passages: {report.invalid_labels}"
        elif report.unsupported_claims:
            reason = f"{len(report.unsupported_claims)} claim(s) not supported by the citation"

        return VerificationResult(
            supported=bool(report.valid),
            checked=True,
            verifier=self.name,
            unsupported_claims=list(report.unsupported_claims),
            invalid_citations=list(report.invalid_labels),
            support_ratio=report.support_ratio,
            reason=reason,
        )


class EntailmentVerifier:
    """Phase G: deterministic gates first, then semantic entailment.

    Order matters, and it is the whole design:

    1. **Citation validity.** Labels are resolved against the passages actually
       supplied, reusing the Phase E validator rather than a second copy of it.
       An invented or partial label fails here and the answer is rejected.
    2. **Claim extraction.** The answer becomes typed claims.
    3. **Exact-evidence gate.** Numbers, amounts, time windows, and identifiers
       in a claim must appear verbatim in its cited passages. This runs *before*
       the model and is not appealable: a fluent judge is exactly the thing that
       would otherwise wave through "3 to 5 business days".
    4. **Entailment.** Only claims that survive the gate are put to the model,
       and only against their own cited passages.

    An unreachable judge yields `checked=False` for that claim and falls back to
    the lexical verdict; it never counts as the model rejecting the claim.
    """

    name = "entailment"

    #: Overlap at which a claim is accepted without consulting the judge. Set
    #: high deliberately: this is a near-verbatim shortcut, not a verdict. The
    #: Phase F threshold (0.25) is far too permissive to *accept* on — a claim
    #: sharing a quarter of its tokens with a passage is exactly the case that
    #: needs semantic judgement, not a bypass of it.
    LEXICAL_ACCEPT_THRESHOLD = 0.85

    def __init__(
        self,
        chain: Any | None = None,
        use_llm: bool | None = None,
        lexical_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.chain = chain
        self.use_llm = settings.graph_use_llm if use_llm is None else use_llm
        # Used only when the judge is unreachable, as the Phase F fallback.
        self.lexical_threshold = (
            settings.citation_support_threshold
            if lexical_threshold is None
            else lexical_threshold
        )

    # -- gates -------------------------------------------------------------

    @staticmethod
    def _token_present(token: str, passage_upper: str) -> bool:
        """Whether a required figure or identifier really appears in the evidence.

        Substring matching is wrong here: "3" occurs inside "30 calendar days",
        which would accept "returns take 3 days" against a 30-day policy. Every
        comparison is therefore boundary-aware, with percentages and money
        normalised so "15 percent" matches "15percent".
        """
        token = token.upper()

        if token.endswith("PERCENT") or token.endswith("%"):
            number = token.removesuffix("%").removesuffix("PERCENT")
            return bool(
                re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:%|PERCENT)", passage_upper)
            )

        if token[:1] in {"£", "$", "€"}:
            number = token[1:]
            return bool(re.search(rf"[£$€]\s*{re.escape(number)}(?![\d.])", passage_upper))

        # Digits and identifiers. The lookahead rejects a match sitting inside a
        # longer number ("3" in "30") or a decimal ("3" in "3.5"), while still
        # allowing an identifier to end a sentence ("rule RT-014.").
        return bool(
            re.search(rf"(?<![\w.]){re.escape(token)}(?!\w)(?!\.\d)", passage_upper)
        )

    def _missing_required(self, claim: Claim, passages: str) -> list[str]:
        """Required figures and identifiers absent from the cited evidence."""
        haystack = passages.upper().replace(",", "")
        return [t for t in claim.required_tokens if not self._token_present(t, haystack)]

    def _lexically_certain(self, claim_text: str, passages: str) -> bool:
        """Near-verbatim restatement, safe to accept without the judge."""
        return lexical_overlap(claim_text, passages) >= self.LEXICAL_ACCEPT_THRESHOLD

    def _lexical_fallback(self, claim_text: str, passages: str) -> bool:
        """Phase F verdict, used only when no judge could be reached."""
        return lexical_overlap(claim_text, passages) >= self.lexical_threshold

    # -- protocol ----------------------------------------------------------

    def verify(
        self, answer: str, citations: list[str], chunks: list[RetrievedChunk]
    ) -> VerificationResult:
        started = time.perf_counter()

        if not answer.strip():
            return VerificationResult(
                supported=False,
                checked=False,
                verifier=self.name,
                reason="no answer text to verify",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        # 1. Deterministic citation validation, reusing Phase E.
        resolved, invalid = validate_citations(citations, chunks)
        cited_chunks = {c.citation_label: c for c in chunks}
        by_label = {c.citation_label: cited_chunks[c.citation_label] for c in resolved}

        if invalid:
            return VerificationResult(
                supported=False,
                checked=True,
                verifier=self.name,
                invalid_citations=list(invalid),
                reason=f"citations not among the supplied passages: {invalid}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        # 2. Claim extraction.
        claims = extract_claims(answer, [c.citation_label for c in resolved])
        if not claims:
            return VerificationResult(
                supported=False,
                checked=True,
                verifier=self.name,
                reason="no checkable claim found in the answer",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        verdicts: list[dict[str, Any]] = []
        unsupported: list[str] = []
        missing_evidence: list[str] = []
        uncited = 0
        confidences: list[float] = []

        for claim in claims:
            passages = "\n\n".join(
                by_label[label].content for label in claim.citation_labels if label in by_label
            )

            if not claim.has_citation or not passages:
                # 7. Citation completeness: an uncited claim is unsupported,
                # however true it may sound.
                uncited += 1
                unsupported.append(claim.claim_text)
                verdicts.append(
                    {**claim.to_dict(), "supported": False, "method": "no-citation",
                     "reason": "claim carries no usable citation"}
                )
                continue

            # 3. Exact-evidence gate, applied before any model call.
            missing = self._missing_required(claim, passages)
            if missing:
                missing_evidence.extend(missing)
                unsupported.append(claim.claim_text)
                verdicts.append(
                    {**claim.to_dict(), "supported": False, "method": "exact-token",
                     "reason": f"not in cited evidence: {missing}"}
                )
                continue

            # Near-verbatim claims skip the judge; everything else is judged.
            supported = self._lexically_certain(claim.claim_text, passages)
            method = "lexical-verbatim"
            confidence = 1.0 if supported else 0.0
            reason = "claim restates the cited passage almost verbatim"

            # 4. Entailment, for every claim the verbatim check did not settle.
            if not supported and self.use_llm:
                verdict = judge_claim(claim.claim_text, passages, chain=self.chain)
                if verdict is not None:
                    method = "entailment"
                    supported = verdict.supported
                    confidence = verdict.confidence
                    reason = verdict.reason or "judged by entailment"
                else:
                    method = "lexical-fallback"
                    supported = self._lexical_fallback(claim.claim_text, passages)
                    reason = "entailment judge unavailable; lexical verdict kept"
            elif not supported:
                method = "lexical-only"
                supported = self._lexical_fallback(claim.claim_text, passages)
                reason = "entailment disabled; lexical verdict kept"

            if not supported:
                unsupported.append(claim.claim_text)

            confidences.append(confidence)
            verdicts.append(
                {**claim.to_dict(), "supported": supported, "method": method,
                 "reason": reason}
            )

        supported_count = sum(1 for v in verdicts if v["supported"])
        all_supported = supported_count == len(verdicts)

        if all_supported:
            reason = f"all {len(verdicts)} claim(s) supported by their cited passages"
        elif uncited:
            reason = f"{uncited} claim(s) carry no citation"
        elif missing_evidence:
            reason = f"figures or identifiers absent from the evidence: {missing_evidence[:3]}"
        else:
            reason = f"{len(unsupported)} of {len(verdicts)} claim(s) not entailed"

        return VerificationResult(
            supported=all_supported,
            checked=True,
            verifier=self.name,
            unsupported_claims=unsupported,
            invalid_citations=[],
            support_ratio=supported_count / len(verdicts),
            confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
            missing_evidence=list(dict.fromkeys(missing_evidence)),
            claims=verdicts,
            claim_count=len(verdicts),
            supported_claim_count=supported_count,
            unsupported_claim_count=len(verdicts) - supported_count,
            uncited_claim_count=uncited,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def get_default_verifier() -> Verifier:
    """The verifier the graph uses unless one is injected.

    Phase G is the default. `RAGUARD_VERIFIER=deterministic` restores the Phase F
    lexical verifier, which is what the offline tiers use when no judge is
    reachable.
    """
    settings = get_settings()
    if settings.verifier_backend == "deterministic":
        return DeterministicVerifier(use_llm=False)
    return EntailmentVerifier()
