"""Risk routing: which questions must leave the automated path.

Some questions are wrong to answer well. A duplicate-charge report is a
potential fraud case, an account-takeover report is a security incident, and a
message mentioning self-harm is not a support ticket. Retrieval would happily
return a plausible policy passage for all three, and that fluent answer is the
harm: it delays the human who should have seen it.

Routing therefore happens *before* retrieval and generation, and the escalation
message deliberately promises nothing. It states that a person will take over
and, where the policy actually says so, what the customer should have ready.
No compensation, no timeline, no entitlement is invented here.

Deterministic by design. A model deciding whether a self-harm disclosure gets
escalated is a model that can be talked out of it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.self_healing.state import RiskAssessment

logger = logging.getLogger(__name__)

__all__ = ["RiskRule", "assess_risk", "escalation_message"]


@dataclass(frozen=True)
class RiskRule:
    category: str
    pattern: re.Pattern[str]
    reason: str
    #: Shown to the customer. Must contain no policy promise.
    response: str


#: Ordered by severity: welfare first, then security, then money.
RISK_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        category="self_harm_or_threat",
        pattern=re.compile(
            r"\b(kill myself|end my life|suicid\w*|self[- ]harm|hurt myself|"
            r"i'?ll kill you|kill you|shoot you|burn (down )?your|"
            r"come to your office|find out where you live)\b",
            re.IGNORECASE,
        ),
        reason="message contains a welfare or safety concern",
        response=(
            "I am not able to help with this here, and I do not want to respond to "
            "something this serious with an automated message. I am passing this to a "
            "member of our team now. If you are in immediate danger, please contact "
            "your local emergency services."
        ),
    ),
    RiskRule(
        category="account_security",
        pattern=re.compile(
            r"\b(hacked|account (was )?(taken over|compromised|breached)|"
            r"someone (else )?(has|got) (access|into) my account|"
            r"unauthorised (access|login)|unauthorized (access|login)|"
            r"password (was )?changed|can'?t log ?in anymore|identity theft)\b",
            re.IGNORECASE,
        ),
        reason="possible account compromise",
        response=(
            "This looks like an account security issue, so I am handing it to our team "
            "rather than answering from the general policies. They will contact you "
            "about securing the account. Please do not share your password with anyone, "
            "including us."
        ),
    ),
    RiskRule(
        category="payment_fraud_or_duplicate_charge",
        pattern=re.compile(
            r"\b(fraud\w*|scam\w*|stolen card|card (was )?stolen|"
            r"charged twice|double charge[d]?|duplicate charge|"
            r"charge[d]? (me )?(twice|two times)|"
            r"(payment|money) (taken|deducted|charged) (but|and) no order|"
            r"charge (with|and) no order|didn'?t authorise|didn'?t authorize)\b",
            re.IGNORECASE,
        ),
        reason="suspected fraudulent or duplicate charge",
        response=(
            "A charge you did not expect needs a person to look at the payment record, "
            "so I am escalating this rather than answering from the general policies. "
            "Please have the gateway error code and the last four digits of the card "
            "ready. Do not send the full card number."
        ),
    ),
    RiskRule(
        category="legal_or_compliance",
        pattern=re.compile(
            r"\b(sue|suing|lawsuit|legal action|solicitor|attorney|lawyer|"
            r"small claims|trading standards|ombudsman|gdpr|data protection|"
            r"subject access request|delete (all )?my data|regulatory complaint)\b",
            re.IGNORECASE,
        ),
        reason="legal or compliance-sensitive request",
        response=(
            "This raises a legal or data-protection point that I should not answer from "
            "the support policies, so I am passing it to the team who handle those "
            "requests. They will respond to you directly."
        ),
    ),
    RiskRule(
        category="high_value_damage_claim",
        pattern=re.compile(
            r"(?:£|\$|eur|gbp|usd)\s?(?:[1-9]\d{3,}|[1-9]\d{0,2}[,.]\d{3})"
            r"|\b(?:[1-9]\d{3,})\s?(?:pounds|gbp|dollars|euros)\b",
            re.IGNORECASE,
        ),
        reason="high-value claim requires manual approval",
        response=(
            "A claim of this value needs manual approval, so I am passing it to an agent "
            "rather than answering from the general policies. Please have your order "
            "number and photographs of the damage ready."
        ),
    ),
)


def assess_risk(question: str) -> RiskAssessment:
    """Classify the question. First matching rule wins, severity-ordered."""
    text = question or ""

    for rule in RISK_RULES:
        if rule.pattern.search(text):
            logger.info("Risk rule matched: %s", rule.category)
            return RiskAssessment(level="high", category=rule.category, reason=rule.reason)

    return RiskAssessment(level="none", category="", reason="no risk pattern matched")


def escalation_message(assessment: RiskAssessment) -> str:
    """The customer-facing text. Contains no policy promise and no reasoning."""
    for rule in RISK_RULES:
        if rule.category == assessment.category:
            return rule.response
    return "I am passing this to a member of our team, who will follow up with you directly."
