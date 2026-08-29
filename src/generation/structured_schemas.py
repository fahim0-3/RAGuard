"""Closed JSON schemas for Groq's native structured-output API.

All fields are required and every object is closed because Groq strict mode
requires that shape. Values that are optional in an internal Pydantic model are
represented with a JSON ``null`` union instead of omitting a key.
"""

from __future__ import annotations

ANSWER_SCHEMA = {
    "title": "GroundedAnswer",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "claim_citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "citations"],
            },
        },
        "sufficient_context": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "claim_citations", "sufficient_context", "confidence"],
}

EVIDENCE_GRADE_SCHEMA = {
    "title": "EvidenceGrade",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevant": {"type": "boolean"},
        "sufficient": {"type": "boolean"},
        "confidence": {"type": "number"},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "relevant",
        "sufficient",
        "confidence",
        "missing_information",
        "rationale",
    ],
}

REWRITE_SCHEMA = {
    "title": "QueryRewrites",
    "type": "object",
    "additionalProperties": False,
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

ENTAILMENT_SCHEMA = {
    "title": "EntailmentVerdict",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "supported": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "confidence", "reason"],
}

CITATION_CHECK_SCHEMA = {
    "title": "CitationCheck",
    "type": "object",
    "additionalProperties": False,
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}
