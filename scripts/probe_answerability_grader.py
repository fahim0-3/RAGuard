"""Probe the configured answerability grader without printing policy text or secrets.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\probe_answerability_grader.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.generation.llm_factory import provider_config  # noqa: E402
from src.generation.llm_routing import route_context  # noqa: E402
from src.reranking import get_reranker  # noqa: E402
from src.retrieval.hybrid import get_hybrid_retriever  # noqa: E402
from src.self_healing.evidence_grader import grade_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="What is the refund policy?")
    args = parser.parse_args()

    settings = get_settings()
    with route_context(settings) as route:
        query = args.query
        chunks = get_reranker().rerank(query, get_hybrid_retriever().retrieve(query))
        grade = grade_evidence(query, chunks)
        answerability = grade.signals.get("answerability") or {}
        print(
            json.dumps(
                {
                    "provider": provider_config("judge")["provider"],
                    "model": provider_config("judge")["model"],
                    "routing_mode": route.mode,
                    "fallbacks": list(route.fallback_reasons),
                    "sufficient": grade.sufficient,
                    "confidence": grade.confidence,
                    "failure_category": grade.failure_category or None,
                    "failure_reason": grade.failure_reason or None,
                    "failure_phase": grade.failure_phase or None,
                    "failure_exception_type": grade.failure_exception_type or None,
                    "answerability": answerability,
                },
                indent=2,
            )
        )
    return 0 if not grade.failure_category else 1


if __name__ == "__main__":
    raise SystemExit(main())
