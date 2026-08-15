"""Evaluation entry point used by CI and by hand.

    python -m src.evaluation.run_eval --tier retrieval --fail-on-regression
    python -m src.evaluation.run_eval --tier e2e
    python -m src.evaluation.run_eval --tier ragas

Exit code 1 signals a metric regression, which is what turns the GitHub Actions
job red and blocks the merge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.evaluation.metrics import (
    EvaluationReport,
    compare_to_baseline,
    evaluate_end_to_end,
    evaluate_retrieval,
    run_ragas,
)

logger = logging.getLogger(__name__)

_TIERS = {
    "retrieval": ("retrieval", lambda args: evaluate_retrieval(k=args.k)),
    "e2e": ("end_to_end", lambda args: evaluate_end_to_end(use_llm_verification=args.llm_verify)),
    "ragas": ("ragas", lambda args: run_ragas()),
}


def _print_report(report: EvaluationReport) -> None:
    print(f"\n=== RAGuard evaluation: {report.tier} ===")
    for name, value in report.metrics.items():
        print(f"  {name:<28} {value:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGuard evaluation")
    parser.add_argument("--tier", choices=sorted(_TIERS), default="retrieval")
    parser.add_argument("--k", type=int, default=5, help="Cut-off for retrieval metrics")
    parser.add_argument("--llm-verify", action="store_true", help="Enable LLM entailment checks")
    parser.add_argument("--save", action="store_true", help="Write a dated JSON report")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when any metric falls below baseline",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    baseline_section, runner = _TIERS[args.tier]
    report = runner(args)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)

    if args.save:
        path = report.save()
        print(f"\nReport written to {path}")

    regressions = compare_to_baseline(report, baseline_section)
    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for regression in regressions:
            print(f"  - {regression}")
        if args.fail_on_regression:
            print(
                "\nMerge blocked. Either fix the regression, or update "
                "src/evaluation/baseline.json in a separate, justified commit."
            )
            return 1
    else:
        print("\nAll metrics at or above baseline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
