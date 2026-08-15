"""Manual retrieval inspection over a fixed set of probe queries.

Each probe targets a different retrieval failure mode:

1. plain policy lookup
2. colloquial phrasing that does not match policy vocabulary
3. exact document identifier, which dense retrieval alone tends to miss
4. symptom described without any policy vocabulary at all
5. cross-document temptation between the damage and return policies

Run:  python scripts/test_retrieval.py
      python scripts/test_retrieval.py --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow execution as a plain script from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import get_hybrid_retriever  # noqa: E402
from src.retrieval.types import RetrievedChunk  # noqa: E402

PROBE_QUERIES = [
    "What is the refund window for items?",
    "How long do I have to return something?",
    "What does policy REF-001 say?",
    "My payment was deducted but my order was not created.",
    "Can I get a replacement for a damaged product?",
]

PREVIEW_CHARS = 250


def _fmt(value: float | None, spec: str = "8.4f") -> str:
    return format(value, spec) if value is not None else "     n/a"


def print_result(rank: int, chunk: RetrievedChunk) -> None:
    preview = chunk.content[:PREVIEW_CHARS].replace("\n", " ")
    print(f"  [{rank:>2}] policy={chunk.policy_id:<10} chunk_id={chunk.chunk_id:<4} "
          f"source={chunk.source}")
    print(f"       bm25={_fmt(chunk.sparse_score)}  "
          f"vector={_fmt(chunk.dense_score)}  "
          f"rrf={_fmt(chunk.fusion_score, '10.6f')}  "
          f"ranks={chunk.retriever_ranks or '{}'}")
    print(f"       {preview}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect hybrid retrieval output")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Results to print per query (default: the full fused result set)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress library logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    retriever = get_hybrid_retriever()
    print("Configuration:")
    config = retriever.config()
    print(f"  dense_top_k={config['retrieval']['dense_top_k']}  "
          f"sparse_top_k={config['retrieval']['sparse_top_k']}  "
          f"final_top_k={config['retrieval']['final_top_k']}  "
          f"rrf_k={config['rrf']['k']}")
    print(f"  bm25: k1={config['bm25']['k1']} b={config['bm25']['b']} "
          f"corpus={config['bm25']['corpus_size']} chunks")
    print(f"  dedup: enabled={config['deduplication']['enabled']} "
          f"near={config['deduplication']['near_duplicate_threshold']} "
          f"adjacent={config['deduplication']['adjacent_threshold']} "
          f"max_run={config['deduplication']['max_adjacent_run']}")

    failures = 0
    for number, query in enumerate(PROBE_QUERIES, start=1):
        print("\n" + "=" * 100)
        print(f"QUERY {number}: {query}")
        print("=" * 100)

        diagnostics = retriever.retrieve_with_diagnostics(query)
        dropped = diagnostics.deduplication.dropped if diagnostics.deduplication else []
        print(f"  dense hits={len(diagnostics.dense_hits)}  "
              f"bm25 hits={len(diagnostics.sparse_hits)}  "
              f"fused={len(diagnostics.fused)}  "
              f"deduplicated out={len(dropped)}  "
              f"returned={len(diagnostics.results)}\n")

        if not diagnostics.results:
            failures += 1
            print("  NO RESULTS")
            continue

        for rank, chunk in enumerate(diagnostics.results[: args.top_k], start=1):
            print_result(rank, chunk)

        if dropped:
            print("  removed by deduplication:")
            for record in dropped:
                print(f"    chunk_id={record.chunk_id} source={record.source} "
                      f"reason={record.reason} similar_to={record.similar_to} "
                      f"similarity={record.similarity}")

    print("\n" + "=" * 100)
    print(f"Queries run: {len(PROBE_QUERIES)}   Queries returning nothing: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
