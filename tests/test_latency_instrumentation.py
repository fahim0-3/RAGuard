"""Contracts for retry-safe, privacy-safe latency instrumentation."""

from __future__ import annotations

from types import SimpleNamespace

from src.evaluation.golden_eval import run_golden_evaluation
from src.retrieval.types import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        content="Card refunds take 5 to 7 business days.",
        source="refund_policy.txt",
        chunk_index=1,
        doc_id="REF-001",
    )


class DiagnosticRetriever:
    def retrieve(self, query, top_k=None):
        return [_chunk()]

    def retrieve_with_diagnostics(self, query, top_k=None):
        return SimpleNamespace(
            results=[_chunk()],
            timings_ms={
                "query_embedding": 10.0,
                "vector_search": 20.0,
                "bm25_search": 2.0,
                "rrf_fusion": 1.0,
                "deduplication": 0.5,
            },
        )

    def config(self):
        return {"retrieval": {"reranking_applied": False}}


def test_golden_report_preserves_total_latency_and_adds_component_latency(monkeypatch):
    monkeypatch.setattr("src.retrieval.vector_store.count_chunks", lambda: 1)
    monkeypatch.setattr(
        "src.retrieval.vector_store.source_policy_ids",
        lambda: {"refund_policy.txt": "REF-001"},
    )
    case = {
        "id": "LAT-001",
        "question": "How long do card refunds take?",
        "case_type": "normal",
        "category": "refund",
        "priority": "P1",
        "expected_outcome": "answer",
        "expected_sources": ["refund_policy.txt"],
        "expected_policy_ids": ["REF-001"],
        "expected_keywords": ["5 to 7"],
        "should_abstain": False,
    }

    report = run_golden_evaluation(cases=[case], retriever=DiagnosticRetriever())

    assert set(("count", "mean_ms", "p50_ms", "p95_ms", "max_ms")) <= set(report["latency"])
    assert report["latency"]["by_stage"]["vector_search"] == {
        "count": 1,
        "mean_ms": 20.0,
        "p50_ms": 20.0,
        "p95_ms": 20.0,
        "max_ms": 20.0,
    }
    assert report["cases"][0]["retrieval_latency_ms"]["query_embedding"] == 10.0
