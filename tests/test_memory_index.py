"""In-memory dense retrieval: exact pgvector semantics, correct dispatch, and a
refresh path that never leaves the API serving a stale corpus."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import memory_index
from src.retrieval.memory_index import InMemoryDenseIndex
from src.retrieval.types import RetrievedChunk


def chunk(chunk_id: int, **overrides) -> RetrievedChunk:
    values = {
        "chunk_id": chunk_id,
        "content": f"policy text {chunk_id}",
        "source": f"policy-{chunk_id}.txt",
        "chunk_index": chunk_id,
        "doc_id": f"POL-{chunk_id:03d}",
        "metadata": {"section": f"S{chunk_id}"},
    }
    values.update(overrides)
    return RetrievedChunk(**values)


@pytest.fixture(autouse=True)
def _clean_index():
    memory_index.reset_memory_index()
    yield
    memory_index.reset_memory_index()


# --------------------------------------------------------------------------
# Search semantics
# --------------------------------------------------------------------------


def test_scores_are_pgvector_cosine_similarity():
    """pgvector returns 1 - (a <=> b); for the cosine operator that is cosine."""
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(5, 16)).astype(np.float32)
    query = rng.normal(size=16).astype(np.float32)
    index = InMemoryDenseIndex([chunk(i) for i in range(5)], vectors)

    hits = index.search(query, top_k=5)

    expected = {
        i: float(vectors[i] @ query / (np.linalg.norm(vectors[i]) * np.linalg.norm(query)))
        for i in range(5)
    }
    for hit in hits:
        assert hit.dense_score == pytest.approx(expected[hit.chunk_id], abs=1e-5)
    assert [h.chunk_id for h in hits] == sorted(expected, key=expected.get, reverse=True)


def test_top_k_is_respected():
    index = InMemoryDenseIndex([chunk(i) for i in range(4)], np.eye(4, dtype=np.float32))

    assert len(index.search([1.0, 0.0, 0.0, 0.0], top_k=2)) == 2


def test_chunk_metadata_is_preserved_and_the_snapshot_is_not_mutated():
    original = chunk(3, metadata={"section": "Refund processing times"})
    index = InMemoryDenseIndex([original], np.ones((1, 4), dtype=np.float32))

    hit = index.search([1.0, 1.0, 1.0, 1.0], top_k=1)[0]

    assert (hit.chunk_id, hit.source, hit.doc_id, hit.chunk_index) == (
        3,
        "policy-3.txt",
        "POL-003",
        3,
    )
    assert hit.metadata == {"section": "Refund processing times"}
    assert original.dense_score is None, "scores go on a copy, never the shared snapshot"


def test_equal_scores_keep_the_database_fetch_order():
    index = InMemoryDenseIndex([chunk(9), chunk(2), chunk(5)], np.ones((3, 2), dtype=np.float32))

    assert [h.chunk_id for h in index.search([1.0, 1.0], top_k=3)] == [9, 2, 5]


def test_a_zero_query_vector_returns_nothing_rather_than_dividing_by_zero():
    index = InMemoryDenseIndex([chunk(1)], np.ones((1, 2), dtype=np.float32))

    assert index.search([0.0, 0.0], top_k=1) == []


def test_mismatched_chunks_and_vectors_are_rejected():
    with pytest.raises(ValueError):
        InMemoryDenseIndex([chunk(1), chunk(2)], np.ones((1, 2), dtype=np.float32))


# --------------------------------------------------------------------------
# Dispatch: memory when built, pgvector otherwise
# --------------------------------------------------------------------------


def _pgvector_spy(monkeypatch) -> list[int]:
    from src.retrieval import vector_store

    calls: list[int] = []

    def fake(embedding, top_k):
        calls.append(top_k)
        return [chunk(99)]

    monkeypatch.setattr(vector_store, "dense_search", fake)
    return calls


def test_a_built_index_serves_the_query_without_the_database(monkeypatch):
    calls = _pgvector_spy(monkeypatch)
    memory_index._index = InMemoryDenseIndex([chunk(1)], np.ones((1, 2), dtype=np.float32))

    hits = memory_index.dense_search([1.0, 1.0], 5)

    assert [h.chunk_id for h in hits] == [1]
    assert calls == [], "no database round trip on the normal path"


def test_an_unbuilt_index_falls_back_to_pgvector(monkeypatch):
    calls = _pgvector_spy(monkeypatch)

    hits = memory_index.dense_search([1.0, 1.0], 5)

    assert [h.chunk_id for h in hits] == [99]
    assert calls == [5]


def test_the_pgvector_backend_ignores_any_resident_index(monkeypatch):
    from src.config import get_settings

    calls = _pgvector_spy(monkeypatch)
    memory_index._index = InMemoryDenseIndex([chunk(1)], np.ones((1, 2), dtype=np.float32))
    monkeypatch.setenv("DENSE_INDEX_BACKEND", "pgvector")
    get_settings.cache_clear()
    try:
        memory_index.dense_search([1.0, 1.0], 5)
    finally:
        get_settings.cache_clear()

    assert calls == [5]


# --------------------------------------------------------------------------
# Refresh: one snapshot for both indexes, rebuilt when the table changes
# --------------------------------------------------------------------------


class FakeTable:
    """Stands in for the chunks table: rows plus a fingerprint derived from them."""

    def __init__(self, rows: list[tuple[RetrievedChunk, list[float]]]):
        self.rows = rows
        self.fetches = 0

    def fingerprint(self) -> str:
        return repr([(c.chunk_id, c.content, v) for c, v in self.rows])

    def fetch(self):
        self.fetches += 1
        return [c for c, _ in self.rows], np.asarray([v for _, v in self.rows], dtype=np.float32)


@pytest.fixture
def table(monkeypatch):
    from src.retrieval import bm25, vector_store

    # BM25 gives a term present in every document zero weight, so the fake
    # corpus needs a few unrelated rows for sparse search to score anything.
    fake = FakeTable(
        [
            (chunk(1, content="card refunds take five days"), [1.0, 0.0]),
            (chunk(3, content="parcels are declared lost after ten days"), [0.6, 0.8]),
            (chunk(4, content="gateway code means a duplicate payment"), [0.8, 0.6]),
        ]
    )
    monkeypatch.setattr(vector_store, "corpus_fingerprint", fake.fingerprint)
    monkeypatch.setattr(vector_store, "fetch_corpus_with_embeddings", fake.fetch)
    monkeypatch.setattr(bm25, "_index", None)
    return fake


def test_one_snapshot_builds_both_bm25_and_the_dense_index(table):
    from src.retrieval.bm25 import get_bm25_index

    result = memory_index.refresh_corpus_indexes()

    assert result == {"bm25_documents": 3, "dense_in_memory": 3}
    assert table.fetches == 1, "one fetch, shared by both indexes"
    assert get_bm25_index().search("refunds", 5)[0].chunk_id == 1
    assert memory_index.dense_search([1.0, 0.0], 5)[0].chunk_id == 1


def test_an_unchanged_table_is_not_rebuilt(table):
    memory_index.refresh_corpus_indexes()

    assert memory_index._refresh_if_changed() is False
    assert table.fetches == 1


def test_an_added_chunk_is_served_after_the_next_freshness_check(table):
    """The stale-index case: ingestion in another process adds a chunk."""
    from src.retrieval.bm25 import get_bm25_index

    memory_index.refresh_corpus_indexes()
    table.rows.append((chunk(2, content="electronics return window fourteen days"), [0.0, 1.0]))

    assert memory_index._refresh_if_changed() is True
    assert [c.chunk_id for c in memory_index.dense_search([0.0, 1.0], 1)] == [2]
    assert get_bm25_index().search("electronics", 5)[0].chunk_id == 2


def test_an_updated_chunk_replaces_its_old_content_and_vector(table):
    memory_index.refresh_corpus_indexes()
    table.rows[0] = (chunk(1, content="card refunds take seven days"), [0.0, 1.0])

    memory_index._refresh_if_changed()

    hit = memory_index.dense_search([0.0, 1.0], 1)[0]
    assert hit.content == "card refunds take seven days"
    assert hit.dense_score == pytest.approx(1.0)


def test_a_failed_freshness_check_keeps_serving_the_last_good_index(table, monkeypatch):
    from src.retrieval import vector_store

    memory_index.refresh_corpus_indexes()

    def database_down():
        raise ConnectionError("down")

    monkeypatch.setattr(vector_store, "corpus_fingerprint", database_down)

    # The check itself propagates; the refresher loop logs it and keeps going.
    with pytest.raises(ConnectionError):
        memory_index._refresh_if_changed()

    assert memory_index.dense_search([1.0, 0.0], 1)[0].chunk_id == 1


def test_a_corpus_above_the_limit_stays_on_pgvector(table, monkeypatch):
    monkeypatch.setattr(memory_index, "MAX_IN_MEMORY_CHUNKS", 0)

    result = memory_index.refresh_corpus_indexes()

    assert result["dense_in_memory"] == 0
    assert memory_index.is_memory_index_built() is False


# --------------------------------------------------------------------------
# Live check: memory and pgvector agree on the real corpus
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.heavy
def test_memory_and_pgvector_return_the_same_top_k_on_the_live_corpus():
    from src.retrieval import vector_store
    from src.retrieval.embeddings import embed_query

    memory_index.refresh_corpus_indexes()
    questions = [
        "What is the refund policy?",
        "How long do card refunds take?",
        "When is a domestic parcel declared lost?",
        "What does gateway code PAY-409 mean?",
        "How long do I have to report concealed damage?",
    ]
    for question in questions:
        embedding = embed_query(question)
        memory_hits = memory_index.dense_search(embedding, 20)
        database_hits = vector_store.dense_search(embedding, 20)

        assert [h.chunk_id for h in memory_hits] == [h.chunk_id for h in database_hits], question
        for mine, theirs in zip(memory_hits, database_hits, strict=True):
            assert mine.dense_score == pytest.approx(theirs.dense_score, abs=1e-4)
            assert (mine.source, mine.doc_id, mine.metadata) == (
                theirs.source,
                theirs.doc_id,
                theirs.metadata,
            )
