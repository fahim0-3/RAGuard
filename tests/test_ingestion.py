"""Phase D: chunking and metadata extraction.

Retrieval can only find what ingestion preserved. Every one of these tests runs
without a database and without a model, because a chunking regression should be
caught in the fast tier rather than by a mysterious drop in HitRate three stages
downstream.

The properties that matter:

- identifiers survive chunking intact, since exact-term retrieval depends on them
- the document ID is extracted, since every citation is anchored to it
- section breadcrumbs are attached, since they are what makes a chunk readable
  out of context
- chunk indices are contiguous from zero, since the golden dataset pins verified
  answers to specific indices
"""

from __future__ import annotations

import pytest

import src.ingestion.ingest as ingestion
from src.config import PROJECT_ROOT, get_settings
from src.ingestion.ingest import _build_splitter, _extract_headings, _split_document

DATA_DIR = PROJECT_ROOT / "data" / "policies"

CORPUS_FILES = sorted(p.name for p in DATA_DIR.glob("*.txt"))

# Identifiers that exact-term golden cases depend on, and the file holding each.
CRITICAL_IDENTIFIERS = [
    ("refund_policy.txt", "RF-101"),
    ("refund_policy.txt", "WINDOW_EXPIRED"),
    ("return_policy.txt", "RT-014"),
    ("return_policy.txt", "RT-REJ-02"),
    ("damaged_product_policy.txt", "DMG-DEREG-PENDING"),
    ("delivery_policy.txt", "DEL-INV-03"),
    ("payment_failure_faq.txt", "PAY-402"),
    ("payment_failure_faq.txt", "PAY-BLK-03"),
    ("product_manual_example.txt", "AB-X200-EU"),
    ("product_manual_example.txt", "E04"),
]


# --------------------------------------------------------------------------
# Heading and metadata extraction
# --------------------------------------------------------------------------


def test_extract_headings_finds_title_and_section():
    title, section = _extract_headings(
        "# Refund Policy (REF-001)\n\n## 1. Refund eligibility\ntext"
    )
    assert title == "Refund Policy (REF-001)"
    assert section == "1. Refund eligibility"


def test_extract_headings_returns_empty_when_absent():
    assert _extract_headings("plain text with no headings") == ("", "")


def test_extract_headings_keeps_the_first_title_only():
    title, _ = _extract_headings("# First\n# Second")
    assert title == "First"


def test_extract_headings_ignores_hash_inside_prose():
    _, section = _extract_headings("The code #4 is not a heading")
    assert section == ""


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_every_document_yields_a_doc_id(filename):
    records = _split_document(DATA_DIR / filename)
    assert records, f"{filename} produced no chunks"
    doc_ids = {r["doc_id"] for r in records}
    assert len(doc_ids) == 1, f"{filename} produced inconsistent doc_ids: {doc_ids}"
    assert doc_ids != {""}


def test_doc_id_is_read_from_the_document_header():
    records = _split_document(DATA_DIR / "refund_policy.txt")
    assert records[0]["doc_id"] == "REF-001"


def test_doc_id_falls_back_to_the_filename(tmp_path):
    """A document with no Document ID line must still be citable."""
    path = tmp_path / "orphan_policy.txt"
    path.write_text("# Orphan Policy\n\n## 1. Rule\nSomething applies.", encoding="utf-8")

    records = _split_document(path)

    assert records[0]["doc_id"] == "orphan_policy"


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_metadata_carries_title_and_section(filename):
    records = _split_document(DATA_DIR / filename)
    for record in records:
        assert record["metadata"]["doc_title"], f"{filename} chunk lost its title"
        assert record["metadata"]["char_count"] > 0


def test_breadcrumb_is_prefixed_to_content():
    records = _split_document(DATA_DIR / "refund_policy.txt")
    assert records[0]["content"].startswith("[Refund Policy (REF-001)")


def test_section_persists_into_chunks_that_do_not_repeat_the_heading():
    """A chunk split mid-section must still know which section it came from."""
    records = _split_document(DATA_DIR / "payment_failure_faq.txt")
    assert all(r["metadata"]["section"] for r in records[1:]), (
        "chunks after the first must inherit a section breadcrumb"
    )


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_chunk_indices_are_contiguous_from_zero(filename):
    """The golden dataset pins verified answers to chunk indices."""
    records = _split_document(DATA_DIR / filename)
    assert [r["chunk_index"] for r in records] == list(range(len(records)))


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_source_is_the_filename(filename):
    records = _split_document(DATA_DIR / filename)
    assert {r["source"] for r in records} == {filename}


@pytest.mark.parametrize(("filename", "identifier"), CRITICAL_IDENTIFIERS)
def test_identifiers_survive_chunking_intact(filename, identifier):
    """An identifier split across two chunks is unfindable by BM25 or by a human."""
    records = _split_document(DATA_DIR / filename)
    assert any(identifier in r["content"] for r in records), (
        f"{identifier} did not survive chunking of {filename}"
    )


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_no_chunk_is_empty(filename):
    records = _split_document(DATA_DIR / filename)
    for record in records:
        body = record["content"].split("]\n", 1)[-1]
        assert body.strip(), f"{filename} chunk {record['chunk_index']} has no body"


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_chunks_respect_the_configured_size_budget(filename):
    """Content may exceed chunk_size only by the breadcrumb prefix."""
    settings = get_settings()
    records = _split_document(DATA_DIR / filename)
    for record in records:
        assert record["metadata"]["char_count"] <= settings.chunk_size, (
            f"{filename} chunk {record['chunk_index']} is "
            f"{record['metadata']['char_count']} chars, over the "
            f"{settings.chunk_size} budget"
        )


@pytest.mark.parametrize("filename", CORPUS_FILES)
def test_chunking_is_deterministic(filename):
    """Two runs must produce byte-identical chunks, or metrics drift for free."""
    first = _split_document(DATA_DIR / filename)
    second = _split_document(DATA_DIR / filename)
    assert [r["content"] for r in first] == [r["content"] for r in second]


def test_splitter_uses_configured_size_and_overlap():
    settings = get_settings()
    splitter = _build_splitter()
    assert splitter._chunk_size == settings.chunk_size
    assert splitter._chunk_overlap == settings.chunk_overlap


def test_whole_corpus_text_is_preserved_in_substance():
    """Chunking may add breadcrumbs and drop whitespace, never sentences."""
    for filename in CORPUS_FILES:
        raw = (DATA_DIR / filename).read_text(encoding="utf-8")
        joined = " ".join(r["content"] for r in _split_document(DATA_DIR / filename))
        for line in raw.splitlines():
            stripped = line.strip()
            if len(stripped) < 25 or stripped.startswith("#"):
                continue
            assert stripped in joined, f"{filename} lost content: {stripped[:60]!r}"


def test_reset_ingestion_replaces_a_source_through_one_atomic_operation(tmp_path, monkeypatch):
    path = tmp_path / "policy.txt"
    path.write_text("ignored by the patched splitter", encoding="utf-8")
    records = [
        {
            "source": path.name,
            "doc_id": "POL-001",
            "chunk_index": 0,
            "content": "Policy text",
            "metadata": {},
        }
    ]
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(ingestion, "_split_document", lambda _path: records)
    monkeypatch.setattr(ingestion, "embed_texts", lambda _texts: [[0.0] * 3])
    monkeypatch.setattr(
        ingestion,
        "insert_chunks",
        lambda _records: (_ for _ in ()).throw(
            AssertionError("reset must not commit a separate insert")
        ),
    )

    def replace(source, embedded_records):
        calls.append((source, embedded_records))
        return 2, len(embedded_records)

    monkeypatch.setattr(ingestion, "replace_source_chunks", replace, raising=False)

    written = ingestion.ingest_file(path, reset=True)

    assert written == 1
    assert calls == [(path.name, records)]
    assert records[0]["embedding"] == [0.0] * 3
