"""Golden-dataset tests.

Three concerns, deliberately separated:

1. **Dataset integrity** (fast, always runs). The golden set is the ruler that
   measures everything else, so it is itself tested: unique IDs, referenced
   files that exist, and expected keywords that actually appear in the source
   document. A silently wrong golden case produces a silently wrong metric.

2. **Guard logic** (fast, always runs). Confidence scoring and citation
   verification are pure functions and are unit-tested without models.

3. **Pipeline behaviour** (marked `integration`). Requires pgvector and an
   ingested corpus; run locally and in the evaluation CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.evaluation.metrics import (
    compare_to_baseline,
    evaluate_retrieval,
    hit_at_k,
    keyword_recall,
    load_baseline,
    load_golden_dataset,
    reciprocal_rank,
)
from src.retrieval.bm25 import tokenize
from src.retrieval.types import RetrievedChunk
from src.self_healing.citation_verifier import (
    lexical_overlap,
    missing_factual_tokens,
    split_claims,
    verify_citations,
)
from src.self_healing.confidence import score_retrieval
from src.self_healing.query_rewriter import extract_identifiers, heuristic_rewrites

DATA_DIR = PROJECT_ROOT / "data" / "policies"


# ---------------------------------------------------------------------------
# 1. Dataset integrity
# ---------------------------------------------------------------------------


def test_dataset_ids_are_unique(golden_cases):
    ids = [case["id"] for case in golden_cases]
    assert len(ids) == len(set(ids)), "Golden case IDs must be unique and never reused"


def test_required_fields_present(golden_cases):
    required = {
        "id",
        "question",
        "ground_truth",
        "expected_sources",
        "expected_keywords",
        "should_abstain",
        "category",
    }
    for case in golden_cases:
        missing = required - set(case)
        assert not missing, f"{case.get('id')} is missing fields: {sorted(missing)}"


def test_expected_sources_exist_on_disk(golden_cases):
    for case in golden_cases:
        for source in case["expected_sources"]:
            assert (DATA_DIR / source).exists(), f"{case['id']} references missing {source}"


def test_expected_keywords_appear_in_their_source(golden_cases):
    """The strongest guard against a rotten ruler.

    If a policy document is edited and a golden keyword no longer appears in it,
    the metric would quietly measure the wrong thing. Fail loudly instead.
    """
    for case in golden_cases:
        if not case["expected_keywords"]:
            continue
        corpus = "\n".join(
            (DATA_DIR / source).read_text(encoding="utf-8").lower()
            for source in case["expected_sources"]
        )
        for keyword in case["expected_keywords"]:
            assert keyword.lower() in corpus, (
                f"{case['id']}: keyword {keyword!r} is not present in "
                f"{case['expected_sources']}. Fix the case or the corpus."
            )


def test_abstention_cases_declare_no_sources(golden_cases):
    for case in golden_cases:
        if case["should_abstain"]:
            assert not case["expected_sources"]
            assert not case["expected_keywords"]


def test_dataset_covers_every_corpus_document(golden_cases):
    covered = {s for case in golden_cases for s in case["expected_sources"]}
    on_disk = {p.name for p in DATA_DIR.glob("*.txt")}
    uncovered = on_disk - covered
    assert not uncovered, f"No golden case exercises: {sorted(uncovered)}"


def test_dataset_contains_at_least_one_abstention_case(golden_cases):
    assert any(case["should_abstain"] for case in golden_cases)


def test_baseline_is_well_formed():
    baseline = load_baseline()
    for section in ("retrieval", "end_to_end", "ragas"):
        assert section in baseline, f"baseline.json is missing the {section} section"
        for metric, value in baseline[section].items():
            assert 0.0 <= float(value) <= 1.0, f"{section}.{metric} must be a ratio"


# ---------------------------------------------------------------------------
# 2. Guard logic (no models, no network)
# ---------------------------------------------------------------------------


def test_tokenizer_preserves_identifiers():
    tokens = tokenize("Error PAY-402 on model AB-X200-EU")
    assert "pay-402" in tokens
    assert "402" in tokens  # sub-token so a bare-code query still matches
    assert "ab-x200-eu" in tokens


def test_identifier_extraction():
    assert "PAY-402" in extract_identifiers("why did I get pay-402 at checkout")
    assert extract_identifiers("how long is the refund window") == []


def test_heuristic_rewrites_map_colloquial_phrasing():
    variants = heuristic_rewrites("my parcel never arrived, where is my money", 3)
    joined = " ".join(variants)
    assert variants, "Heuristic fallback must always produce something to retry with"
    assert "lost parcel investigation" in joined or "refund" in joined


def test_confidence_high_when_top_score_strong(sample_chunks):
    confidence = score_retrieval(sample_chunks)
    assert confidence.level == "high"
    assert not confidence.should_abstain


def test_confidence_insufficient_on_empty_retrieval():
    confidence = score_retrieval([])
    assert confidence.level == "insufficient"
    assert confidence.should_abstain


def test_confidence_weak_triggers_healing(sample_chunks):
    weak = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            content=c.content,
            source=c.source,
            chunk_index=c.chunk_index,
            normalised_rerank_score=0.4,
        )
        for c in sample_chunks
    ]
    confidence = score_retrieval(weak)
    assert confidence.level == "weak"
    assert confidence.should_heal


def test_hit_and_mrr(sample_chunks):
    assert hit_at_k(sample_chunks, ["refund_policy.txt"], 5) == 1.0
    assert hit_at_k(sample_chunks, ["return_policy.txt"], 5) == 0.0
    assert reciprocal_rank(sample_chunks, ["payment_failure_faq.txt"]) == pytest.approx(0.5)


def test_keyword_recall_is_case_insensitive():
    assert keyword_recall("Refunds take 5 TO 7 BUSINESS DAYS.", ["5 to 7 business days"]) == 1.0
    assert keyword_recall("Refunds are quick.", ["5 to 7 business days"]) == 0.0


def test_split_claims_ignores_fragments():
    claims = split_claims("Refunds take 5 to 7 business days. Yes. The clock starts at scan.")
    assert len(claims) == 2  # "Yes." is too short to be a checkable claim


def test_lexical_overlap_and_missing_facts(sample_chunks):
    passage = sample_chunks[0].content
    assert lexical_overlap("Refunds take 5 to 7 business days", passage) > 0.5
    assert missing_factual_tokens("Refunds take 5 to 7 business days", passage) == []
    # A fabricated figure must be reported as unsupported.
    assert missing_factual_tokens("Refunds take 21 business days", passage) == ["21"]


def test_citation_verification_rejects_invented_label(sample_chunks):
    report = verify_citations(
        "Refunds take 5 to 7 business days to reach a credit card.",
        ["nonexistent_policy.txt#9"],
        sample_chunks,
    )
    assert not report.valid
    assert report.invalid_labels == ["nonexistent_policy.txt#9"]


def test_citation_verification_rejects_fabricated_number(sample_chunks):
    report = verify_citations(
        "Refunds take 21 business days to reach a credit card.",
        ["refund_policy.txt#2"],
        sample_chunks,
    )
    assert not report.valid
    assert report.unsupported_claims


def test_citation_verification_accepts_grounded_answer(sample_chunks):
    report = verify_citations(
        "Credit and debit card refunds take 5 to 7 business days.",
        ["refund_policy.txt#2"],
        sample_chunks,
    )
    assert report.valid
    assert report.support_ratio == 1.0


def test_citation_verification_rejects_uncited_answer(sample_chunks):
    report = verify_citations("Refunds take about a week.", [], sample_chunks)
    assert not report.valid


# ---------------------------------------------------------------------------
# 3. Pipeline behaviour (requires database + models)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.heavy
def test_retrieval_meets_baseline():
    """The deterministic merge gate, expressed as a test."""
    report = evaluate_retrieval()
    regressions = compare_to_baseline(report, "retrieval")
    assert not regressions, "Retrieval regressed:\n" + "\n".join(str(r) for r in regressions)


@pytest.mark.integration
@pytest.mark.heavy
@pytest.mark.parametrize("case", load_golden_dataset(), ids=lambda c: c["id"])
def test_each_case_retrieves_its_source(case):
    from src.self_healing.pipeline import get_pipeline

    chunks, confidence = get_pipeline().retrieve_only(case["question"])

    if case["should_abstain"]:
        assert confidence.should_abstain or confidence.level == "weak", (
            f"{case['id']}: out-of-scope question produced confident retrieval "
            f"({confidence.top_score:.3f}), which invites a hallucination"
        )
        return

    retrieved = {c.source for c in chunks}
    assert retrieved & set(case["expected_sources"]), (
        f"{case['id']}: expected one of {case['expected_sources']}, got {sorted(retrieved)}"
    )


@pytest.mark.integration
@pytest.mark.heavy
@pytest.mark.llm
def test_out_of_scope_question_abstains():
    from src.self_healing.pipeline import get_pipeline

    response = get_pipeline().answer("Can I get a mortgage or a personal loan through your store?")
    assert response.abstained
    assert response.abstain_reason


def test_reports_directory_is_not_committed():
    """Reports are build output; only the baseline is version-controlled."""
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "reports/" in gitignore


def test_golden_dataset_is_valid_json():
    path = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "golden_dataset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cases"], "Golden dataset must not be empty"
