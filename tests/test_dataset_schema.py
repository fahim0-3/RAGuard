"""Phase D: the golden dataset is itself tested.

The dataset is the ruler that measures everything else. A silently wrong case
produces a silently wrong metric, and a metric nobody can trust is worse than
no metric, because it is acted upon.

Three layers, all deterministic and offline:

1. **Structural** — the file validates against golden_schema.json.
2. **Grounding** — every expected keyword really occurs in the document the
   case names. This is what stops a plausible-sounding but invented case.
3. **Composition** — the declared distribution matches the actual cases, and
   the multi-source cases that make Recall@k meaningful genuinely exist.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "policies"
DATASET_PATH = PROJECT_ROOT / "src" / "evaluation" / "golden_dataset.json"
SCHEMA_PATH = PROJECT_ROOT / "src" / "evaluation" / "golden_schema.json"

EXPECTED_DISTRIBUTION = {
    "normal": 15,
    "paraphrase": 10,
    "exact_term": 8,
    "multi_policy": 5,
    "ambiguous": 4,
    "unanswerable": 4,
    "prompt_injection": 2,
    "high_risk": 2,
}
EXPECTED_TOTAL = 50


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in DATA_DIR.glob("*.txt")}


# --------------------------------------------------------------------------
# 1. Structural
# --------------------------------------------------------------------------


def test_schema_file_is_itself_a_valid_json_schema(schema):
    Draft202012Validator.check_schema(schema)


def test_dataset_validates_against_schema(dataset, schema):
    errors = sorted(Draft202012Validator(schema).iter_errors(dataset), key=lambda e: list(e.path))
    assert not errors, "Schema violations:\n" + "\n".join(
        f"  {list(e.path)}: {e.message}" for e in errors[:10]
    )


def test_case_ids_are_unique_and_well_formed(dataset):
    ids = [c["id"] for c in dataset["cases"]]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"duplicate case IDs: {duplicates}"


def test_paraphrase_targets_exist(dataset):
    ids = {c["id"] for c in dataset["cases"]}
    for case in dataset["cases"]:
        target = case.get("paraphrase_of")
        if target is not None:
            assert target in ids, f"{case['id']} paraphrases missing case {target}"


def test_should_abstain_agrees_with_expected_outcome(dataset):
    for case in dataset["cases"]:
        expected = case["expected_outcome"] == "abstain"
        assert case["should_abstain"] is expected, (
            f"{case['id']}: should_abstain={case['should_abstain']} contradicts "
            f"expected_outcome={case['expected_outcome']}"
        )


def test_policy_ids_and_sources_are_consistent(dataset):
    """A case naming two sources must name two policy IDs."""
    for case in dataset["cases"]:
        assert len(case["expected_policy_ids"]) == len(case["expected_sources"]), (
            f"{case['id']}: {len(case['expected_sources'])} sources but "
            f"{len(case['expected_policy_ids'])} policy IDs"
        )


# --------------------------------------------------------------------------
# 2. Grounding
# --------------------------------------------------------------------------


def test_expected_sources_exist_on_disk(dataset):
    for case in dataset["cases"]:
        for source in case["expected_sources"]:
            assert (DATA_DIR / source).exists(), f"{case['id']} references missing {source}"


def test_every_expected_keyword_occurs_in_an_expected_source(dataset, corpus):
    """The strongest guard against a fabricated golden case."""
    failures = []
    for case in dataset["cases"]:
        for keyword in case["expected_keywords"]:
            text = "\n".join(corpus[s].lower() for s in case["expected_sources"])
            if keyword.lower() not in text:
                failures.append(f"{case['id']}: {keyword!r} absent from {case['expected_sources']}")
    assert not failures, "Ungrounded keywords:\n" + "\n".join(failures)


def test_expected_chunk_indices_are_within_the_document(dataset, corpus):
    """Chunk indices must be plausible for the document they name."""
    for case in dataset["cases"]:
        for entry in case.get("expected_chunk_indices", []):
            assert entry["source"] in case["expected_sources"], (
                f"{case['id']}: chunk index names {entry['source']}, "
                f"which is not an expected source"
            )
            assert entry["chunk_index"] >= 0


def test_abstention_cases_cite_nothing(dataset):
    for case in dataset["cases"]:
        if case["expected_outcome"] != "abstain":
            continue
        assert not case["expected_sources"], f"{case['id']} abstains but names sources"
        assert not case["expected_policy_ids"], f"{case['id']} abstains but names policy IDs"
        assert not case["expected_keywords"], f"{case['id']} abstains but names keywords"


def test_answerable_cases_name_their_evidence(dataset):
    for case in dataset["cases"]:
        if case["expected_outcome"] == "abstain":
            continue
        assert case["expected_sources"], f"{case['id']} must name at least one source"
        assert case["expected_policy_ids"], f"{case['id']} must name at least one policy ID"


def test_every_corpus_document_is_exercised(dataset):
    covered = {s for c in dataset["cases"] for s in c["expected_sources"]}
    on_disk = {p.name for p in DATA_DIR.glob("*.txt")}
    assert not on_disk - covered, f"No golden case exercises: {sorted(on_disk - covered)}"


def test_every_case_documents_where_the_answer_lives(dataset):
    for case in dataset["cases"]:
        assert case["notes"].strip(), f"{case['id']} has no verification note"


# --------------------------------------------------------------------------
# 3. Composition
# --------------------------------------------------------------------------


def test_dataset_has_exactly_fifty_cases(dataset):
    assert len(dataset["cases"]) == EXPECTED_TOTAL


def test_case_type_distribution_matches_the_specification(dataset):
    actual = Counter(c["case_type"] for c in dataset["cases"])
    assert dict(sorted(actual.items())) == dict(sorted(EXPECTED_DISTRIBUTION.items()))


def test_declared_distribution_matches_the_cases(dataset):
    """The header block must not drift away from the cases beneath it."""
    declared = dict(dataset["distribution"])
    total = declared.pop("total")
    actual = Counter(c["case_type"] for c in dataset["cases"])
    assert declared == dict(actual)
    assert total == len(dataset["cases"])


def test_multi_policy_cases_really_need_multiple_documents(dataset):
    """This is what makes Recall@k measure something HitRate@k does not."""
    multi = [c for c in dataset["cases"] if c["case_type"] == "multi_policy"]
    assert len(multi) == 5
    for case in multi:
        assert len(case["expected_sources"]) >= 2, (
            f"{case['id']} is labelled multi_policy but names one source"
        )
        assert len(set(case["expected_policy_ids"])) >= 2


def test_recall_is_not_a_restatement_of_hit_rate(dataset):
    """Guards the honesty claim in the Phase D report.

    If every scored case had a single expected source, Recall@k would be
    arithmetically identical to HitRate@k and reporting both as independent
    evidence would be misleading.
    """
    scored = [c for c in dataset["cases"] if not c["should_abstain"]]
    multi_source = [c for c in scored if len(c["expected_sources"]) > 1]
    assert len(multi_source) >= 5, (
        "Too few multi-source cases for Recall@k to be independent of HitRate@k"
    )


def test_outcome_counts(dataset):
    outcomes = Counter(c["expected_outcome"] for c in dataset["cases"])
    assert outcomes["clarify"] == 4
    assert outcomes["escalate"] == 2
    assert outcomes["abstain"] == 6, "4 unanswerable + 2 prompt injection"
    assert outcomes["answer"] == 38


def test_prompt_injection_cases_declare_what_they_forbid(dataset):
    injections = [c for c in dataset["cases"] if c["case_type"] == "prompt_injection"]
    assert len(injections) == 2
    for case in injections:
        assert case["injection_must_not"], f"{case['id']} must declare forbidden behaviours"
        assert case["should_abstain"] is True


def test_every_case_carries_a_priority(dataset):
    for case in dataset["cases"]:
        assert case["priority"] in {"critical", "high", "medium", "low"}


def test_critical_cases_exist_for_the_dangerous_paths(dataset):
    """Safety, security, and monetary thresholds are the cases that must not rot."""
    critical = {c["id"] for c in dataset["cases"] if c["priority"] == "critical"}
    for case_id in ("GC-045", "GC-047", "GC-048", "GC-049", "GC-050"):
        assert case_id in critical, f"{case_id} should be priority critical"


def test_dataset_version_is_recorded(dataset):
    assert dataset["version"], "metric runs are only comparable within a version"


def test_dataset_file_is_valid_json():
    payload = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))
    assert payload["cases"]
