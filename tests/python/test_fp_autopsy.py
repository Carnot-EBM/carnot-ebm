"""Tests for FP autopsy module: FPCategory, AutopsyCase, categorize_fp, and helpers.

Spec: REQ-EXTRACT-013, REQ-EXTRACT-014,
      SCENARIO-EXTRACT-027, SCENARIO-EXTRACT-028,
      SCENARIO-EXTRACT-029, SCENARIO-EXTRACT-030
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from carnot.pipeline.fp_autopsy import (
    AutopsyCase,
    FPCategory,
    _extract_broken_from_rows,
    _looks_like_regex_artifact,
    categorize_fp,
    compute_category_distribution,
    load_broken_cases,
)


# ---------------------------------------------------------------------------
# FPCategory enum
# ---------------------------------------------------------------------------


def test_fpcategory_has_five_values() -> None:
    """REQ-EXTRACT-013: enum defines exactly the five required categories."""
    expected = {
        "VALID_INTERMEDIATE",
        "PRECISION_LIMIT",
        "REGEX_ARTIFACT",
        "REPAIR_DEGRADATION",
        "UNCATEGORIZED",
    }
    assert {c.value for c in FPCategory} == expected


def test_fpcategory_values_are_string_serializable() -> None:
    """REQ-EXTRACT-013: FPCategory values serialize to their string names."""
    assert FPCategory.VALID_INTERMEDIATE.value == "VALID_INTERMEDIATE"
    assert FPCategory.PRECISION_LIMIT.value == "PRECISION_LIMIT"
    assert FPCategory.REGEX_ARTIFACT.value == "REGEX_ARTIFACT"
    assert FPCategory.REPAIR_DEGRADATION.value == "REPAIR_DEGRADATION"
    assert FPCategory.UNCATEGORIZED.value == "UNCATEGORIZED"


# ---------------------------------------------------------------------------
# AutopsyCase dataclass
# ---------------------------------------------------------------------------


def test_autopsy_case_defaults() -> None:
    """REQ-EXTRACT-013: AutopsyCase initializes with UNCATEGORIZED and empty evidence."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="42",
        vr_answer="43",
        correct_answer="42",
    )
    assert case.fp_category is FPCategory.UNCATEGORIZED
    assert case.evidence == ""
    assert case.violations_flagged == []


def test_autopsy_case_fields_stored() -> None:
    """REQ-EXTRACT-013: AutopsyCase stores all required fields."""
    case = AutopsyCase(
        question="What is 2+2?",
        baseline_answer="4",
        vr_answer="5",
        correct_answer="4",
        violations_flagged=["2 + 2 = 5"],
        fp_category=FPCategory.REGEX_ARTIFACT,
        evidence="matched year",
    )
    assert case.question == "What is 2+2?"
    assert case.baseline_answer == "4"
    assert case.vr_answer == "5"
    assert case.correct_answer == "4"
    assert case.violations_flagged == ["2 + 2 = 5"]
    assert case.fp_category is FPCategory.REGEX_ARTIFACT
    assert case.evidence == "matched year"


# ---------------------------------------------------------------------------
# categorize_fp: VALID_INTERMEDIATE
# ---------------------------------------------------------------------------


def test_categorize_fp_valid_intermediate_via_step_keyword() -> None:
    """SCENARIO-EXTRACT-027: step keyword in violation → VALID_INTERMEDIATE."""
    case = AutopsyCase(
        question="What is the total?",
        baseline_answer="14",
        vr_answer="7",
        correct_answer="14",
        violations_flagged=["step 1: 10 - 3 = 7 (correct: 7)"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.VALID_INTERMEDIATE
    assert case.fp_category is FPCategory.VALID_INTERMEDIATE
    assert "evidence" in case.__dataclass_fields__
    assert len(case.evidence) > 0


def test_categorize_fp_valid_intermediate_via_then_keyword() -> None:
    """SCENARIO-EXTRACT-027: 'then' in violation text → VALID_INTERMEDIATE."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="10",
        vr_answer="5",
        correct_answer="10",
        violations_flagged=["5 + 0 = 5, then we proceed"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.VALID_INTERMEDIATE


def test_categorize_fp_valid_intermediate_via_so_keyword() -> None:
    """REQ-EXTRACT-013: 'so' keyword also triggers VALID_INTERMEDIATE."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="10",
        vr_answer="3",
        correct_answer="10",
        violations_flagged=["6 - 3 = 3, so the answer is 10"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.VALID_INTERMEDIATE


# ---------------------------------------------------------------------------
# categorize_fp: PRECISION_LIMIT
# ---------------------------------------------------------------------------


def test_categorize_fp_precision_limit_via_approx() -> None:
    """REQ-EXTRACT-013: 'approx' in violation → PRECISION_LIMIT."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="33",
        vr_answer="34",
        correct_answer="33",
        violations_flagged=["result is approx 33.3 flagged as 33"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.PRECISION_LIMIT


def test_categorize_fp_precision_limit_via_decimal() -> None:
    """REQ-EXTRACT-013: '0.' decimal substring in violation → PRECISION_LIMIT."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="7",
        vr_answer="8",
        correct_answer="7",
        violations_flagged=["0.5 + 6.5 = 7 flagged"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.PRECISION_LIMIT


def test_categorize_fp_precision_limit_via_round_keyword() -> None:
    """REQ-EXTRACT-013: 'round' keyword in violation → PRECISION_LIMIT."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="5",
        vr_answer="6",
        correct_answer="5",
        violations_flagged=["rounded result: 5"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.PRECISION_LIMIT


# ---------------------------------------------------------------------------
# categorize_fp: REGEX_ARTIFACT
# ---------------------------------------------------------------------------


def test_categorize_fp_regex_artifact_year_subtraction() -> None:
    """SCENARIO-EXTRACT-028: year-like 4-digit number in violation → REGEX_ARTIFACT."""
    case = AutopsyCase(
        question="What year was it?",
        baseline_answer="2023",
        vr_answer="2024",
        correct_answer="2023",
        violations_flagged=["2024 - 1 = 2023"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.REGEX_ARTIFACT
    assert "year" in case.evidence.lower() or "4-digit" in case.evidence.lower()


def test_categorize_fp_regex_artifact_1900s_year() -> None:
    """SCENARIO-EXTRACT-028: 19xx year-like number also triggers REGEX_ARTIFACT."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="1990",
        vr_answer="1991",
        correct_answer="1990",
        violations_flagged=["1990 + 0 = 1991 (correct: 1990)"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.REGEX_ARTIFACT


def test_looks_like_regex_artifact_positive() -> None:
    """REQ-EXTRACT-013: helper correctly identifies year-like substring."""
    assert _looks_like_regex_artifact("2024 - 1 = 2023") is True


def test_looks_like_regex_artifact_negative() -> None:
    """REQ-EXTRACT-013: helper returns False for ordinary arithmetic."""
    assert _looks_like_regex_artifact("5 + 3 = 8") is False


# ---------------------------------------------------------------------------
# categorize_fp: REPAIR_DEGRADATION
# ---------------------------------------------------------------------------


def test_categorize_fp_repair_degradation_no_violations() -> None:
    """REQ-EXTRACT-013: no violations flagged → REPAIR_DEGRADATION (repair itself caused damage)."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="10",
        vr_answer="9",
        correct_answer="10",
        violations_flagged=[],
    )
    result = categorize_fp(case)
    assert result is FPCategory.REPAIR_DEGRADATION
    assert "repair step" in case.evidence.lower()


def test_categorize_fp_repair_degradation_real_violation_but_worse() -> None:
    """REQ-EXTRACT-013: genuine violation but answer made worse → REPAIR_DEGRADATION."""
    case = AutopsyCase(
        question="Q",
        baseline_answer="10",
        vr_answer="7",
        correct_answer="10",
        # Violation text contains no intermediate/precision/year keywords,
        # so it falls through to REPAIR_DEGRADATION.
        violations_flagged=["8 + 3 = 10 (correct: 11)"],
    )
    result = categorize_fp(case)
    assert result is FPCategory.REPAIR_DEGRADATION


# ---------------------------------------------------------------------------
# categorize_fp: evidence is always set
# ---------------------------------------------------------------------------


def test_categorize_fp_always_sets_evidence() -> None:
    """REQ-EXTRACT-013: evidence field is non-empty after categorize_fp for all branches."""
    cases = [
        AutopsyCase("Q", "10", "9", "10", []),  # REPAIR_DEGRADATION (no violations)
        AutopsyCase("Q", "10", "9", "10", ["2024 - 1 = 2023"]),  # REGEX_ARTIFACT
        AutopsyCase("Q", "10", "9", "10", ["about 10.0"]),  # PRECISION_LIMIT
        AutopsyCase("Q", "10", "9", "10", ["step 1: 5 + 5 = 10"]),  # VALID_INTERMEDIATE
        AutopsyCase("Q", "10", "9", "10", ["8 + 3 = 10 (correct: 11)"]),  # REPAIR_DEGRADATION
    ]
    for case in cases:
        categorize_fp(case)
        assert len(case.evidence) > 0, f"evidence empty for category {case.fp_category}"


# ---------------------------------------------------------------------------
# load_broken_cases
# ---------------------------------------------------------------------------


def _write_json(data: object) -> str:
    """Write data to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    return path


def test_load_broken_cases_from_cases_key() -> None:
    """REQ-EXTRACT-013: loads broken cases from 'cases' list key."""
    data = {
        "cases": [
            # broken: baseline correct, vr wrong
            {
                "question": "Q1",
                "baseline_answer": "5",
                "vr_answer": "6",
                "correct_answer": "5",
                "violations_flagged": ["step 1: 3 + 2 = 5"],
            },
            # not broken: vr also correct
            {
                "question": "Q2",
                "baseline_answer": "7",
                "vr_answer": "7",
                "correct_answer": "7",
                "violations_flagged": [],
            },
            # not broken: baseline also wrong
            {
                "question": "Q3",
                "baseline_answer": "8",
                "vr_answer": "9",
                "correct_answer": "10",
                "violations_flagged": [],
            },
        ]
    }
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert len(cases) == 1
        assert cases[0].question == "Q1"
        assert cases[0].baseline_answer == "5"
        assert cases[0].vr_answer == "6"
        assert cases[0].correct_answer == "5"
        assert cases[0].violations_flagged == ["step 1: 3 + 2 = 5"]
    finally:
        os.unlink(path)


def test_load_broken_cases_from_per_question_results_key() -> None:
    """REQ-EXTRACT-013: loads broken cases from 'per_question_results' key."""
    data = {
        "per_question_results": [
            {
                "question": "What is 3+4?",
                "baseline_answer": "7",
                "vr_answer": "6",
                "correct_answer": "7",
            }
        ]
    }
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert len(cases) == 1
        assert cases[0].question == "What is 3+4?"
    finally:
        os.unlink(path)


def test_load_broken_cases_aggregate_only_returns_empty() -> None:
    """SCENARIO-EXTRACT-029: aggregate-only result file returns empty list (not error)."""
    # Matches the actual Exp 316/328 artifact shape — no per-question data.
    data = {
        "experiment": 316,
        "per_model_results": {
            "Qwen3.5-0.8B": {
                "baseline": {"number_swap": {"accuracy": 0.245, "n_total": 200}}
            }
        },
        "summary_table": [],
    }
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert cases == []
    finally:
        os.unlink(path)


def test_load_broken_cases_file_not_found() -> None:
    """REQ-EXTRACT-013: missing file returns empty list without raising."""
    cases = load_broken_cases("/nonexistent/path/to/results.json")
    assert cases == []


def test_load_broken_cases_invalid_json() -> None:
    """REQ-EXTRACT-013: malformed JSON returns empty list without raising."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write("NOT VALID JSON{{{")
    try:
        cases = load_broken_cases(path)
        assert cases == []
    finally:
        os.unlink(path)


def test_load_broken_cases_skips_rows_missing_required_keys() -> None:
    """REQ-EXTRACT-013: rows with missing required keys are skipped gracefully."""
    data = {
        "cases": [
            {"question": "Q1"},  # missing answer fields
            {
                "question": "Q2",
                "baseline_answer": "3",
                "vr_answer": "4",
                "correct_answer": "3",
            },
        ]
    }
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert len(cases) == 1
        assert cases[0].question == "Q2"
    finally:
        os.unlink(path)


def test_load_broken_cases_questions_key() -> None:
    """REQ-EXTRACT-013: also loads from 'questions' list key."""
    data = {
        "questions": [
            {
                "question": "Q",
                "baseline_answer": "1",
                "vr_answer": "2",
                "correct_answer": "1",
            }
        ]
    }
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert len(cases) == 1
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# _extract_broken_from_rows (internal helper — tested for 100% coverage)
# ---------------------------------------------------------------------------


def test_extract_broken_from_rows_filters_correctly() -> None:
    """REQ-EXTRACT-013: internal helper selects baseline-correct-vr-wrong rows only."""
    rows = [
        {"question": "Q1", "baseline_answer": "5", "vr_answer": "6", "correct_answer": "5"},
        {"question": "Q2", "baseline_answer": "5", "vr_answer": "5", "correct_answer": "5"},
        {"question": "Q3", "baseline_answer": "4", "vr_answer": "5", "correct_answer": "5"},
    ]
    broken = _extract_broken_from_rows(rows)
    assert len(broken) == 1
    assert broken[0].question == "Q1"


def test_extract_broken_from_rows_handles_violations_and_evidence() -> None:
    """REQ-EXTRACT-013: violations_flagged and evidence fields are preserved."""
    rows = [
        {
            "question": "Q",
            "baseline_answer": "10",
            "vr_answer": "11",
            "correct_answer": "10",
            "violations_flagged": ["a + b = c"],
            "evidence": "some evidence text",
        }
    ]
    broken = _extract_broken_from_rows(rows)
    assert broken[0].violations_flagged == ["a + b = c"]
    assert broken[0].evidence == "some evidence text"


# ---------------------------------------------------------------------------
# compute_category_distribution
# ---------------------------------------------------------------------------


def test_compute_category_distribution_all_categories_present() -> None:
    """REQ-EXTRACT-013: distribution always has all five categories as keys."""
    cases: list[AutopsyCase] = []
    dist = compute_category_distribution(cases)
    assert set(dist.keys()) == set(FPCategory)
    assert all(v == 0 for v in dist.values())


def test_compute_category_distribution_counts_correctly() -> None:
    """REQ-EXTRACT-013: counts match the fp_category values of the input cases."""
    cases = [
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.VALID_INTERMEDIATE),
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.VALID_INTERMEDIATE),
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.REGEX_ARTIFACT),
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.UNCATEGORIZED),
    ]
    dist = compute_category_distribution(cases)
    assert dist[FPCategory.VALID_INTERMEDIATE] == 2
    assert dist[FPCategory.REGEX_ARTIFACT] == 1
    assert dist[FPCategory.UNCATEGORIZED] == 1
    assert dist[FPCategory.PRECISION_LIMIT] == 0
    assert dist[FPCategory.REPAIR_DEGRADATION] == 0


def test_compute_category_distribution_single_case() -> None:
    """REQ-EXTRACT-013: single case produces expected distribution."""
    cases = [AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.PRECISION_LIMIT)]
    dist = compute_category_distribution(cases)
    assert dist[FPCategory.PRECISION_LIMIT] == 1
    assert sum(dist.values()) == 1


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-029: inconclusive when n < 5 (tested via load_broken_cases
# returning empty; experiment script handles the artifact logic — here we verify
# the data layer returns an empty list cleanly)
# ---------------------------------------------------------------------------


def test_load_broken_cases_empty_cases_list() -> None:
    """SCENARIO-EXTRACT-029: empty cases list in file returns empty list."""
    data = {"cases": []}
    path = _write_json(data)
    try:
        cases = load_broken_cases(path)
        assert cases == []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-030: recommended_fix mapped from primary_fp_type
# (distribution logic tested here; mapping itself in experiment script tests)
# ---------------------------------------------------------------------------


def test_fpcategory_uncategorized_is_default() -> None:
    """REQ-EXTRACT-013: UNCATEGORIZED is the default fp_category on a new AutopsyCase."""
    case = AutopsyCase("Q", "1", "2", "1")
    assert case.fp_category is FPCategory.UNCATEGORIZED


def test_primary_fp_type_derived_from_distribution() -> None:
    """SCENARIO-EXTRACT-030: primary FP type is the category with the highest count."""
    cases = [
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.REGEX_ARTIFACT),
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.REGEX_ARTIFACT),
        AutopsyCase("Q", "1", "2", "1", fp_category=FPCategory.VALID_INTERMEDIATE),
    ]
    dist = compute_category_distribution(cases)
    primary = max(dist, key=lambda k: dist[k])
    assert primary is FPCategory.REGEX_ARTIFACT
