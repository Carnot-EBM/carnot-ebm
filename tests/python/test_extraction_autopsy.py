"""Tests for live extraction autopsy helpers.

Spec: REQ-VERIFY-008, SCENARIO-VERIFY-008
"""

from __future__ import annotations

import pytest
from carnot.pipeline.extraction_autopsy import (
    build_case_record,
    capture_arithmetic_matches,
    diagnose_case,
    extract_final_number,
    select_showcase_cases,
    summarize_cases,
)


def test_capture_arithmetic_matches_preserves_text_and_verdicts() -> None:
    """REQ-VERIFY-008: raw regex spans and satisfaction flags are preserved."""
    matches = capture_arithmetic_matches("2 + 3 = 5. Then 10 - 2 = 9.")

    assert len(matches) == 2
    assert matches[0]["matched_text"] == "2 + 3 = 5"
    assert matches[0]["satisfied"] is True
    assert matches[1]["matched_text"] == "10 - 2 = 9"
    assert matches[1]["satisfied"] is False
    assert matches[1]["correct_result"] == 8


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#### 42", 42),
        ("Answer: -7", -7),
        ("I got 12 and then 13", 13),
        ("No digits here.", None),
    ],
)
def test_extract_final_number_supports_common_gsm8k_formats(
    text: str,
    expected: int | None,
) -> None:
    """REQ-VERIFY-008: final-answer extraction handles benchmark response formats."""
    assert extract_final_number(text) == expected


def test_diagnose_case_honors_manual_override() -> None:
    """REQ-VERIFY-008: manual autopsy overrides take precedence."""
    override = {
        "failure_category": "manual_override",
        "actual_error": "Reviewed by hand.",
        "proposed_extraction": "Use the hand-written note.",
    }

    diagnosis = diagnose_case(
        response="Anything at all.",
        regex_matches=[],
        extracted_answer=5,
        ground_truth=7,
        manual_override=override,
    )

    assert diagnosis == override


def test_diagnose_case_detects_rounding_and_format_mismatch() -> None:
    """REQ-VERIFY-008: diagnosis distinguishes approximation from regex misses."""
    rounding = diagnose_case(
        response="About 33.3, so approximately 33 in total. Answer: 33",
        regex_matches=[],
        extracted_answer=33,
        ground_truth=34,
    )
    mismatch = diagnose_case(
        response="Half of 48 is 24, then add 10 more. Answer: 35",
        regex_matches=[],
        extracted_answer=35,
        ground_truth=34,
    )

    assert rounding["failure_category"] == "rounding_or_approximation"
    assert "approximate" in rounding["actual_error"]
    assert mismatch["failure_category"] == "format_mismatch"
    assert "without emitting any `a +/- b = c` equation" in mismatch["actual_error"]


def test_diagnose_case_detects_logic_and_missing_answer_failures() -> None:
    """REQ-VERIFY-008: internally consistent equations can still be wrong overall."""
    logic = diagnose_case(
        response="2 + 3 = 5. 5 + 5 = 10. Answer: 11",
        regex_matches=capture_arithmetic_matches("2 + 3 = 5. 5 + 5 = 10. Answer: 11"),
        extracted_answer=11,
        ground_truth=10,
    )
    missing = diagnose_case(
        response="I think the result is the same as before.",
        regex_matches=[],
        extracted_answer=None,
        ground_truth=10,
    )

    assert logic["failure_category"] == "subtle_logic_error"
    assert "internally consistent" in logic["actual_error"]
    assert missing["failure_category"] == "missing_final_answer"
    assert "final numeric answer" in missing["actual_error"]


def test_diagnose_case_detects_explicit_arithmetic_violation() -> None:
    """REQ-VERIFY-008: explicit wrong equations remain first-class autopsy findings."""
    response = "2 + 3 = 6. Answer: 6"
    diagnosis = diagnose_case(
        response=response,
        regex_matches=capture_arithmetic_matches(response),
        extracted_answer=6,
        ground_truth=5,
    )

    assert diagnosis["failure_category"] == "explicit_arithmetic_violation"
    assert "2 + 3 = 6" in diagnosis["actual_error"]


def test_build_case_record_summary_and_showcase_selection() -> None:
    """SCENARIO-VERIFY-008: wrong answers and three correct contrast cases are retained."""
    wrong = build_case_record(
        sample_position=0,
        dataset_idx=12,
        question="Q0",
        ground_truth=10,
        response="Half of 48 is 24, then add 10 more. Answer: 35",
    )
    correct1 = build_case_record(
        sample_position=1,
        dataset_idx=13,
        question="Q1",
        ground_truth=5,
        response="2 + 3 = 5\nAnswer: 5",
    )
    correct2 = build_case_record(
        sample_position=2,
        dataset_idx=14,
        question="Q2",
        ground_truth=8,
        response="3 + 5 = 8\nAnswer: 8",
    )
    correct3 = build_case_record(
        sample_position=3,
        dataset_idx=15,
        question="Q3",
        ground_truth=9,
        response="4 + 5 = 9\nAnswer: 9",
    )
    correct4 = build_case_record(
        sample_position=4,
        dataset_idx=16,
        question="Q4",
        ground_truth=11,
        response="5 + 6 = 11\nAnswer: 11",
    )
    correct_no_equation = build_case_record(
        sample_position=5,
        dataset_idx=17,
        question="Q5",
        ground_truth=12,
        response="Answer: 12",
    )

    cases = [wrong, correct1, correct2, correct3, correct4, correct_no_equation]
    showcase = select_showcase_cases(cases, n_correct_examples=3)
    summary = summarize_cases(cases)

    assert wrong["correct"] is False
    assert wrong["failure_category"] == "format_mismatch"
    assert correct1["correct"] is True
    assert correct1["failure_category"] == "correct_answer"
    assert "did not emit an explicit regex-readable equation" in correct_no_equation["actual_error"]
    assert len(showcase["wrong_answer_autopsies"]) == 1
    assert len(showcase["correct_answer_examples"]) == 3
    assert summary["n_questions"] == 6
    assert summary["n_wrong"] == 1
    assert summary["n_wrong_with_regex_violations"] == 0
    assert summary["failure_categories"]["format_mismatch"] == 1
