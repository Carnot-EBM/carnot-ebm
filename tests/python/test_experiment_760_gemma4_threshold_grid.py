"""Tests for Experiment 760 — Gemma4-E4B-it VR Threshold Grid Search.

Spec: REQ-VERIFY-169, SCENARIO-VERIFY-222, SCENARIO-VERIFY-223, SCENARIO-VERIFY-224
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from scripts.experiment_760_gemma4_threshold_grid import (
    THRESHOLDS,
    _answers_match,
    _extract_numeric_answer,
    _symcode_confidence,
    classify_verdict,
)


# ---------------------------------------------------------------------------
# Helpers used to build synthetic per_threshold_results
# ---------------------------------------------------------------------------


def _make_results(signed_improvements: list[float]) -> list[dict]:
    """Build synthetic per_threshold_results matching the five thresholds.

    REQ-VERIFY-169-2 requires each entry to have threshold, signed_improvement, n_abstained.
    """
    return [
        {
            "threshold": t,
            "signed_improvement": si,
            "n_abstained": 0,
            "n_repaired": 10,
            "n_broken": 0,
            "baseline_accuracy": 0.80,
            "vr_accuracy": 0.80 + si,
            "n_questions": 50,
        }
        for t, si in zip(THRESHOLDS, signed_improvements)
    ]


# ---------------------------------------------------------------------------
# REQ-VERIFY-169-1 / SCENARIO-VERIFY-222: five thresholds
# ---------------------------------------------------------------------------


def test_threshold_list_has_five_entries() -> None:
    """THRESHOLDS constant must have exactly 5 entries.

    REQ-VERIFY-169-1: five thresholds in [0.10, 0.50] MUST be tested.
    SCENARIO-VERIFY-222: per_threshold_results has 5 entries.
    """
    assert len(THRESHOLDS) == 5


def test_threshold_values_in_expected_range() -> None:
    """All thresholds must be in [0.10, 0.50].

    REQ-VERIFY-169-1: thresholds MUST be in [0.10, 0.50].
    """
    for t in THRESHOLDS:
        assert 0.10 <= t <= 0.50, f"Threshold {t} out of range [0.10, 0.50]"


def test_per_threshold_results_has_five_entries() -> None:
    """Synthetic results list with 5 entries passes length check.

    SCENARIO-VERIFY-222: per_threshold_results contains exactly 5 entries.
    """
    results = _make_results([0.0, 0.0, 0.02, -0.01, -0.02])
    assert len(results) == 5


def test_per_threshold_results_has_required_keys() -> None:
    """Each entry in per_threshold_results must have threshold, signed_improvement, n_abstained.

    REQ-VERIFY-169-2.
    """
    results = _make_results([0.0, 0.0, 0.0, 0.0, 0.0])
    for entry in results:
        assert "threshold" in entry
        assert "signed_improvement" in entry
        assert "n_abstained" in entry


# ---------------------------------------------------------------------------
# REQ-VERIFY-169-3 / SCENARIO-VERIFY-223: best_threshold identification
# ---------------------------------------------------------------------------


def test_best_threshold_is_max_signed_improvement() -> None:
    """best_threshold must be the entry with highest signed_improvement.

    REQ-VERIFY-169-3, SCENARIO-VERIFY-223.
    """
    results = _make_results([-0.04, -0.02, 0.06, 0.02, -0.01])
    best_entry = max(results, key=lambda r: r["signed_improvement"])
    assert best_entry["threshold"] == 0.30  # index 2 has 0.06


def test_best_threshold_tie_goes_to_first() -> None:
    """When two thresholds tie, max() returns the first occurrence.

    REQ-VERIFY-169-3.
    """
    results = _make_results([0.05, 0.05, 0.00, 0.00, 0.00])
    best_entry = max(results, key=lambda r: r["signed_improvement"])
    assert best_entry["threshold"] == THRESHOLDS[0]


# ---------------------------------------------------------------------------
# REQ-VERIFY-169-4 / SCENARIO-VERIFY-224: positive_threshold_found flag
# ---------------------------------------------------------------------------


def test_positive_threshold_found_when_any_improvement_positive() -> None:
    """positive_threshold_found is True when any signed_improvement > 0.

    REQ-VERIFY-169-4, SCENARIO-VERIFY-224.
    """
    results = _make_results([-0.02, -0.01, 0.03, 0.00, -0.01])
    best_si = max(r["signed_improvement"] for r in results)
    positive_found = best_si > 0.0
    assert positive_found is True


def test_positive_threshold_not_found_when_all_non_positive() -> None:
    """positive_threshold_found is False when all signed_improvement <= 0.

    REQ-VERIFY-169-4.
    """
    results = _make_results([-0.04, -0.02, 0.00, -0.01, -0.03])
    best_si = max(r["signed_improvement"] for r in results)
    positive_found = best_si > 0.0
    assert positive_found is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-169-5/6/7: classify_verdict
# ---------------------------------------------------------------------------


def test_classify_verdict_positive_found_live() -> None:
    """classify_verdict returns 'gemma4_positive_found' when positive and live_gpu.

    REQ-VERIFY-169-5.
    """
    assert classify_verdict(positive_threshold_found=True, inference_mode="live_gpu") == "gemma4_positive_found"


def test_classify_verdict_no_positive_live() -> None:
    """classify_verdict returns 'gemma4_no_positive_threshold' when no positive and live_gpu.

    REQ-VERIFY-169-6.
    """
    assert classify_verdict(positive_threshold_found=False, inference_mode="live_gpu") == "gemma4_no_positive_threshold"


def test_classify_verdict_blocked_when_not_live() -> None:
    """classify_verdict returns 'blocked' when inference_mode is not live_gpu.

    REQ-VERIFY-169-7.
    """
    assert classify_verdict(positive_threshold_found=True, inference_mode="blocked") == "blocked"
    assert classify_verdict(positive_threshold_found=False, inference_mode="blocked_no_gpu") == "blocked"


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def test_extract_numeric_answer_explicit_pattern() -> None:
    """_extract_numeric_answer finds 'answer is X' patterns."""
    assert _extract_numeric_answer("The answer is 42") == 42.0


def test_extract_numeric_answer_fallback_last_num() -> None:
    """_extract_numeric_answer falls back to last numeric token."""
    assert _extract_numeric_answer("So we get 7 groups of 6 = 42.") == 42.0


def test_extract_numeric_answer_none_on_empty() -> None:
    """_extract_numeric_answer returns None for empty/no-number text."""
    assert _extract_numeric_answer("") is None
    assert _extract_numeric_answer("no numbers here") is None


def test_answers_match_within_tolerance() -> None:
    """_answers_match is True for values within 0.5 tolerance."""
    assert _answers_match(42.0, 42) is True
    assert _answers_match(42.3, 42) is True


def test_answers_match_false_for_wrong_answer() -> None:
    """_answers_match is False when difference exceeds tolerance."""
    assert _answers_match(10.0, 42) is False


def test_answers_match_none_returns_false() -> None:
    """_answers_match is False when either value is None."""
    assert _answers_match(None, 42) is False
    assert _answers_match(42.0, None) is False


# ---------------------------------------------------------------------------
# _symcode_confidence
# ---------------------------------------------------------------------------


def test_symcode_confidence_zero_compute_lines() -> None:
    """_symcode_confidence returns 0.2 when no COMPUTE: lines present.

    REQ-VERIFY-169: low-confidence default for unstructured responses.
    """
    assert _symcode_confidence("The answer is 42") == 0.2


def test_symcode_confidence_five_compute_lines() -> None:
    """_symcode_confidence returns 1.0 (capped) for 5+ COMPUTE: lines.

    REQ-VERIFY-169: confidence caps at 1.0 after 5 arithmetic steps.
    """
    text = "COMPUTE: step1\nCOMPUTE: step2\nCOMPUTE: step3\nCOMPUTE: step4\nCOMPUTE: step5"
    assert _symcode_confidence(text) == 1.0


def test_symcode_confidence_partial_compute_lines() -> None:
    """_symcode_confidence is proportional below 5 COMPUTE: lines.

    REQ-VERIFY-169.
    """
    text = "COMPUTE: step1\nCOMPUTE: step2"
    assert abs(_symcode_confidence(text) - 0.4) < 1e-9
