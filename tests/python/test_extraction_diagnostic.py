"""Tests for carnot.extraction.extraction_diagnostic — 100% coverage.

Every test references a REQ-EXTRACT-030, SCENARIO-EXTRACT-055/056/057 identifier.

Spec: REQ-EXTRACT-030, SCENARIO-EXTRACT-055, SCENARIO-EXTRACT-056, SCENARIO-EXTRACT-057
"""

from __future__ import annotations

from typing import Any

import pytest

from carnot.extraction.extraction_diagnostic import (
    ExtractionDiagnosticResult,
    run_extractor_diagnostic,
)


# ---------------------------------------------------------------------------
# Minimal stub extractors for testing
# ---------------------------------------------------------------------------


class AlwaysFlagsExtractor:
    """Stub: always returns one violation regardless of input.

    Used to verify that correct responses become false positives (FP).
    """

    def detect_violations(self, text: str) -> list[Any]:
        return [{"rule": "stub", "error": "always"}]


class NeverFlagsExtractor:
    """Stub: always returns no violations.

    Used to verify that incorrect responses become false negatives (FN).
    """

    def detect_violations(self, text: str) -> list[Any]:
        return []


class SelectiveFlagsExtractor:
    """Stub: flags responses that contain the word 'WRONG'.

    Used to build a controlled confusion matrix with known TP/FP counts.
    """

    def detect_violations(self, text: str) -> list[Any]:
        if "WRONG" in text:
            return [{"rule": "stub"}]
        return []


# ---------------------------------------------------------------------------
# Labeled response fixtures
# ---------------------------------------------------------------------------


def _make_response(text: str, is_correct: bool) -> dict:
    return {"response": text, "is_correct": is_correct}


# 4 responses: 2 correct, 2 incorrect
_MIXED_RESPONSES = [
    _make_response("Good answer", True),
    _make_response("Another good answer", True),
    _make_response("WRONG answer one", False),
    _make_response("WRONG answer two", False),
]

# All correct
_ALL_CORRECT = [
    _make_response("correct A", True),
    _make_response("correct B", True),
]

# All incorrect
_ALL_INCORRECT = [
    _make_response("WRONG one", False),
    _make_response("WRONG two", False),
]


# ---------------------------------------------------------------------------
# Tests: ExtractionDiagnosticResult
# ---------------------------------------------------------------------------


class TestExtractionDiagnosticResult:
    """REQ-EXTRACT-030: ExtractionDiagnosticResult stores confusion matrix fields."""

    def test_fields_stored(self) -> None:
        """SCENARIO-EXTRACT-055: All confusion matrix fields are accessible."""
        result = ExtractionDiagnosticResult(
            extractor_name="test",
            n_tested=10,
            n_violations_found=5,
            n_true_positive=3,
            n_false_positive=2,
            n_true_negative=4,
            n_false_negative=1,
            tp_rate=0.75,
            fp_rate=0.33,
        )
        assert result.extractor_name == "test"
        assert result.n_tested == 10
        assert result.n_violations_found == 5
        assert result.n_true_positive == 3
        assert result.n_false_positive == 2
        assert result.n_true_negative == 4
        assert result.n_false_negative == 1
        assert result.tp_rate == pytest.approx(0.75)
        assert result.fp_rate == pytest.approx(0.33)

    def test_to_dict_contains_all_fields(self) -> None:
        """SCENARIO-EXTRACT-055: to_dict() serialises every field for JSON embedding."""
        result = ExtractionDiagnosticResult(
            extractor_name="vprm",
            n_tested=4,
            n_violations_found=2,
            n_true_positive=1,
            n_false_positive=1,
            n_true_negative=2,
            n_false_negative=0,
            tp_rate=1.0,
            fp_rate=0.5,
        )
        d = result.to_dict()
        assert d["extractor_name"] == "vprm"
        assert d["n_tested"] == 4
        assert d["n_violations_found"] == 2
        assert d["n_true_positive"] == 1
        assert d["n_false_positive"] == 1
        assert d["n_true_negative"] == 2
        assert d["n_false_negative"] == 0
        assert d["tp_rate"] == pytest.approx(1.0)
        assert d["fp_rate"] == pytest.approx(0.5)
        assert "per_question_flags" in d

    def test_default_per_question_flags_empty(self) -> None:
        """SCENARIO-EXTRACT-055: per_question_flags defaults to empty list."""
        result = ExtractionDiagnosticResult(
            extractor_name="x",
            n_tested=0,
            n_violations_found=0,
            n_true_positive=0,
            n_false_positive=0,
            n_true_negative=0,
            n_false_negative=0,
            tp_rate=0.0,
            fp_rate=0.0,
        )
        assert result.per_question_flags == []


# ---------------------------------------------------------------------------
# Tests: run_extractor_diagnostic — confusion matrix logic
# ---------------------------------------------------------------------------


class TestRunExtractorDiagnostic:
    """REQ-EXTRACT-030: run_extractor_diagnostic builds correct confusion matrix."""

    def test_always_flags_on_mixed(self) -> None:
        """SCENARIO-EXTRACT-056: AlwaysFlags → every correct response is FP, every incorrect is TP."""
        result = run_extractor_diagnostic(
            AlwaysFlagsExtractor(), "always", _MIXED_RESPONSES
        )
        # 2 correct flagged → FP; 2 incorrect flagged → TP; no FN, no TN
        assert result.n_false_positive == 2
        assert result.n_true_positive == 2
        assert result.n_true_negative == 0
        assert result.n_false_negative == 0
        assert result.n_violations_found == 4
        assert result.n_tested == 4
        assert result.tp_rate == pytest.approx(1.0)
        assert result.fp_rate == pytest.approx(1.0)

    def test_never_flags_on_mixed(self) -> None:
        """SCENARIO-EXTRACT-056: NeverFlags → every incorrect response is FN, every correct is TN."""
        result = run_extractor_diagnostic(
            NeverFlagsExtractor(), "never", _MIXED_RESPONSES
        )
        assert result.n_false_positive == 0
        assert result.n_true_positive == 0
        assert result.n_true_negative == 2
        assert result.n_false_negative == 2
        assert result.n_violations_found == 0
        assert result.tp_rate == pytest.approx(0.0)
        assert result.fp_rate == pytest.approx(0.0)

    def test_selective_flags_on_mixed(self) -> None:
        """SCENARIO-EXTRACT-056: SelectiveFlags → TP on WRONG responses, TN on correct."""
        result = run_extractor_diagnostic(
            SelectiveFlagsExtractor(), "selective", _MIXED_RESPONSES
        )
        # "Good answer" and "Another good answer" → no flag → TN
        # "WRONG answer one" and "WRONG answer two" → flagged → TP
        assert result.n_true_positive == 2
        assert result.n_false_positive == 0
        assert result.n_true_negative == 2
        assert result.n_false_negative == 0
        assert result.tp_rate == pytest.approx(1.0)
        assert result.fp_rate == pytest.approx(0.0)

    def test_per_question_flags_populated(self) -> None:
        """SCENARIO-EXTRACT-057: per_question_flags has one entry per response with cell label."""
        result = run_extractor_diagnostic(
            SelectiveFlagsExtractor(), "selective", _MIXED_RESPONSES
        )
        assert len(result.per_question_flags) == 4
        cells = [f["cell"] for f in result.per_question_flags]
        assert cells == ["TN", "TN", "TP", "TP"]

    def test_per_question_flags_has_is_correct_and_violation_found(self) -> None:
        """SCENARIO-EXTRACT-057: Each per_question_flags entry has is_correct and violation_found."""
        result = run_extractor_diagnostic(
            AlwaysFlagsExtractor(), "always", [_make_response("x", True)]
        )
        flag = result.per_question_flags[0]
        assert "is_correct" in flag
        assert "violation_found" in flag
        assert "cell" in flag
        assert flag["is_correct"] is True
        assert flag["violation_found"] is True
        assert flag["cell"] == "FP"

    def test_extractor_name_stored(self) -> None:
        """SCENARIO-EXTRACT-055: extractor_name is stored in result."""
        result = run_extractor_diagnostic(
            NeverFlagsExtractor(), "my_extractor_v2", _MIXED_RESPONSES
        )
        assert result.extractor_name == "my_extractor_v2"

    def test_empty_responses(self) -> None:
        """SCENARIO-EXTRACT-056: Empty response list returns zero counts and zero rates."""
        result = run_extractor_diagnostic(
            AlwaysFlagsExtractor(), "always", []
        )
        assert result.n_tested == 0
        assert result.n_violations_found == 0
        assert result.n_true_positive == 0
        assert result.n_false_positive == 0
        assert result.tp_rate == pytest.approx(0.0)
        assert result.fp_rate == pytest.approx(0.0)

    def test_all_correct_with_never_flags(self) -> None:
        """SCENARIO-EXTRACT-056: All correct + NeverFlags → fp_rate=0.0, tp_rate=0.0."""
        result = run_extractor_diagnostic(
            NeverFlagsExtractor(), "never", _ALL_CORRECT
        )
        assert result.fp_rate == pytest.approx(0.0)
        assert result.tp_rate == pytest.approx(0.0)
        assert result.n_true_negative == 2

    def test_all_correct_with_always_flags(self) -> None:
        """SCENARIO-EXTRACT-056: All correct + AlwaysFlags → fp_rate=1.0, tp_rate=0.0 (no incorrect)."""
        result = run_extractor_diagnostic(
            AlwaysFlagsExtractor(), "always", _ALL_CORRECT
        )
        assert result.fp_rate == pytest.approx(1.0)
        assert result.tp_rate == pytest.approx(0.0)
        assert result.n_false_positive == 2
        assert result.n_true_positive == 0

    def test_all_incorrect_with_never_flags(self) -> None:
        """SCENARIO-EXTRACT-056: All incorrect + NeverFlags → tp_rate=0.0, fp_rate=0.0 (no correct)."""
        result = run_extractor_diagnostic(
            NeverFlagsExtractor(), "never", _ALL_INCORRECT
        )
        assert result.tp_rate == pytest.approx(0.0)
        assert result.fp_rate == pytest.approx(0.0)
        assert result.n_false_negative == 2

    def test_all_incorrect_with_always_flags(self) -> None:
        """SCENARIO-EXTRACT-056: All incorrect + AlwaysFlags → tp_rate=1.0, fp_rate=0.0 (no correct)."""
        result = run_extractor_diagnostic(
            AlwaysFlagsExtractor(), "always", _ALL_INCORRECT
        )
        assert result.tp_rate == pytest.approx(1.0)
        # fp_rate should be 0.0 because there are no correct responses
        assert result.fp_rate == pytest.approx(0.0)
        assert result.n_true_positive == 2
        assert result.n_false_positive == 0

    def test_fn_cell_label(self) -> None:
        """SCENARIO-EXTRACT-057: FN cell label produced when incorrect response not flagged."""
        incorrect_response = [_make_response("missed violation", False)]
        result = run_extractor_diagnostic(
            NeverFlagsExtractor(), "never", incorrect_response
        )
        assert result.per_question_flags[0]["cell"] == "FN"
