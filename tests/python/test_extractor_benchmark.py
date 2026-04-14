"""Tests for the extractor benchmark primitives used in Exp 311.

Covers ExtractorBenchmarkRow, compute_fp_rate, compute_tp_rate, BenchmarkResult,
artifact schema, and the deterministic corpus loader at 100% branch coverage.

Spec: REQ-EXTRACT-012, SCENARIO-EXTRACT-025, SCENARIO-EXTRACT-026
"""

from __future__ import annotations

import pytest

from scripts.experiment_311_extractor_benchmark import (
    BenchmarkResult,
    ExtractorBenchmarkRow,
    build_labeled_corpus,
    compute_fp_rate,
    compute_tp_rate,
    select_winner,
)


# ---------------------------------------------------------------------------
# ExtractorBenchmarkRow
# ---------------------------------------------------------------------------


class TestExtractorBenchmarkRow:
    """REQ-EXTRACT-012: ExtractorBenchmarkRow dataclass contracts."""

    def test_fields_present(self) -> None:
        """Row must carry all required fields."""
        row = ExtractorBenchmarkRow(
            question="What is 2+2?",
            response="2+2=4",
            correct=True,
            extractor_name="ArithmeticExtractor",
            fp=False,
            tp=False,
            runtime_ms=1.5,
        )
        assert row.question == "What is 2+2?"
        assert row.response == "2+2=4"
        assert row.correct is True
        assert row.extractor_name == "ArithmeticExtractor"
        assert row.fp is False
        assert row.tp is False
        assert row.runtime_ms == pytest.approx(1.5)

    def test_fp_true_only_on_correct_response(self) -> None:
        """FP=True is only meaningful on a correct response."""
        row = ExtractorBenchmarkRow(
            question="q",
            response="r",
            correct=True,
            extractor_name="X",
            fp=True,
            tp=False,
            runtime_ms=0.0,
        )
        assert row.fp is True
        assert row.tp is False

    def test_tp_true_only_on_incorrect_response(self) -> None:
        """TP=True is only meaningful on an incorrect response."""
        row = ExtractorBenchmarkRow(
            question="q",
            response="r",
            correct=False,
            extractor_name="X",
            fp=False,
            tp=True,
            runtime_ms=0.0,
        )
        assert row.tp is True
        assert row.fp is False

    def test_error_field_defaults_none(self) -> None:
        """Error field defaults to None when no exception occurred."""
        row = ExtractorBenchmarkRow(
            question="q",
            response="r",
            correct=True,
            extractor_name="X",
            fp=False,
            tp=False,
            runtime_ms=0.0,
        )
        assert row.error is None

    def test_error_field_set(self) -> None:
        """Error field captures exception message."""
        row = ExtractorBenchmarkRow(
            question="q",
            response="r",
            correct=True,
            extractor_name="X",
            fp=False,
            tp=False,
            runtime_ms=0.0,
            error="TimeoutError",
        )
        assert row.error == "TimeoutError"


# ---------------------------------------------------------------------------
# compute_fp_rate
# ---------------------------------------------------------------------------


class TestComputeFpRate:
    """REQ-EXTRACT-012: compute_fp_rate contracts.

    SCENARIO-EXTRACT-025: FP rate = n_fp / n_correct_responses.
    """

    def test_zero_fp(self) -> None:
        """compute_fp_rate is 0.0 when no FPs on correct responses."""
        rows = [
            ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0),
            ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0),
        ]
        assert compute_fp_rate(rows) == pytest.approx(0.0)

    def test_all_correct_flagged(self) -> None:
        """compute_fp_rate is 1.0 when all correct responses are flagged."""
        rows = [
            ExtractorBenchmarkRow("q", "r", True, "X", fp=True, tp=False, runtime_ms=0.0),
            ExtractorBenchmarkRow("q", "r", True, "X", fp=True, tp=False, runtime_ms=0.0),
        ]
        assert compute_fp_rate(rows) == pytest.approx(1.0)

    def test_partial_fp(self) -> None:
        """SCENARIO-EXTRACT-025: 2 FP out of 15 correct gives rate 2/15."""
        rows = (
            [ExtractorBenchmarkRow("q", "r", True, "X", fp=True, tp=False, runtime_ms=0.0)] * 2
            + [ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0)] * 13
            + [ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=True, runtime_ms=0.0)] * 5
        )
        assert compute_fp_rate(rows) == pytest.approx(2 / 15)

    def test_no_correct_responses_returns_zero(self) -> None:
        """compute_fp_rate returns 0.0 when there are no correct responses."""
        rows = [
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=True, runtime_ms=0.0),
        ]
        assert compute_fp_rate(rows) == pytest.approx(0.0)

    def test_ignores_incorrect_responses(self) -> None:
        """compute_fp_rate counts only rows where correct=True."""
        rows = [
            ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0),
            # This incorrect row with fp=True should NOT count toward fp_rate
            ExtractorBenchmarkRow("q", "r", False, "X", fp=True, tp=False, runtime_ms=0.0),
        ]
        assert compute_fp_rate(rows) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_tp_rate
# ---------------------------------------------------------------------------


class TestComputeTpRate:
    """REQ-EXTRACT-012: compute_tp_rate contracts.

    SCENARIO-EXTRACT-025: TP rate = n_tp / n_incorrect_responses.
    """

    def test_zero_tp(self) -> None:
        """compute_tp_rate is 0.0 when no TPs on incorrect responses."""
        rows = [
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=False, runtime_ms=0.0),
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=False, runtime_ms=0.0),
        ]
        assert compute_tp_rate(rows) == pytest.approx(0.0)

    def test_all_incorrect_detected(self) -> None:
        """compute_tp_rate is 1.0 when all incorrect responses are detected."""
        rows = [
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=True, runtime_ms=0.0),
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=True, runtime_ms=0.0),
        ]
        assert compute_tp_rate(rows) == pytest.approx(1.0)

    def test_partial_tp(self) -> None:
        """SCENARIO-EXTRACT-025: 5 TP out of 15 incorrect gives rate 5/15."""
        rows = (
            [ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0)] * 15
            + [ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=True, runtime_ms=0.0)] * 5
            + [ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=False, runtime_ms=0.0)] * 10
        )
        assert compute_tp_rate(rows) == pytest.approx(5 / 15)

    def test_no_incorrect_responses_returns_zero(self) -> None:
        """compute_tp_rate returns 0.0 when there are no incorrect responses."""
        rows = [
            ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=False, runtime_ms=0.0),
        ]
        assert compute_tp_rate(rows) == pytest.approx(0.0)

    def test_ignores_correct_responses(self) -> None:
        """compute_tp_rate counts only rows where correct=False."""
        rows = [
            ExtractorBenchmarkRow("q", "r", False, "X", fp=False, tp=False, runtime_ms=0.0),
            # This correct row with tp=True should NOT count toward tp_rate
            ExtractorBenchmarkRow("q", "r", True, "X", fp=False, tp=True, runtime_ms=0.0),
        ]
        assert compute_tp_rate(rows) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """REQ-EXTRACT-012: BenchmarkResult dataclass contracts."""

    def test_fields_stored(self) -> None:
        """BenchmarkResult stores all metric fields correctly."""
        br = BenchmarkResult(
            extractor="ArithmeticExtractor",
            fp_rate=0.1,
            tp_rate=0.5,
            mean_runtime_ms=25.0,
            n_total=30,
        )
        assert br.extractor == "ArithmeticExtractor"
        assert br.fp_rate == pytest.approx(0.1)
        assert br.tp_rate == pytest.approx(0.5)
        assert br.mean_runtime_ms == pytest.approx(25.0)
        assert br.n_total == 30

    def test_zero_rates(self) -> None:
        """BenchmarkResult with zero TP is stored truthfully (SCENARIO-EXTRACT-026)."""
        br = BenchmarkResult(
            extractor="ArithmeticExtractor",
            fp_rate=0.0,
            tp_rate=0.0,
            mean_runtime_ms=5.0,
            n_total=30,
        )
        assert br.tp_rate == pytest.approx(0.0)

    def test_to_dict(self) -> None:
        """BenchmarkResult.to_dict() produces serializable dict."""
        br = BenchmarkResult(
            extractor="NL2Z3Extractor",
            fp_rate=0.05,
            tp_rate=0.2,
            mean_runtime_ms=300.0,
            n_total=30,
        )
        d = br.to_dict()
        assert d["extractor"] == "NL2Z3Extractor"
        assert d["fp_rate"] == pytest.approx(0.05)
        assert d["tp_rate"] == pytest.approx(0.2)
        assert d["mean_runtime_ms"] == pytest.approx(300.0)
        assert d["n_total"] == 30


# ---------------------------------------------------------------------------
# select_winner
# ---------------------------------------------------------------------------


class TestSelectWinner:
    """REQ-EXTRACT-012: winner selection logic.

    SCENARIO-EXTRACT-026: prefer TP > 0, then lowest FP.
    """

    def test_prefer_tp_over_zero_tp(self) -> None:
        """SCENARIO-EXTRACT-026: extractor with TP > 0 beats one with TP = 0."""
        results = [
            BenchmarkResult("ArithmeticExtractor", fp_rate=0.0, tp_rate=0.0, mean_runtime_ms=1.0, n_total=30),
            BenchmarkResult("NL2Z3Extractor", fp_rate=0.1, tp_rate=0.3, mean_runtime_ms=200.0, n_total=30),
        ]
        winner = select_winner(results)
        assert winner == "NL2Z3Extractor"

    def test_lowest_fp_among_tp_positive(self) -> None:
        """Among extractors with TP > 0, prefer lowest FP rate."""
        results = [
            BenchmarkResult("A", fp_rate=0.2, tp_rate=0.5, mean_runtime_ms=1.0, n_total=30),
            BenchmarkResult("B", fp_rate=0.05, tp_rate=0.4, mean_runtime_ms=1.0, n_total=30),
            BenchmarkResult("C", fp_rate=0.1, tp_rate=0.6, mean_runtime_ms=1.0, n_total=30),
        ]
        winner = select_winner(results)
        assert winner == "B"

    def test_all_zero_tp_select_lowest_fp(self) -> None:
        """When all extractors have TP = 0, pick the one with lowest FP."""
        results = [
            BenchmarkResult("A", fp_rate=0.3, tp_rate=0.0, mean_runtime_ms=1.0, n_total=30),
            BenchmarkResult("B", fp_rate=0.05, tp_rate=0.0, mean_runtime_ms=1.0, n_total=30),
        ]
        winner = select_winner(results)
        assert winner == "B"

    def test_single_extractor(self) -> None:
        """Single extractor is always the winner."""
        results = [
            BenchmarkResult("Only", fp_rate=0.5, tp_rate=0.0, mean_runtime_ms=1.0, n_total=30),
        ]
        assert select_winner(results) == "Only"


# ---------------------------------------------------------------------------
# build_labeled_corpus
# ---------------------------------------------------------------------------


class TestBuildLabeledCorpus:
    """REQ-EXTRACT-012: deterministic corpus loader."""

    def test_corpus_length(self) -> None:
        """Corpus must contain at least 30 entries (15 correct + 15 incorrect)."""
        corpus = build_labeled_corpus()
        assert len(corpus) >= 30

    def test_has_correct_and_incorrect(self) -> None:
        """Corpus must have both correct=True and correct=False entries."""
        corpus = build_labeled_corpus()
        correct = [c for c in corpus if c["correct"]]
        incorrect = [c for c in corpus if not c["correct"]]
        assert len(correct) >= 15
        assert len(incorrect) >= 15

    def test_required_fields_present(self) -> None:
        """Each corpus entry must have question, response, correct fields."""
        corpus = build_labeled_corpus()
        for entry in corpus:
            assert "question" in entry
            assert "response" in entry
            assert "correct" in entry
            assert isinstance(entry["correct"], bool)
            assert isinstance(entry["question"], str)
            assert isinstance(entry["response"], str)

    def test_deterministic(self) -> None:
        """Calling build_labeled_corpus() twice yields identical results."""
        first = build_labeled_corpus()
        second = build_labeled_corpus()
        assert first == second

    def test_non_empty_questions_and_responses(self) -> None:
        """Each entry must have non-empty question and response strings."""
        corpus = build_labeled_corpus()
        for entry in corpus:
            assert entry["question"].strip()
            assert entry["response"].strip()
