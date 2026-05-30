"""Unit tests for the trained-energy-vs-correctness calibration module.

Traces to:
  REQ-KONA-3461 (Trained Energy Correctness Calibration v2)
  SCENARIO-KONA-3461 (trained energy lifts AUROC above untrained floor)
  SCENARIO-KONA-3461-BLOCKED (honest block on missing corpus/substrate)

Uses small synthetic corpora so tests run in milliseconds with no live model.
"""

from __future__ import annotations

import pytest

from carnot.phase3.p01_trained_energy_correctness_calibration import (
    UNTRAINED_AUROC_BASELINE,
    TrainedCalibrationResult,
    _within_problem_argmin_rate,
    compute_trained_calibration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(problem_id: str, gold: int, answers: list[int | None],
                 texts: list[str] | None = None, logprobs: list[float | None] | None = None) -> dict:
    """Build a synthetic corpus record."""
    n = len(answers)
    texts = texts or [""] * n
    logprobs = logprobs or [None] * n
    return {
        "problem_id": problem_id,
        "gold": gold,
        "greedy": {"answer": answers[0]},
        "samples": [
            {"text": t, "answer": a, "mean_token_logprob": lp, "n_tokens": 10}
            for t, a, lp in zip(texts, answers, logprobs)
        ],
        "k": n,
        "temperature": 0.7,
    }


def _minimal_corpus(n_problems: int = 12) -> list[dict]:
    """A minimal corpus with perfect signal: correct candidates have low index."""
    records = []
    for i in range(n_problems):
        pid = f"prob-{i}"
        gold = 42 + i
        # 6 samples: first 3 correct, last 3 incorrect
        answers = [gold, gold, gold, gold + 1, gold + 2, gold + 3]
        records.append(_make_record(pid, gold, answers))
    return records


# ---------------------------------------------------------------------------
# _within_problem_argmin_rate
# ---------------------------------------------------------------------------

class TestWithinProblemArgminRate:
    """REQ-KONA-3461: within-problem argmin rate helper."""

    def test_empty_returns_zero(self):
        assert _within_problem_argmin_rate({}, higher_is_better=True) == pytest.approx(0.0)

    def test_always_correct_higher_is_better(self):
        """When correct candidate always has the highest score, rate = 1.0."""
        problem_scores = {
            "p1": [(0.9, 1), (0.3, 0), (0.2, 0)],
            "p2": [(0.8, 1), (0.4, 0)],
        }
        rate = _within_problem_argmin_rate(problem_scores, higher_is_better=True)
        assert rate == pytest.approx(1.0)

    def test_never_correct_higher_is_better(self):
        """When incorrect candidate always has the highest score, rate = 0.0."""
        problem_scores = {
            "p1": [(0.9, 0), (0.3, 1)],
            "p2": [(0.8, 0), (0.4, 1)],
        }
        rate = _within_problem_argmin_rate(problem_scores, higher_is_better=True)
        assert rate == pytest.approx(0.0)

    def test_half_correct(self):
        """Half the problems pick correctly."""
        problem_scores = {
            "p1": [(0.9, 1), (0.1, 0)],  # correct
            "p2": [(0.9, 0), (0.1, 1)],  # incorrect
        }
        rate = _within_problem_argmin_rate(problem_scores, higher_is_better=True)
        assert rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TrainedCalibrationResult dataclass
# ---------------------------------------------------------------------------

class TestTrainedCalibrationResult:
    """REQ-KONA-3461: result dataclass holds all required fields."""

    def _make_result(self) -> TrainedCalibrationResult:
        return TrainedCalibrationResult(
            n_candidates_heldout=100,
            trained_energy_correctness_auroc=0.72,
            trained_energy_correctness_spearman=-0.25,
            fover_energy_correctness_auroc=0.61,
            fover_energy_correctness_spearman=-0.10,
            trained_energy_auroc_lift_over_untrained=0.72 - UNTRAINED_AUROC_BASELINE,
            within_problem_argmin_correct_rate_trained=0.65,
            within_problem_argmin_correct_rate_fover=0.58,
            n_problems_heldout=20,
        )

    def test_lift_field_computed_correctly(self):
        """REQ-KONA-3461: lift = trained_auroc - 0.516 (the exp3450 baseline)."""
        r = self._make_result()
        assert r.trained_energy_auroc_lift_over_untrained == pytest.approx(
            0.72 - UNTRAINED_AUROC_BASELINE, rel=1e-6
        )

    def test_all_fields_present(self):
        r = self._make_result()
        assert r.n_candidates_heldout == 100
        assert r.trained_energy_correctness_auroc == pytest.approx(0.72)
        assert r.fover_energy_correctness_auroc == pytest.approx(0.61)
        assert r.within_problem_argmin_correct_rate_trained == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# compute_trained_calibration: integration on a tiny synthetic corpus
# ---------------------------------------------------------------------------

class TestComputeTrainedCalibration:
    """SCENARIO-KONA-3461: calibration on a small corpus runs end-to-end."""

    def test_runs_and_returns_result(self):
        """SCENARIO-KONA-3461: function runs without error on a minimal corpus."""
        corpus = _minimal_corpus(n_problems=12)
        result = compute_trained_calibration(
            corpus, seed=42, n_folds=3, reranker_iter=50
        )
        assert isinstance(result, TrainedCalibrationResult)
        assert result.n_candidates_heldout > 0
        assert result.n_problems_heldout == len(corpus)

    def test_auroc_bounds(self):
        """SCENARIO-KONA-3461: AUROCs are in [0, 1]."""
        corpus = _minimal_corpus(n_problems=12)
        result = compute_trained_calibration(
            corpus, seed=42, n_folds=3, reranker_iter=50
        )
        assert 0.0 <= result.trained_energy_correctness_auroc <= 1.0
        assert 0.0 <= result.fover_energy_correctness_auroc <= 1.0

    def test_argmin_rates_are_fractions(self):
        """SCENARIO-KONA-3461: argmin rates are in [0, 1]."""
        corpus = _minimal_corpus(n_problems=12)
        result = compute_trained_calibration(
            corpus, seed=42, n_folds=3, reranker_iter=50
        )
        assert 0.0 <= result.within_problem_argmin_correct_rate_trained <= 1.0
        assert 0.0 <= result.within_problem_argmin_correct_rate_fover <= 1.0

    def test_lift_equals_auroc_minus_baseline(self):
        """SCENARIO-KONA-3461: lift field is exactly AUROC - UNTRAINED_AUROC_BASELINE."""
        corpus = _minimal_corpus(n_problems=12)
        result = compute_trained_calibration(
            corpus, seed=42, n_folds=3, reranker_iter=50
        )
        assert result.trained_energy_auroc_lift_over_untrained == pytest.approx(
            result.trained_energy_correctness_auroc - UNTRAINED_AUROC_BASELINE, rel=1e-9
        )

    def test_n_candidates_heldout_matches_corpus_size(self):
        """SCENARIO-KONA-3461: every problem's candidates appear exactly once as held-out."""
        corpus = _minimal_corpus(n_problems=12)
        total_candidates = sum(len(r["samples"]) for r in corpus)
        result = compute_trained_calibration(
            corpus, seed=42, n_folds=3, reranker_iter=50
        )
        assert result.n_candidates_heldout == total_candidates

    def test_untrained_auroc_baseline_value(self):
        """REQ-KONA-3461: baseline constant matches exp3450 reference."""
        assert UNTRAINED_AUROC_BASELINE == pytest.approx(0.516)
