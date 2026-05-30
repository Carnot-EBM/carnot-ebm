"""Tests for p01_energy_correctness_calibration.

REQ-KONA-3450: energy-correctness calibration audit for the P0.1 GSM8K corpus.

Each test references the spec requirement it exercises.
"""

from __future__ import annotations

import math

import pytest

from carnot.phase3.p01_energy_correctness_calibration import (
    CalibrationResult,
    CandidateRecord,
    binary_auroc,
    compute_candidate_energies,
    run_calibration_audit,
    spearman_correlation,
    within_problem_argmin_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_problem(
    problem_id: int,
    gold: int,
    answers: list[int],
    texts: list[str] | None = None,
) -> dict:
    """Build a minimal problem dict matching the corpus schema."""
    if texts is None:
        texts = [str(a) for a in answers]
    return {
        "problem_id": problem_id,
        "gold": gold,
        "samples": [
            {
                "text": t,
                "answer": a,
                "mean_token_logprob": -0.1,
                "n_tokens": 10,
            }
            for t, a in zip(texts, answers)
        ],
        "k": len(answers),
        "temperature": 0.7,
    }


# ---------------------------------------------------------------------------
# spearman_correlation tests
# ---------------------------------------------------------------------------


def test_spearman_perfect_positive():
    """REQ-KONA-3450: Spearman returns +1 for identical rank orders."""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    rho = spearman_correlation(x, y)
    assert abs(rho - 1.0) < 1e-9


def test_spearman_perfect_negative():
    """REQ-KONA-3450: Spearman returns -1 for perfectly reversed rank orders."""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 3.0, 2.0, 1.0]
    rho = spearman_correlation(x, y)
    assert abs(rho - (-1.0)) < 1e-9


def test_spearman_zero_variance():
    """REQ-KONA-3450: Spearman returns 0 when y is constant (no variance)."""
    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]
    rho = spearman_correlation(x, y)
    assert rho == 0.0


def test_spearman_two_elements():
    """REQ-KONA-3450: Spearman handles n=2 without error."""
    rho = spearman_correlation([1.0, 2.0], [0.0, 1.0])
    assert abs(rho - 1.0) < 1e-9


def test_spearman_degenerate_n1():
    """REQ-KONA-3450: Spearman returns 0 for single-element inputs (no correlation defined)."""
    assert spearman_correlation([1.0], [0.0]) == 0.0


def test_spearman_with_ties():
    """REQ-KONA-3450: Tied values use average-rank convention; result stays in [-1, 1]."""
    x = [1.0, 1.0, 2.0, 3.0]
    y = [0.0, 1.0, 0.0, 1.0]
    rho = spearman_correlation(x, y)
    assert -1.0 <= rho <= 1.0


def test_spearman_cross_validates_scipy():
    """REQ-KONA-3450: Confirm implementation matches scipy.stats.spearmanr on real data."""
    from scipy import stats

    x = [2.5, 1.0, 4.0, 3.0, 5.0]
    y = [0.0, 1.0, 0.0, 1.0, 0.0]
    our_rho = spearman_correlation(x, y)
    scipy_rho, _ = stats.spearmanr(x, y)
    assert abs(our_rho - scipy_rho) < 1e-9


# ---------------------------------------------------------------------------
# binary_auroc tests
# ---------------------------------------------------------------------------


def test_auroc_perfect():
    """REQ-KONA-3450: AUROC is 1.0 when all positives score above all negatives."""
    labels = [1, 1, 0, 0]
    scores = [3.0, 4.0, 1.0, 2.0]
    assert abs(binary_auroc(labels, scores) - 1.0) < 1e-9


def test_auroc_random():
    """REQ-KONA-3450: AUROC is 0.5 when scores are identical (random classifier)."""
    labels = [1, 0, 1, 0]
    scores = [0.0, 0.0, 0.0, 0.0]
    assert abs(binary_auroc(labels, scores) - 0.5) < 1e-9


def test_auroc_anti():
    """REQ-KONA-3450: AUROC is 0.0 when all negatives score above all positives."""
    labels = [1, 1, 0, 0]
    scores = [1.0, 2.0, 3.0, 4.0]
    assert abs(binary_auroc(labels, scores) - 0.0) < 1e-9


def test_auroc_no_positives():
    """REQ-KONA-3450: AUROC returns 0.5 (undefined/chance) when no positive examples."""
    labels = [0, 0, 0]
    scores = [1.0, 2.0, 3.0]
    assert binary_auroc(labels, scores) == 0.5


def test_auroc_no_negatives():
    """REQ-KONA-3450: AUROC returns 0.5 (undefined/chance) when no negative examples."""
    labels = [1, 1, 1]
    scores = [1.0, 2.0, 3.0]
    assert binary_auroc(labels, scores) == 0.5


def test_auroc_in_range():
    """REQ-KONA-3450: AUROC is always in [0, 1] for arbitrary inputs."""
    import random
    rng = random.Random(42)
    labels = [rng.randint(0, 1) for _ in range(20)]
    scores = [rng.random() for _ in range(20)]
    auroc = binary_auroc(labels, scores)
    assert 0.0 <= auroc <= 1.0


# ---------------------------------------------------------------------------
# CandidateRecord tests
# ---------------------------------------------------------------------------


def test_candidate_record_is_correct():
    """REQ-KONA-3450: CandidateRecord.is_correct returns True iff answer == gold."""
    rec_correct = CandidateRecord(problem_id=1, text="35", answer=35, gold=35, energy=0.1)
    rec_wrong = CandidateRecord(problem_id=1, text="36", answer=36, gold=35, energy=0.2)
    assert rec_correct.is_correct is True
    assert rec_wrong.is_correct is False


# ---------------------------------------------------------------------------
# within_problem_argmin_rate tests
# ---------------------------------------------------------------------------


def test_argmin_rate_all_correct():
    """REQ-KONA-3450: argmin rate is 1.0 when the lowest-energy candidate is always correct."""
    records = [
        CandidateRecord(problem_id=1, text="a", answer=5, gold=5, energy=0.1),  # correct, lowest E
        CandidateRecord(problem_id=1, text="b", answer=6, gold=5, energy=0.5),
        CandidateRecord(problem_id=2, text="c", answer=7, gold=7, energy=0.2),  # correct, lowest E
        CandidateRecord(problem_id=2, text="d", answer=8, gold=7, energy=0.9),
    ]
    assert within_problem_argmin_rate(records) == 1.0


def test_argmin_rate_none_correct():
    """REQ-KONA-3450: argmin rate is 0.0 when the lowest-energy candidate is always wrong."""
    records = [
        CandidateRecord(problem_id=1, text="a", answer=9, gold=5, energy=0.1),  # wrong, lowest E
        CandidateRecord(problem_id=1, text="b", answer=5, gold=5, energy=0.9),  # correct, high E
    ]
    assert within_problem_argmin_rate(records) == 0.0


def test_argmin_rate_empty():
    """REQ-KONA-3450: argmin rate is 0.0 for an empty record list."""
    assert within_problem_argmin_rate([]) == 0.0


def test_argmin_rate_tie_first_wins():
    """REQ-KONA-3450: ties in energy are broken by first occurrence in corpus order."""
    records = [
        # Both have energy=0.5; first is wrong, second is correct.
        CandidateRecord(problem_id=1, text="a", answer=9, gold=5, energy=0.5),
        CandidateRecord(problem_id=1, text="b", answer=5, gold=5, energy=0.5),
    ]
    # min() returns the FIRST element with the minimum value — first is wrong
    assert within_problem_argmin_rate(records) == 0.0


# ---------------------------------------------------------------------------
# compute_candidate_energies tests
# ---------------------------------------------------------------------------


def test_compute_candidate_energies_count():
    """REQ-KONA-3450: compute_candidate_energies returns one record per (problem, sample)."""
    problems = [
        _make_problem(1, 10, [10, 11]),
        _make_problem(2, 20, [20, 21, 22]),
    ]
    records = compute_candidate_energies(problems)
    assert len(records) == 5  # 2 + 3


def test_compute_candidate_energies_correctness_labels():
    """REQ-KONA-3450: CandidateRecord.is_correct reflects answer == gold."""
    problems = [_make_problem(1, 42, [42, 43])]
    records = compute_candidate_energies(problems)
    assert records[0].is_correct is True
    assert records[1].is_correct is False


def test_compute_candidate_energies_energy_is_float():
    """REQ-KONA-3450: each computed energy is a finite float."""
    problems = [_make_problem(1, 5, [5, 6, 7])]
    records = compute_candidate_energies(problems)
    for rec in records:
        assert isinstance(rec.energy, float)
        assert math.isfinite(rec.energy)


def test_compute_candidate_energies_empty_samples():
    """REQ-KONA-3450: problems with no samples produce no records (no crash)."""
    problem = {"problem_id": 99, "gold": 1, "samples": []}
    records = compute_candidate_energies([problem])
    assert records == []


# ---------------------------------------------------------------------------
# run_calibration_audit integration test
# ---------------------------------------------------------------------------


def test_run_calibration_audit_returns_result():
    """REQ-KONA-3450: run_calibration_audit returns a CalibrationResult with valid fields."""
    problems = [
        _make_problem(i, i * 10, [i * 10, i * 10 + 1, i * 10 + 2])
        for i in range(1, 6)  # 5 problems x 3 samples = 15 candidates
    ]
    result = run_calibration_audit(problems)
    assert isinstance(result, CalibrationResult)
    assert result.n_candidates == 15
    assert result.n_problems == 5
    assert -1.0 <= result.energy_correctness_spearman <= 1.0
    assert 0.0 <= result.energy_as_correctness_auroc <= 1.0
    assert 0.0 <= result.within_problem_argmin_correct_rate <= 1.0
    assert math.isfinite(result.correct_mean_energy)
    assert math.isfinite(result.incorrect_mean_energy)


def test_run_calibration_audit_energy_gap_sign():
    """REQ-KONA-3450: energy_gap = incorrect_mean - correct_mean (reported, not asserted to be positive)."""
    problems = [_make_problem(i, i, [i, i + 1]) for i in range(1, 4)]
    result = run_calibration_audit(problems)
    assert result.energy_gap == pytest.approx(
        result.incorrect_mean_energy - result.correct_mean_energy
    )
