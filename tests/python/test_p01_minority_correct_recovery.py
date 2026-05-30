"""Unit tests for p01_minority_correct_recovery.

Traces to:
  REQ-KONA-3473 (Process-energy minority-correct recovery analysis)
  SCENARIO-KONA-3473 (energy recovers minority-correct problems)
  SCENARIO-KONA-3473-BLOCKED (honest block on small corpus)

All tests use small synthetic data so they run in milliseconds with no
live model or filesystem dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.phase3.p01_minority_correct_recovery import (  # noqa: E402
    MinorityCorrectResult,
    _majority_vote_answer,
    binary_auroc,
    compute_minority_correct_recovery,
    spearman_correlation,
)


# ---------------------------------------------------------------------------
# binary_auroc
# ---------------------------------------------------------------------------

def test_auroc_perfect_classifier():
    # REQ-KONA-3473: a classifier that always scores positives above negatives
    # should yield AUROC = 1.0.
    scores = [1.0, 1.0, 0.0, 0.0]
    labels = [1, 1, 0, 0]
    assert binary_auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_random_classifier():
    # A classifier that scores all examples the same yields AUROC = 0.5 (ties).
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert binary_auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_no_positives_returns_half():
    # SCENARIO-KONA-3473: degenerate case — no positive examples is handled
    # gracefully (0.5 = undefined but safe).
    assert binary_auroc([1.0, 0.5, 0.0], [0, 0, 0]) == pytest.approx(0.5)


def test_auroc_no_negatives_returns_half():
    # Same degenerate case on the other side.
    assert binary_auroc([1.0, 0.5, 0.0], [1, 1, 1]) == pytest.approx(0.5)


def test_auroc_worst_classifier():
    # A classifier that ranks negatives above positives yields AUROC ~ 0.
    scores = [0.0, 0.0, 1.0, 1.0]
    labels = [1, 1, 0, 0]
    assert binary_auroc(scores, labels) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive():
    # REQ-KONA-3473: perfectly aligned ranks → ρ = 1.0.
    assert spearman_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    # Perfectly anti-correlated → ρ = −1.0.
    assert spearman_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_degenerate_constant():
    # A constant sequence has zero variance — returns 0.0 without crashing.
    assert spearman_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_spearman_two_elements():
    # Edge: minimum valid input.
    assert spearman_correlation([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_spearman_one_element_returns_zero():
    # < 2 observations → 0.0 (no meaningful rank correlation).
    assert spearman_correlation([1.0], [1.0]) == pytest.approx(0.0)


def test_spearman_length_mismatch_returns_zero():
    # Mismatched lengths are handled gracefully.
    assert spearman_correlation([1.0, 2.0], [1.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _majority_vote_answer
# ---------------------------------------------------------------------------

def test_majority_vote_clear_winner():
    # REQ-KONA-3473: the answer with the most votes wins.
    assert _majority_vote_answer(["7", "7", "9"]) == "7"


def test_majority_vote_skips_none():
    # None values are ignored in the vote count.
    assert _majority_vote_answer([None, "5", "5"]) == "5"


def test_majority_vote_all_none_returns_none():
    # All-None → no valid answer → None.
    assert _majority_vote_answer([None, None]) is None


def test_majority_vote_tie_picks_first():
    # On a tie the first-appearing answer wins (deterministic).
    assert _majority_vote_answer(["a", "b", "a", "b"]) == "a"


# ---------------------------------------------------------------------------
# compute_minority_correct_recovery — synthetic corpora
# ---------------------------------------------------------------------------

def _make_record(gold: str, answers: list[str | None]) -> dict:
    return {
        "gold": gold,
        "samples": [{"answer": a} for a in answers],
    }


def test_minority_fraction_zero_when_sc_always_correct():
    # SCENARIO-KONA-3473: if SC picks the right answer on every problem,
    # minority_correct_fraction = 0.  No energy can improve on SC here.
    records = [
        _make_record("42", ["42", "42", "99"]),  # SC="42" correct
        _make_record("7", ["7", "7", "99"]),      # SC="7" correct
    ]
    proc_es = [[0.1, 0.1, 0.9], [0.2, 0.2, 0.8]]
    trained_ps = [[0.8, 0.8, 0.1], [0.9, 0.9, 0.1]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.minority_correct_fraction == pytest.approx(0.0)
    assert result.n_minority_correct_problems == 0
    # Recovery rate is undefined (0/0) — should return 0.0.
    assert result.minority_correct_recovery_rate_process == pytest.approx(0.0)


def test_minority_fraction_one_when_sc_always_wrong():
    # When SC picks the wrong answer on every problem,
    # minority_correct_fraction = 1.0 — maximum headroom.
    records = [
        _make_record("42", ["99", "99", "42"]),  # SC="99" wrong, correct is minority
        _make_record("7", ["99", "99", "7"]),     # SC="99" wrong
    ]
    proc_es = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    trained_ps = [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.minority_correct_fraction == pytest.approx(1.0)
    assert result.n_minority_correct_problems == 2


def test_process_recovery_perfect_when_energy_always_picks_correct():
    # REQ-KONA-3473: if the process energy argmin always picks the correct
    # candidate on minority-correct problems, recovery rate = 1.0.
    records = [
        # SC picks "99" (majority = wrong); correct is "42" at index 2.
        # Process energy: index 2 has the LOWEST energy (0.1 < 0.5 < 0.5).
        _make_record("42", ["99", "99", "42"]),
    ]
    proc_es = [[0.5, 0.5, 0.1]]  # lowest energy at index 2 (correct answer)
    trained_ps = [[0.1, 0.1, 0.9]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.minority_correct_fraction == pytest.approx(1.0)
    assert result.minority_correct_recovery_rate_process == pytest.approx(1.0)


def test_trained_recovery_perfect_when_proba_picks_correct():
    # Same as above but for the trained energy (highest P(correct) at correct idx).
    records = [_make_record("42", ["99", "99", "42"])]
    proc_es = [[0.5, 0.5, 0.9]]   # process energy picks wrong (highest at idx 2 = 0.9, but argmin picks 0.5 at 0 or 1)
    trained_ps = [[0.1, 0.1, 0.9]]  # trained P(correct) highest at idx 2 (correct)
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.minority_correct_recovery_rate_trained == pytest.approx(1.0)


def test_recovery_zero_when_energy_never_picks_correct():
    # SCENARIO-KONA-3473: if the energy always picks the wrong candidate,
    # recovery rate = 0.0.
    records = [_make_record("42", ["99", "99", "42"])]
    # process energy lowest at index 0 (wrong answer "99")
    proc_es = [[0.1, 0.5, 0.9]]
    trained_ps = [[0.9, 0.1, 0.1]]  # trained picks idx 0 (wrong)
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.minority_correct_recovery_rate_process == pytest.approx(0.0)
    assert result.minority_correct_recovery_rate_trained == pytest.approx(0.0)


def test_auroc_above_half_when_lower_energy_is_correct():
    # REQ-KONA-3473: when lower process energy correlates with correctness,
    # process_energy_correctness_auroc > 0.5.
    #
    # Problem 1: correct answer at idx 0 (low energy), wrong at idx 1 (high energy).
    # Problem 2: correct answer at idx 2 (low energy), wrong at idx 0, 1 (high energy).
    records = [
        _make_record("42", ["42", "99"]),
        _make_record("7", ["99", "99", "7"]),
    ]
    proc_es = [[0.1, 0.9], [0.8, 0.7, 0.1]]  # lowest energy at correct candidate
    trained_ps = [[0.9, 0.1], [0.1, 0.2, 0.9]]  # highest P(correct) at correct candidate
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.process_energy_correctness_auroc > 0.5
    assert result.trained_energy_correctness_auroc > 0.5


def test_within_problem_argmin_correct_rate_tracks_energy_quality():
    # REQ-KONA-3473: argmin-correct rate equals the fraction of problems where
    # the lowest-energy candidate is the correct one.
    records = [
        _make_record("42", ["42", "99"]),    # correct at idx 0
        _make_record("7", ["99", "7"]),       # correct at idx 1
        _make_record("3", ["99", "88"]),      # no correct candidate
    ]
    # Problem 0: lowest energy at idx 0 (correct) → hit
    # Problem 1: lowest energy at idx 1 (correct) → hit
    # Problem 2: lowest energy at idx 0 (wrong) → miss
    proc_es = [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
    trained_ps = [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    # 2 hits out of 3 problems
    assert result.within_problem_argmin_correct_rate_process == pytest.approx(2.0 / 3.0)


def test_n_candidates_and_problems_are_accurate():
    # REQ-KONA-3473: the returned counts reflect the actual corpus dimensions.
    records = [
        _make_record("1", ["1", "2", "3"]),
        _make_record("4", ["4", "5"]),
    ]
    proc_es = [[0.1, 0.2, 0.3], [0.1, 0.2]]
    trained_ps = [[0.9, 0.8, 0.7], [0.9, 0.8]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.n_problems == 2
    assert result.n_candidates == 5  # 3 + 2


def test_empty_proc_energy_list_does_not_crash():
    # Defensive: an empty process-energy list per problem yields False for argmin.
    records = [_make_record("42", ["42", "99"])]
    proc_es = [[]]  # edge case: no per-step scores computed
    trained_ps = [[0.9, 0.1]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    assert result.within_problem_argmin_correct_rate_process == pytest.approx(0.0)


def test_none_answers_handled_in_candidates():
    # SCENARIO-KONA-3473-BLOCKED: if all answers are None, the problem contributes
    # no positive labels and SC also returns None (wrong, so it's minority-correct
    # with denominator 0 for recovery, returned as 0.0).
    records = [_make_record("42", [None, None])]
    proc_es = [[0.5, 0.6]]
    trained_ps = [[0.4, 0.3]]
    result = compute_minority_correct_recovery(records, proc_es, trained_ps)
    # SC picks None → not correct → minority problem; but recovery picks None too
    assert result.n_minority_correct_problems == 1
    assert result.minority_correct_recovery_rate_process == pytest.approx(0.0)
    assert result.minority_correct_recovery_rate_trained == pytest.approx(0.0)
