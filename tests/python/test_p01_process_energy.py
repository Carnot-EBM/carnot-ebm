"""Tests for the P0.1 v6 process-energy + optimal-aggregation substrate.

Spec: REQ-KONA-3472, SCENARIO-KONA-3472, SCENARIO-KONA-3472-BLOCKED

These tests exercise the v6-specific logic ADDED on top of the v5 trained
reranker: the per-step process energy, the optimal SC+energy aggregator (with its
train-fold-only lambda fit), the tautology-clean flip-count metric, the
seven-condition cross-validated scoring, the headroom gate, and the verdict
ladder. We use small synthetic corpora so the logic is verified deterministically
without loading a live model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.phase3.p01_process_energy import (  # noqa: E402
    HEADROOM_HIGH,
    HEADROOM_LOW,
    FlipMetrics,
    _accuracy,
    _aggregate_with_lambda,
    _candidate_steps,
    _Verifiers,
    derive_v6_verdict,
    fit_optimal_lambda,
    flip_metrics,
    optimal_aggregate,
    process_energy_argmin,
    process_energy_per_step,
    score_corpus_process_cv,
)


def test_accuracy_empty_golds_is_zero():
    # Guard: no held-out problems -> 0.0 accuracy (avoids division by zero).
    assert _accuracy([], []) == 0.0


def test_candidate_steps_falls_back_to_parsing_text():
    # A sample missing its `steps` list re-derives steps from `text` (older shards).
    parsed = _candidate_steps({"text": "Step 1: do a thing.\nStep 2: conclude."})
    assert isinstance(parsed, list) and len(parsed) >= 1
    # And an explicit non-empty `steps` list is used verbatim (as strings).
    assert _candidate_steps({"steps": ["a", "b"], "text": "ignored"}) == ["a", "b"]


# ---------------------------------------------------------------------------
# Per-step process energy
# ---------------------------------------------------------------------------
def test_process_energy_empty_steps_is_zero():
    # SCENARIO-KONA-3472: no steps means no detectable step-level violation.
    verifiers = _Verifiers()
    assert process_energy_per_step([], verifiers) == 0.0


def test_process_energy_nonnegative_and_finite():
    # REQ-KONA-3472: the aggregated process energy is a finite real number.
    verifiers = _Verifiers()
    energy = process_energy_per_step(
        ["First, 2 + 2 = 4.", "Therefore the answer is 4."], verifiers
    )
    assert energy == pytest.approx(energy)  # finite (not NaN)
    assert energy >= 0.0


def test_process_energy_argmin_picks_lowest_and_skips_none():
    # The per-step condition selects the lowest-energy candidate, skipping Nones.
    answers = [None, "7", "9", "7"]
    energies = [0.0, 5.0, 1.0, 2.0]
    # None is skipped despite energy 0.0; lowest valid is "9" at 1.0.
    assert process_energy_argmin(answers, energies) == "9"


def test_process_energy_argmin_all_none_returns_none():
    assert process_energy_argmin([None, None], [1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# Optimal SC + energy aggregation
# ---------------------------------------------------------------------------
def test_aggregate_lambda_zero_is_majority_vote():
    # lambda=0 -> pure self-consistency (vote count), ignores energy mass.
    answers = ["a", "a", "b"]
    proba = [0.01, 0.01, 0.99]  # energy strongly favours "b"
    assert _aggregate_with_lambda(answers, proba, 0.0) == "a"


def test_aggregate_lambda_one_is_pure_energy_mass():
    # lambda=1 -> pure energy mass; "b" wins on P(correct) despite fewer votes.
    answers = ["a", "a", "b"]
    proba = [0.01, 0.01, 0.99]
    assert _aggregate_with_lambda(answers, proba, 1.0) == "b"


def test_aggregate_skips_none_answers():
    answers = [None, "x", "x"]
    proba = [0.9, 0.1, 0.1]
    assert _aggregate_with_lambda(answers, proba, 0.5) == "x"


def test_aggregate_all_none_returns_none():
    assert _aggregate_with_lambda([None, None], [0.5, 0.5], 0.3) is None


def test_aggregate_tie_breaks_by_first_appearance():
    # Equal count and equal mass -> deterministic first-appearance winner.
    answers = ["q", "r"]
    proba = [0.5, 0.5]
    assert _aggregate_with_lambda(answers, proba, 0.5) == "q"


def test_fit_optimal_lambda_prefers_energy_when_it_helps_on_train():
    # One problem: majority is wrong ("a"x2), energy favours the correct "b".
    train_answers = [["a", "a", "b"]]
    train_proba = [[0.05, 0.05, 0.95]]
    train_golds = ["b"]
    lam = fit_optimal_lambda(train_answers, train_proba, train_golds)
    # A positive lambda is needed to recover "b"; pure SC (lambda 0) is wrong.
    assert lam > 0.0
    assert optimal_aggregate(train_answers[0], train_proba[0], lam) == "b"


def test_fit_optimal_lambda_prefers_smallest_on_tie():
    # When energy never helps, the smallest lambda (closest to SC) is chosen.
    train_answers = [["a", "a", "b"]]
    train_proba = [[0.9, 0.9, 0.1]]  # energy agrees with majority -> no flip ever
    train_golds = ["a"]
    lam = fit_optimal_lambda(train_answers, train_proba, train_golds)
    assert lam == 0.0


def test_fit_optimal_lambda_empty_golds_returns_zero():
    assert fit_optimal_lambda([], [], []) == 0.0


# ---------------------------------------------------------------------------
# Flip-count metric (tautology-clean)
# ---------------------------------------------------------------------------
def test_flip_metrics_counts_correct_and_incorrect_flips():
    # SCENARIO-KONA-3472: flips split into correct (recovered) and incorrect (cost).
    cond = ["b", "x", "a"]
    sc = ["a", "a", "a"]
    gold = ["b", "z", "a"]
    m = flip_metrics(cond, sc, gold)
    assert m.flip_count == 2  # positions 0 and 1 differ from SC
    assert m.flips_correct == 1  # pos 0: cond "b" == gold "b"
    assert m.flips_incorrect == 1  # pos 1: cond "x" != gold "z"
    assert m.net_correctness_gain == 0


def test_flip_metrics_no_flips_is_all_zero():
    # A condition that agrees with SC everywhere reports zeros — no bit-identical
    # accuracy field to trip the exp3460 tautology flag.
    m = flip_metrics(["a", "b"], ["a", "b"], ["a", "b"])
    assert m == FlipMetrics(0, 0, 0, 0)


def test_flip_metrics_net_gain_positive_when_recovering_minority_correct():
    cond = ["b", "c"]
    sc = ["a", "a"]
    gold = ["b", "c"]
    m = flip_metrics(cond, sc, gold)
    assert m.flip_count == 2
    assert m.net_correctness_gain == 2


# ---------------------------------------------------------------------------
# End-to-end CV scoring on a small synthetic corpus
# ---------------------------------------------------------------------------
def _synthetic_record(pid: str, gold: str, sample_answers: list[str]) -> dict:
    """Build a corpus row with parsed steps + logprobs for each sample answer."""
    samples = []
    for i, ans in enumerate(sample_answers):
        samples.append(
            {
                "text": f"Step one: compute. The answer is {ans}.",
                "answer": ans,
                "steps": [f"Step one: compute candidate {i}.", f"So the answer is {ans}."],
                "n_steps": 2,
                "token_logprobs": [-0.1, -0.2],
                "mean_token_logprob": -0.15 - 0.01 * i,
                "n_tokens": 2,
            }
        )
    return {
        "problem_id": pid,
        "question": f"question {pid}",
        "gold": gold,
        "level": "Level 5",
        "greedy": {
            "text": f"The answer is {sample_answers[0]}.",
            "answer": sample_answers[0],
            "steps": [f"So the answer is {sample_answers[0]}."],
            "n_steps": 1,
            "mean_token_logprob": -0.2,
        },
        "samples": samples,
        "k": len(samples),
        "temperature": 0.8,
    }


def _headroom_corpus() -> list[dict]:
    """Six problems whose majority vote lands in the headroom band (~0.5)."""
    return [
        # Majority correct (3 of 5 vote gold):
        _synthetic_record("p1", "10", ["10", "10", "10", "2", "3"]),
        _synthetic_record("p2", "20", ["20", "20", "20", "5", "6"]),
        _synthetic_record("p3", "30", ["30", "30", "30", "7", "8"]),
        # Majority wrong (gold is the minority answer):
        _synthetic_record("p4", "40", ["1", "1", "1", "40", "40"]),
        _synthetic_record("p5", "50", ["2", "2", "2", "50", "50"]),
        _synthetic_record("p6", "60", ["3", "3", "3", "60", "60"]),
    ]


def test_score_corpus_process_cv_smoke():
    # SCENARIO-KONA-3472: seven held-out conditions are scored without leakage.
    corpus = _headroom_corpus()
    result = score_corpus_process_cv(corpus, seed=20260602, n_folds=3, n_boot=200)
    assert result.n_problems_heldout == 6
    assert result.k_samples == 5
    assert result.reranker_param_count == 7
    assert result.aggregator_param_count == 1
    # Every condition accuracy is a valid fraction.
    for acc in (
        result.ar_greedy_accuracy,
        result.self_consistency_accuracy,
        result.self_certainty_bon_accuracy,
        result.process_energy_argmin_accuracy,
        result.trained_energy_weighted_vote_accuracy,
        result.trained_energy_sc_hybrid_accuracy,
        result.optimal_aggregation_accuracy,
    ):
        assert 0.0 <= acc <= 1.0
    # The deltas are consistent with the reported accuracies.
    assert result.delta_optimal_vs_self_consistency == pytest.approx(
        result.optimal_aggregation_accuracy - result.self_consistency_accuracy
    )
    # Significance dict carries all three comparisons.
    assert set(result.paired_significance) == {
        "optimal_aggregation",
        "process_energy",
        "hybrid",
    }
    assert len(result.fitted_lambdas) == 3  # one lambda per fold


def test_score_corpus_sc_accuracy_in_headroom_band():
    # The synthetic corpus is engineered so SC sits at 0.5 (3 majority-correct of 6).
    corpus = _headroom_corpus()
    result = score_corpus_process_cv(corpus, seed=1, n_folds=3, n_boot=100)
    assert result.self_consistency_accuracy == pytest.approx(0.5)
    assert result.self_consistency_in_headroom_band is True
    assert HEADROOM_LOW <= result.self_consistency_accuracy <= HEADROOM_HIGH


# ---------------------------------------------------------------------------
# Verdict ladder
# ---------------------------------------------------------------------------
def _result_with(**overrides):
    """Build a minimal ProcessScoringResult for verdict tests."""
    from carnot.phase3.p01_process_energy import ProcessScoringResult

    base = dict(
        n_problems_heldout=40,
        k_samples=5,
        reranker_param_count=7,
        aggregator_param_count=1,
        fitted_lambdas=[0.5],
        train_test_split_note="note",
        self_consistency_in_headroom_band=True,
        ar_greedy_accuracy=0.4,
        self_consistency_accuracy=0.5,
        self_certainty_bon_accuracy=0.5,
        process_energy_argmin_accuracy=0.5,
        trained_energy_weighted_vote_accuracy=0.5,
        trained_energy_sc_hybrid_accuracy=0.5,
        optimal_aggregation_accuracy=0.6,
        flip_optimal=FlipMetrics(5, 4, 1, 3),
        flip_process_energy=FlipMetrics(2, 1, 1, 0),
        flip_hybrid=FlipMetrics(1, 1, 0, 1),
        delta_optimal_vs_self_consistency=0.1,
        delta_process_energy_vs_self_consistency=0.0,
        delta_hybrid_vs_self_consistency=0.0,
        paired_significance={
            "optimal_aggregation": {"mcnemar_exact_p": 0.01, "bootstrap_ci95": [0.02, 0.18]},
            "process_energy": {"mcnemar_exact_p": 0.9, "bootstrap_ci95": [-0.1, 0.1]},
            "hybrid": {"mcnemar_exact_p": 0.9, "bootstrap_ci95": [-0.1, 0.1]},
        },
    )
    base.update(overrides)
    return ProcessScoringResult(**base)


def test_verdict_g0_ceiling_blocks():
    r = _result_with(
        self_consistency_in_headroom_band=False, self_consistency_accuracy=0.92
    )
    v = derive_v6_verdict(r)
    assert v.startswith("complete: blocked_corpus_at_ceiling_no_headroom_sc=")
    assert "0.92" in v


def test_verdict_g1_energy_beats_sc():
    r = _result_with()  # positive net gain, positive delta, p<0.05
    assert derive_v6_verdict(r) == (
        "complete: process_energy_beats_self_consistency_with_headroom_"
        "phase3_premise_validated"
    )


def test_verdict_g2_flips_but_no_net_win():
    # Flips occur but the delta is not significant -> changes-selections verdict.
    r = _result_with(
        delta_optimal_vs_self_consistency=0.02,
        paired_significance={
            "optimal_aggregation": {"mcnemar_exact_p": 0.5, "bootstrap_ci95": [-0.1, 0.2]},
            "process_energy": {"mcnemar_exact_p": 0.9, "bootstrap_ci95": [-0.1, 0.1]},
            "hybrid": {"mcnemar_exact_p": 0.9, "bootstrap_ci95": [-0.1, 0.1]},
        },
    )
    assert derive_v6_verdict(r) == (
        "complete: process_energy_changes_selections_but_does_not_beat_"
        "self_consistency_with_headroom"
    )


def test_verdict_no_flips_premise_refuted():
    r = _result_with(
        flip_optimal=FlipMetrics(0, 0, 0, 0),
        optimal_aggregation_accuracy=0.5,
        delta_optimal_vs_self_consistency=0.0,
    )
    assert derive_v6_verdict(r) == (
        "complete: process_energy_does_not_change_selections_selection_premise_"
        "refuted_on_this_substrate"
    )
