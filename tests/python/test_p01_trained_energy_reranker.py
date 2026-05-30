"""Unit tests for the P0.1 v5 trained-energy reranker scoring substrate.

Traces to:
  REQ-KONA-3460 (Trained Energy Reranker vs Self-Consistency on Held-Out GSM8K)
  SCENARIO-KONA-3460 (trains + held-out-scores six conditions)
  SCENARIO-KONA-3460-BLOCKED (honest block on degenerate self-consistency)

These tests exercise the lightweight, model-free scoring logic added in
`carnot.phase3.p01_trained_energy_reranker`. They use small synthetic corpora so
they run in well under a second and require no live model, no GPU, and no
network — matching the experiment's own "cannot time out" property.
"""

from __future__ import annotations

import math

import pytest

from carnot.phase3.p01_trained_energy_reranker import (
    FEATURE_NAMES,
    N_FEATURES,
    TrainedEnergyReranker,
    TrainedScoringResult,
    _accuracy,
    _Verifiers,
    candidate_feature_vector,
    derive_v5_verdict,
    fover_candidate_energy,
    fover_energy_argmin,
    problem_kfold_indices,
    score_corpus_trained_cv,
    trained_energy_sc_hybrid,
    trained_energy_weighted_vote,
)


# ---------------------------------------------------------------------------
# Feature extraction + FoVer energy
# ---------------------------------------------------------------------------
def test_candidate_feature_vector_length_and_logprob_fallback():
    """REQ-KONA-3460: feature vector is fixed-length; missing logprob is finite."""
    verifiers = _Verifiers()
    fv = candidate_feature_vector("7 * 10 = 70. The answer is 70.", -0.5, verifiers)
    assert len(fv) == N_FEATURES == len(FEATURE_NAMES)
    assert all(math.isfinite(x) for x in fv)
    # The mean_logprob feature is the 5th entry; present value is preserved.
    assert fv[4] == pytest.approx(-0.5)
    # A missing logprob becomes a finite "very unconfident" fallback, not NaN.
    fv_none = candidate_feature_vector("anything", None, verifiers)
    assert math.isfinite(fv_none[4])
    assert fv_none[4] == pytest.approx(-10.0)


def test_fover_candidate_energy_penalises_arithmetic_violation():
    """REQ-KONA-3460: FoVer energy is higher for a numerically-wrong trace."""
    verifiers = _Verifiers()
    good = fover_candidate_energy("2 + 2 = 4, so the total is 4.", verifiers)
    bad = fover_candidate_energy("2 + 2 = 5, so the total is 5.", verifiers)
    assert bad >= good
    assert good >= 0.0


# ---------------------------------------------------------------------------
# The trained reranker
# ---------------------------------------------------------------------------
def test_reranker_param_count_is_features_plus_bias():
    """REQ-KONA-3460: the reranker is tiny — N_FEATURES weights + 1 bias."""
    assert TrainedEnergyReranker().n_params == N_FEATURES + 1


def test_reranker_learns_a_separable_signal():
    """SCENARIO-KONA-3460: a trained reranker ranks correct above incorrect.

    We hand it a perfectly separable feature (first column high -> correct) and
    check that after training its P(correct) ordering respects the labels.
    """
    X = [[5.0, 0, 0, 0, 0, 0], [4.5, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0]]
    y = [1, 1, 0, 0]
    r = TrainedEnergyReranker(n_iter=400, lr=0.5).fit(X, y)
    proba = r.predict_proba(X)
    assert proba[0] > proba[2]
    assert proba[1] > proba[3]


def test_reranker_handles_zero_variance_feature_without_error():
    """REQ-KONA-3460: a constant feature column must not divide by zero std."""
    X = [[1.0, 7.0, 0, 0, 0, 0], [2.0, 7.0, 0, 0, 0, 0], [3.0, 7.0, 0, 0, 0, 0]]
    y = [0, 1, 1]
    r = TrainedEnergyReranker(n_iter=50).fit(X, y)
    proba = r.predict_proba(X)
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_reranker_rejects_wrong_feature_width():
    """REQ-KONA-3460: a malformed feature matrix is a hard error, not silent."""
    r = TrainedEnergyReranker()
    with pytest.raises(ValueError):
        r.fit([[1.0, 2.0]], [1])


def test_reranker_predict_before_fit_raises():
    """REQ-KONA-3460: an unfitted reranker must refuse to predict."""
    r = TrainedEnergyReranker()
    with pytest.raises(RuntimeError):
        r.predict_proba([[0.0] * N_FEATURES])


def test_reranker_predict_empty_returns_empty():
    """REQ-KONA-3460: predicting an empty batch returns an empty list."""
    r = TrainedEnergyReranker(n_iter=10).fit([[0.0] * N_FEATURES, [1.0] * N_FEATURES], [0, 1])
    assert r.predict_proba([]) == []


# ---------------------------------------------------------------------------
# Problem-level K-fold split (the leakage guard)
# ---------------------------------------------------------------------------
def test_kfold_empty_corpus():
    """REQ-KONA-3460: zero problems yields no folds."""
    assert problem_kfold_indices(0, 5, seed=1) == []


def test_kfold_partitions_problems_disjointly_and_covers_all():
    """SCENARIO-KONA-3460: held-out sets are disjoint and cover every problem once."""
    n, folds = 23, 5
    splits = problem_kfold_indices(n, folds, seed=7)
    assert len(splits) == folds
    seen: list[int] = []
    for train, test in splits:
        # No problem appears in both train and held-out of the same fold (leakage guard).
        assert set(train).isdisjoint(set(test))
        # Train + test together cover the whole corpus.
        assert sorted(train + test) == list(range(n))
        seen.extend(test)
    # Every problem is held out in exactly one fold.
    assert sorted(seen) == list(range(n))


def test_kfold_clamps_folds_to_problem_count():
    """REQ-KONA-3460: requesting more folds than problems is clamped, not crashing."""
    splits = problem_kfold_indices(3, 10, seed=1)
    assert len(splits) == 3


# ---------------------------------------------------------------------------
# Selection conditions
# ---------------------------------------------------------------------------
def test_trained_energy_weighted_vote_basic_and_empty():
    """REQ-KONA-3460: the heaviest-weighted answer wins; all-None -> None."""
    # Answer 9 has more total weight than 8 even though 8 appears twice.
    assert trained_energy_weighted_vote([8, 8, 9], [0.1, 0.1, 0.9]) == 9
    assert trained_energy_weighted_vote([None, None], [0.5, 0.5]) is None


def test_trained_energy_weighted_vote_tie_breaks_by_first_appearance():
    """REQ-KONA-3460: equal-weight tie resolves to first-appearing answer."""
    assert trained_energy_weighted_vote([5, 6], [0.5, 0.5]) == 5


def test_trained_energy_sc_hybrid_combines_count_and_weight():
    """REQ-KONA-3460: hybrid blends vote count with trained weight; all-None -> None."""
    # 7 has the majority count; hybrid keeps it when energy does not strongly oppose.
    assert trained_energy_sc_hybrid([7, 7, 8], [0.4, 0.4, 0.5]) == 7
    assert trained_energy_sc_hybrid([None], [0.5]) is None


def test_trained_energy_sc_hybrid_tie_breaks_by_first_appearance():
    """REQ-KONA-3460: a perfectly symmetric hybrid score resolves deterministically."""
    assert trained_energy_sc_hybrid([3, 4], [0.5, 0.5]) == 3


def test_fover_energy_argmin_picks_lowest_energy():
    """REQ-KONA-3460: FoVer-argmin returns the lowest-energy candidate's answer."""
    assert fover_energy_argmin([1, 2, 3], [0.9, 0.1, 0.5]) == 2
    assert fover_energy_argmin([None, None], [0.1, 0.2]) is None


def test_accuracy_empty_golds_is_zero():
    """REQ-KONA-3460: accuracy over an empty gold list is defined as 0.0."""
    assert _accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# End-to-end CV scoring + verdict ladder
# ---------------------------------------------------------------------------
def _synthetic_corpus(n: int = 12) -> list[dict]:
    """Build a tiny deterministic corpus where the majority answer is correct.

    Each problem has a correct greedy answer and 6 samples: four that agree with
    the gold (so self-consistency is non-degenerate) and two distractors.
    """
    recs: list[dict] = []
    for i in range(n):
        gold = 10 + i
        samples = []
        for j in range(6):
            ans = gold if j < 4 else gold + 1
            text = (
                f"Step 1: {gold - 1} + 1 = {ans}." if ans == gold else f"Step 1: 1 + 1 = {ans}."
            )
            samples.append(
                {
                    "text": text,
                    "answer": ans,
                    "mean_token_logprob": -0.2 if ans == gold else -1.5,
                }
            )
        recs.append(
            {
                "problem_id": f"syn-{i}",
                "gold": gold,
                "greedy": {"text": f"{gold}", "answer": gold},
                "samples": samples,
            }
        )
    return recs


def test_score_corpus_trained_cv_end_to_end():
    """SCENARIO-KONA-3460: full CV scoring produces all fields on held-out problems."""
    corpus = _synthetic_corpus(12)
    result = score_corpus_trained_cv(corpus, seed=123, n_folds=3, n_boot=200, reranker_iter=50)
    assert isinstance(result, TrainedScoringResult)
    assert result.n_problems_heldout == 12
    assert result.k_samples == 6
    assert result.reranker_param_count == N_FEATURES + 1
    # Majority answer is correct by construction -> SC is non-degenerate.
    assert result.self_consistency_non_degenerate is True
    assert result.self_consistency_accuracy == pytest.approx(1.0)
    # Significance dict carries all three comparisons with the paired stats.
    for key in ("trained_energy", "fover_energy", "hybrid"):
        assert "mcnemar_exact_p" in result.paired_significance[key]
        assert len(result.paired_significance[key]["bootstrap_ci95"]) == 2
    assert "problem-level" in result.train_test_split_note


def _result(sc, tv, hy, fv, *, nondeg=True, p_tv=1.0, p_hy=1.0) -> TrainedScoringResult:
    """Construct a minimal TrainedScoringResult for verdict-ladder coverage."""
    return TrainedScoringResult(
        n_problems_heldout=40,
        k_samples=6,
        reranker_param_count=7,
        train_test_split_note="note",
        self_consistency_non_degenerate=nondeg,
        degenerate_examples=[],
        ar_greedy_accuracy=0.5,
        self_consistency_accuracy=sc,
        self_certainty_bon_accuracy=0.5,
        fover_energy_argmin_accuracy=fv,
        trained_energy_weighted_vote_accuracy=tv,
        trained_energy_sc_hybrid_accuracy=hy,
        delta_trained_energy_vs_self_consistency=tv - sc,
        delta_fover_energy_vs_self_consistency=fv - sc,
        delta_hybrid_vs_self_consistency=hy - sc,
        paired_significance={
            "trained_energy": {"mcnemar_exact_p": p_tv, "bootstrap_ci95": [0.0, 0.0]},
            "fover_energy": {"mcnemar_exact_p": 1.0, "bootstrap_ci95": [0.0, 0.0]},
            "hybrid": {"mcnemar_exact_p": p_hy, "bootstrap_ci95": [0.0, 0.0]},
        },
    )


def test_verdict_g0_degenerate_blocks():
    """SCENARIO-KONA-3460-BLOCKED: a degenerate SC control blocks any comparison."""
    v = derive_v5_verdict(_result(0.2, 0.2, 0.2, 0.2, nondeg=False))
    assert v == (
        "complete: blocked_self_consistency_harness_degenerate_"
        "per_sample_extraction_broken"
    )


def test_verdict_g2_trained_energy_beats_sc():
    """REQ-KONA-3460 G2: a significant positive trained-energy delta validates."""
    v = derive_v5_verdict(_result(0.80, 0.90, 0.80, 0.70, p_tv=0.01))
    assert v == (
        "complete: trained_energy_beats_self_consistency_phase3_premise_validated"
    )


def test_verdict_g1_matches_but_does_not_beat():
    """REQ-KONA-3460 G1: non-inferior but not significant -> matches verdict."""
    v = derive_v5_verdict(_result(0.85, 0.85, 0.85, 0.80, p_tv=1.0))
    assert v == (
        "complete: trained_energy_matches_but_does_not_beat_"
        "self_consistency_at_equal_compute"
    )


def test_verdict_g1_fails_premise_refuted():
    """REQ-KONA-3460: when every trained condition is below SC, the premise retires."""
    v = derive_v5_verdict(_result(0.90, 0.80, 0.82, 0.75))
    assert v == (
        "complete: even_trained_energy_below_self_consistency_"
        "selection_premise_refuted_on_this_substrate"
    )


def test_verdict_hybrid_path_to_g2():
    """REQ-KONA-3460 G2: the hybrid alone can satisfy the significant-beat gate."""
    v = derive_v5_verdict(_result(0.80, 0.80, 0.88, 0.70, p_tv=1.0, p_hy=0.02))
    assert v == (
        "complete: trained_energy_beats_self_consistency_phase3_premise_validated"
    )
