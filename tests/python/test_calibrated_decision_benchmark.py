"""Tests for python/carnot/autoresearch/calibrated_decision_benchmark.py --
fitness target #3 (REQ-AUTO-018), the Energy-Based Calibrated-Decision
Training Floor's first concrete benchmark.

Spec: REQ-AUTO-018
"""

from __future__ import annotations

import numpy as np

from carnot.autoresearch import calibrated_decision_benchmark as cdb


def _zero_state() -> dict:
    return {
        "w1": [[0.0] * cdb.INPUT_DIM for _ in range(cdb.HIDDEN_DIM)],
        "b1": [0.0] * cdb.HIDDEN_DIM,
        "w_out": [0.0] * cdb.HIDDEN_DIM,
        "b_out": 0.0,
    }


def _real_state(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "w1": (rng.standard_normal((cdb.HIDDEN_DIM, cdb.INPUT_DIM)) * 0.5).tolist(),
        "b1": [0.0] * cdb.HIDDEN_DIM,
        "w_out": (rng.standard_normal(cdb.HIDDEN_DIM) * 0.5).tolist(),
        "b_out": 0.0,
    }


class TestBucketOf:
    def test_deterministic_across_calls(self) -> None:
        assert cdb._bucket_of("156") == cdb._bucket_of("156")

    def test_in_range(self) -> None:
        for qid in ("1", "2", "abc", "9999", None):
            bucket = cdb._bucket_of(qid)
            assert 0 <= bucket < cdb._BUCKET_MODULUS

    def test_independent_salt_from_verifier_auroc(self) -> None:
        """This module must not silently reproduce
        verifier_auroc_benchmark._bucket_of's exact partition -- confirm at
        least one id lands in a different bucket under the two salts."""
        from carnot.autoresearch import verifier_auroc_benchmark as vab

        qids = [str(i) for i in range(50)]
        assert any(cdb._bucket_of(q) != vab._bucket_of(q) for q in qids)


class TestSplitFeatures:
    def test_missing_corpus_returns_empty_rows(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(cdb, "repo_path", lambda *_parts: tmp_path / "missing.json")
        cdb._load_corpus_rows.cache_clear()
        try:
            assert cdb._load_corpus_rows() == ()
        finally:
            cdb._load_corpus_rows.cache_clear()

    def test_train_and_held_out_features_are_disjoint_in_size(self) -> None:
        train, held_out = cdb._split_features()
        assert len(train) + len(held_out) == len(cdb._pcib_features())

    def test_both_splits_are_non_empty(self) -> None:
        train, held_out = cdb._split_features()
        assert len(train) > 0
        assert len(held_out) > 0

    def test_held_out_has_both_classes(self) -> None:
        _, held_out = cdb._split_features()
        labels = {label for _, _, label in held_out}
        assert 0 in labels
        assert 1 in labels

    def test_split_is_stable_across_calls(self) -> None:
        train1, held_out1 = cdb._split_features()
        train2, held_out2 = cdb._split_features()
        assert train1 == train2
        assert held_out1 == held_out2


class TestTrainFeaturesForPrompt:
    def test_returns_correct_and_incorrect_keys(self) -> None:
        features = cdb.train_features_for_prompt()
        assert set(features.keys()) == {"correct", "incorrect"}

    def test_each_row_is_two_floats(self) -> None:
        features = cdb.train_features_for_prompt()
        for row in features["correct"][:5] + features["incorrect"][:5]:
            assert len(row) == 2
            assert all(isinstance(v, float) for v in row)

    def test_matches_the_train_split_size(self) -> None:
        train, _ = cdb._split_features()
        features = cdb.train_features_for_prompt()
        assert len(features["correct"]) + len(features["incorrect"]) == len(train)


class TestBinaryAuroc:
    def test_perfect_separation_is_one(self) -> None:
        assert cdb._binary_auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0

    def test_ties_count_as_half_a_win(self) -> None:
        assert cdb._binary_auroc([1, 0], [0.5, 0.5]) == 0.5

    def test_no_positive_examples_returns_none(self) -> None:
        assert cdb._binary_auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None


class TestValidateFinalState:
    def test_valid_shape_round_trips(self) -> None:
        state = _real_state()
        result = cdb._validate_final_state(state)
        assert result is not None
        w1, b1, w_out, b_out = result
        assert w1.shape == (cdb.HIDDEN_DIM, cdb.INPUT_DIM)
        assert b1.shape == (cdb.HIDDEN_DIM,)
        assert w_out.shape == (cdb.HIDDEN_DIM,)
        assert isinstance(b_out, float)

    def test_non_dict_is_none(self) -> None:
        assert cdb._validate_final_state([1.0, 2.0]) is None
        assert cdb._validate_final_state(None) is None
        assert cdb._validate_final_state("not a state") is None

    def test_wrong_shape_is_none(self) -> None:
        state = _real_state()
        state["w1"] = [[0.0, 0.0]]  # wrong hidden dim
        assert cdb._validate_final_state(state) is None

    def test_missing_key_is_none(self) -> None:
        state = _real_state()
        del state["b_out"]
        assert cdb._validate_final_state(state) is None

    def test_nan_and_inf_are_none(self) -> None:
        state = _real_state()
        state["b_out"] = float("nan")
        assert cdb._validate_final_state(state) is None
        state = _real_state()
        state["w1"][0][0] = float("inf")
        assert cdb._validate_final_state(state) is None

    def test_out_of_range_is_none(self) -> None:
        state = _real_state()
        state["w_out"][0] = cdb.MAX_ABS_WEIGHT + 1.0
        assert cdb._validate_final_state(state) is None

    def test_overflow_is_none_not_raised(self) -> None:
        state = _real_state()
        state["b_out"] = 10**400
        assert cdb._validate_final_state(state) is None


class TestRecomputeCalibratedDecisionMetrics:
    def test_real_weights_recompute_real_metrics_in_range(self) -> None:
        result = cdb.recompute_calibrated_decision_metrics(_real_state())
        assert result is not None
        assert 0.0 <= result["final_energy"] <= 1.0
        assert 0.0 <= result["brier"] <= 1.0

    def test_malformed_state_returns_none(self) -> None:
        assert cdb.recompute_calibrated_decision_metrics(None) is None
        assert cdb.recompute_calibrated_decision_metrics("not a state") is None
        assert cdb.recompute_calibrated_decision_metrics({}) is None

    def test_deterministic_for_the_same_weights(self) -> None:
        state = _real_state()
        r1 = cdb.recompute_calibrated_decision_metrics(state)
        r2 = cdb.recompute_calibrated_decision_metrics(state)
        assert r1 == r2

    def test_degenerate_zero_weights_are_rejected_by_default(self) -> None:
        """A zero-initialized model scores every held-out row identically
        (energy 0.0 everywhere) -- the exact fabrication class
        verifier_auroc_benchmark.py was hardened against. Must be rejected
        from the accept path."""
        assert cdb.recompute_calibrated_decision_metrics(_zero_state()) is None

    def test_degenerate_weights_are_not_rejected_when_flag_is_false(self) -> None:
        """The seed-measurement path needs the honest chance-level number,
        not a rejection -- see measure_default_calibration_energy."""
        result = cdb.recompute_calibrated_decision_metrics(_zero_state(), reject_degenerate=False)
        assert result is not None
        assert result["final_energy"] == 0.5
        assert result["brier"] == 0.25

    def test_a_real_non_degenerate_state_is_not_rejected(self) -> None:
        assert cdb.recompute_calibrated_decision_metrics(_real_state()) is not None

    def test_empty_held_out_split_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(cdb, "_split_features", lambda: ((), ()))
        assert cdb.recompute_calibrated_decision_metrics(_real_state()) is None

    def test_never_raises_on_pathological_input(self) -> None:
        for bad in (
            {"w1": "not a matrix", "b1": [], "w_out": [], "b_out": 0.0},
            {"w1": [[1.0, 2.0]] * 4, "b1": [1.0] * 4, "w_out": [1.0] * 4, "b_out": object()},
            [],
            {},
        ):
            assert cdb.recompute_calibrated_decision_metrics(bad) is None


class TestMeasureDefaultCalibrationEnergy:
    def test_matches_direct_recompute_with_flag(self) -> None:
        expected = cdb.recompute_calibrated_decision_metrics(_zero_state(), reject_degenerate=False)
        assert cdb.measure_default_calibration_energy() == expected

    def test_is_the_honest_chance_level_numbers(self) -> None:
        """Regression guard: a fresh GibbsModel's zero-initialized output
        layer is exactly the degenerate/chance case (AUROC 0.5, Brier 0.25),
        not a fabricated placeholder -- see module docstring."""
        seed = cdb.measure_default_calibration_energy()
        assert seed["final_energy"] == 0.5
        assert seed["brier"] == 0.25
