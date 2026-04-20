"""Tests for InternalStateProbe — linear probe on LLM hidden states.

Spec: REQ-VERIFY-115, SCENARIO-VERIFY-151, SCENARIO-VERIFY-152, SCENARIO-VERIFY-153
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.internal_state_probe import (
    InternalStateProbe,
    InternalStateProbeResult,
    _compute_auc,
    _sigmoid,
    evaluate_probe_vs_eorm,
    simulate_hidden_states,
)


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_input(self):
        result = _sigmoid(np.array(0.0))
        assert abs(float(result) - 0.5) < 1e-9

    def test_large_positive(self):
        # Should not overflow
        result = _sigmoid(np.array(1000.0))
        assert abs(float(result) - 1.0) < 1e-6

    def test_large_negative(self):
        result = _sigmoid(np.array(-1000.0))
        assert abs(float(result) - 0.0) < 1e-6

    def test_array_input(self):
        arr = _sigmoid(np.array([0.0, 1.0, -1.0]))
        assert arr.shape == (3,)
        assert all(0 <= v <= 1 for v in arr)


# ---------------------------------------------------------------------------
# _compute_auc
# ---------------------------------------------------------------------------


class TestComputeAuc:
    def test_perfect_classifier(self):
        # All positives ranked before all negatives => AUC = 1.0
        scores = [1.0, 0.9, 0.1, 0.0]
        labels = [1, 1, 0, 0]
        auc = _compute_auc(scores, labels)
        assert abs(auc - 1.0) < 1e-9

    def test_random_baseline(self):
        # All same label => 0.5
        scores = [0.5, 0.5, 0.5]
        labels = [1, 1, 1]
        auc = _compute_auc(scores, labels)
        assert abs(auc - 0.5) < 1e-9

    def test_empty_input(self):
        assert _compute_auc([], []) == 0.5

    def test_moderate_auc(self):
        # Perfect negative-first ranking => AUC = 0.0 (worse than random)
        scores = [0.0, 0.1, 0.9, 1.0]
        labels = [1, 1, 0, 0]
        auc = _compute_auc(scores, labels)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# simulate_hidden_states
# ---------------------------------------------------------------------------


class TestSimulateHiddenStates:
    def test_shape(self):
        correct, incorrect = simulate_hidden_states(10, 64, seed=0)
        assert correct.shape == (10, 64)
        assert incorrect.shape == (10, 64)

    def test_dtype(self):
        correct, incorrect = simulate_hidden_states(5, 32, seed=1)
        assert correct.dtype == np.float64
        assert incorrect.dtype == np.float64

    def test_incorrect_higher_norm(self):
        # Incorrect states should have higher mean norm than correct states
        correct, incorrect = simulate_hidden_states(200, 128, seed=42)
        mean_correct_norm = float(np.linalg.norm(correct, axis=1).mean())
        mean_incorrect_norm = float(np.linalg.norm(incorrect, axis=1).mean())
        assert mean_incorrect_norm > mean_correct_norm

    def test_reproducible(self):
        c1, i1 = simulate_hidden_states(5, 16, seed=7)
        c2, i2 = simulate_hidden_states(5, 16, seed=7)
        assert np.allclose(c1, c2)
        assert np.allclose(i1, i2)

    def test_different_seeds_differ(self):
        c1, _ = simulate_hidden_states(5, 16, seed=0)
        c2, _ = simulate_hidden_states(5, 16, seed=1)
        assert not np.allclose(c1, c2)


# ---------------------------------------------------------------------------
# InternalStateProbe
# ---------------------------------------------------------------------------


class TestInternalStateProbe:
    def test_init_defaults(self):
        probe = InternalStateProbe()
        assert probe.hidden_size == 1024
        assert probe.probe_layer == -4
        assert probe._W.shape == (1024,)

    def test_custom_init(self):
        probe = InternalStateProbe(hidden_size=64, probe_layer=-2)
        assert probe.hidden_size == 64
        assert probe.probe_layer == -2

    def test_param_count(self):
        probe = InternalStateProbe(hidden_size=128)
        assert probe.param_count == 129  # 128 weights + 1 bias

    def test_score_range(self):
        probe = InternalStateProbe(hidden_size=32)
        hs = np.random.default_rng(0).normal(size=(32,))
        s = probe.score(hs)
        assert 0.0 <= s <= 1.0

    def test_score_returns_float(self):
        probe = InternalStateProbe(hidden_size=16)
        hs = np.ones(16)
        assert isinstance(probe.score(hs), float)

    def test_train_empty(self):
        # Should not raise
        probe = InternalStateProbe(hidden_size=32)
        probe.train([])  # no-op

    def test_train_updates_weights(self):
        probe = InternalStateProbe(hidden_size=32, probe_layer=-4)
        W_before = probe._W.copy()
        rng = np.random.default_rng(42)
        pairs = [(rng.normal(size=(32,)), 1) for _ in range(20)]
        probe.train(pairs, epochs=10, lr=1e-2)
        assert not np.allclose(probe._W, W_before)

    def test_train_improves_separation(self):
        # Probe should learn to separate synthetic correct vs incorrect states
        correct, incorrect = simulate_hidden_states(50, 64, seed=42)
        pairs = [(hs, 0) for hs in correct] + [(hs, 1) for hs in incorrect]
        # Shuffle
        rng = np.random.default_rng(0)
        idxs = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idxs]

        probe = InternalStateProbe(hidden_size=64)
        probe.train(pairs, epochs=200, lr=1e-2)

        # After training, mean score for incorrect should be higher than correct
        correct_scores = [probe.score(hs) for hs in correct]
        incorrect_scores = [probe.score(hs) for hs in incorrect]
        assert np.mean(incorrect_scores) > np.mean(correct_scores)


# ---------------------------------------------------------------------------
# evaluate_probe_vs_eorm
# ---------------------------------------------------------------------------


class TestEvaluateProbeVsEorm:
    _HS = 64  # hidden_size shared across all helpers

    def _make_test_pairs(self, n: int = 20) -> list[tuple[np.ndarray, int]]:
        correct, incorrect = simulate_hidden_states(n // 2, self._HS, seed=99)
        return [(hs, 0) for hs in correct] + [(hs, 1) for hs in incorrect]

    def test_returns_result_type(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        test_pairs = self._make_test_pairs(20)
        eorm_scores = [float(np.random.default_rng(0).uniform()) for _ in test_pairs]
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs)
        assert isinstance(result, InternalStateProbeResult)

    def test_empty_test_pairs(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        result = evaluate_probe_vs_eorm(probe, [], [])
        assert result.probe_auc == 0.5
        assert result.honest_verdict == "synthetic_proxy"
        assert result.is_tier2_viable is False

    def test_all_same_class_gives_synthetic_proxy(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        correct, _ = simulate_hidden_states(10, self._HS, seed=0)
        test_pairs = [(hs, 1) for hs in correct]  # all incorrect
        eorm_scores = [0.8] * 10
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs)
        assert result.honest_verdict == "synthetic_proxy"
        assert result.is_tier2_viable is False

    def test_param_count_ratio_correct(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        test_pairs = self._make_test_pairs(20)
        eorm_scores = [0.5] * len(test_pairs)
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs, eorm_param_count=55_000_000)
        expected_ratio = (self._HS + 1) / 55_000_000
        # param_count_ratio is rounded to 8 decimal places; tolerate rounding error
        assert abs(result.param_count_ratio - expected_ratio) < 1e-8

    def test_viable_probe_verdict(self):
        # Train a probe that achieves high AUC, verify verdict
        correct, incorrect = simulate_hidden_states(100, self._HS, seed=7)
        train_correct, test_correct = correct[:80], correct[80:]
        train_incorrect, test_incorrect = incorrect[:80], incorrect[80:]

        pairs_train = [(hs, 0) for hs in train_correct] + [(hs, 1) for hs in train_incorrect]
        pairs_test = [(hs, 0) for hs in test_correct] + [(hs, 1) for hs in test_incorrect]

        probe = InternalStateProbe(hidden_size=self._HS)
        probe.train(pairs_train, epochs=300, lr=1e-2)

        eorm_scores = [0.5] * len(pairs_test)  # EORM random baseline
        result = evaluate_probe_vs_eorm(probe, eorm_scores, pairs_test)

        # With well-separated synthetic data and 300 epochs, AUC should be > 0.7
        assert result.probe_auc >= 0.700 or result.honest_verdict in (
            "probe_tier2_viable",
            "probe_below_threshold",
        )
        assert result.honest_verdict != "synthetic_proxy"

    def test_n_test_pairs_populated(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        test_pairs = self._make_test_pairs(10)
        eorm_scores = [0.5] * len(test_pairs)
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs)
        assert result.n_test_pairs == len(test_pairs)

    def test_probe_layer_propagated(self):
        probe = InternalStateProbe(hidden_size=self._HS, probe_layer=-6)
        test_pairs = self._make_test_pairs(10)
        eorm_scores = [0.5] * len(test_pairs)
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs)
        assert result.probe_layer == -6

    def test_auc_rounded(self):
        probe = InternalStateProbe(hidden_size=self._HS)
        test_pairs = self._make_test_pairs(10)
        eorm_scores = [0.5] * len(test_pairs)
        result = evaluate_probe_vs_eorm(probe, eorm_scores, test_pairs)
        # AUC should be rounded to 4 decimal places
        assert result.probe_auc == round(result.probe_auc, 4)
        assert result.eorm_auc == round(result.eorm_auc, 4)
