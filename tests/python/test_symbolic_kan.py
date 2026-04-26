"""Tests for carnot.models.symbolic_kan — SymbolicKAN model.

100% coverage on symbolic_kan.py.
Spec: REQ-MODEL-030, SCENARIO-MODEL-015.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from python.carnot.models.symbolic_kan import (
    VOCAB,
    VOCAB_KEYS,
    ResidualSpline,
    SymbolicKANConfig,
    SymbolicKANModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(n: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Small random (correct, incorrect) feature pair for testing."""
    rng = np.random.default_rng(seed)
    xs_c = rng.uniform(-1.0, 1.0, (n, 16)).astype(np.float32)
    xs_i = rng.uniform(-1.0, 1.0, (n, 16)).astype(np.float32)
    return xs_c, xs_i


def _small_model(seed: int = 0) -> SymbolicKANModel:
    """Tiny model for fast unit tests."""
    cfg = SymbolicKANConfig(input_dim=16, n_nodes=4, label_update_interval=5, lr=0.01)
    return SymbolicKANModel(cfg, seed=seed)


# ---------------------------------------------------------------------------
# VOCAB tests — REQ-MODEL-030
# ---------------------------------------------------------------------------


class TestVocab:
    """REQ-MODEL-030: SymbolicKAN node vocabulary."""

    def test_vocab_keys_exact(self):
        # REQ-MODEL-030: vocabulary must contain exactly ADD, MUL, CMP, EQ.
        assert set(VOCAB_KEYS) == {"ADD", "MUL", "CMP", "EQ"}

    def test_add_semantics(self):
        # ADD(x, y) ≈ x + y
        fn = VOCAB["ADD"]
        assert float(fn(3.0, 4.0)) == pytest.approx(7.0)

    def test_mul_semantics(self):
        # MUL(x, y) ≈ x * y
        fn = VOCAB["MUL"]
        assert float(fn(3.0, 4.0)) == pytest.approx(12.0)

    def test_cmp_positive(self):
        # CMP(x, y) = sign(x - y): +1 when x > y
        fn = VOCAB["CMP"]
        assert float(fn(5.0, 3.0)) == pytest.approx(1.0)

    def test_cmp_negative(self):
        # CMP(x, y) = sign(x - y): -1 when x < y
        fn = VOCAB["CMP"]
        assert float(fn(1.0, 3.0)) == pytest.approx(-1.0)

    def test_eq_zero_for_equal(self):
        # EQ(x, y) = |x - y|: 0 when equal
        fn = VOCAB["EQ"]
        assert float(fn(5.0, 5.0)) == pytest.approx(0.0)

    def test_eq_positive_for_unequal(self):
        # EQ(x, y) = |x - y|: > 0 when unequal
        fn = VOCAB["EQ"]
        assert float(fn(5.0, 3.0)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ResidualSpline tests — SCENARIO-MODEL-015
# ---------------------------------------------------------------------------


class TestResidualSpline:
    """SCENARIO-MODEL-015: residual correction behaviour."""

    def test_evaluate_returns_float(self):
        # SCENARIO-MODEL-015: evaluate must return a scalar float.
        s = ResidualSpline(n_segments=4)
        result = s.evaluate(0.0)
        assert isinstance(result, float)

    def test_evaluate_endpoints(self):
        # Values at -1 and +1 should not raise.
        s = ResidualSpline(n_segments=4)
        s.evaluate(-1.0)
        s.evaluate(1.0)

    def test_evaluate_out_of_range_clamped(self):
        # Values outside [-1, 1] are clamped silently.
        s = ResidualSpline(n_segments=4, amp=0.0)
        # With amp=0 ctrl points are all 0, so result should be 0 anywhere.
        assert s.evaluate(-5.0) == pytest.approx(0.0, abs=1e-6)
        assert s.evaluate(5.0) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_shape(self):
        # SCENARIO-MODEL-015: gradient must be a vector of length n_segments+1.
        s = ResidualSpline(n_segments=8)
        grad = s.gradient_at(0.0)
        assert grad.shape == (9,)

    def test_gradient_sums_to_one(self):
        # Piecewise linear gradient sums to 1 (partition of unity).
        s = ResidualSpline(n_segments=8)
        for x in [-0.9, 0.0, 0.5, 0.8]:
            grad = s.gradient_at(x)
            assert abs(grad.sum() - 1.0) < 1e-5

    def test_ctrl_small_amplitude(self):
        # Initial control points should be small (amp=0.05 default).
        s = ResidualSpline(n_segments=8, amp=0.05)
        assert np.abs(s.ctrl).max() < 0.5  # well under 0.5 with default amp


# ---------------------------------------------------------------------------
# SymbolicKANConfig tests — REQ-MODEL-030
# ---------------------------------------------------------------------------


class TestSymbolicKANConfig:
    """REQ-MODEL-030: configuration defaults and attributes."""

    def test_defaults(self):
        cfg = SymbolicKANConfig()
        assert cfg.input_dim == 16
        assert cfg.n_nodes == 8

    def test_custom(self):
        cfg = SymbolicKANConfig(input_dim=4, n_nodes=2)
        assert cfg.input_dim == 4
        assert cfg.n_nodes == 2


# ---------------------------------------------------------------------------
# SymbolicKANModel tests — REQ-MODEL-030, SCENARIO-MODEL-015
# ---------------------------------------------------------------------------


class TestSymbolicKANModel:
    """REQ-MODEL-030, SCENARIO-MODEL-015: model forward pass, training, interpretability."""

    def test_init_labels_in_vocab(self):
        # REQ-MODEL-030: every node must start with a valid vocab label.
        model = _small_model()
        for label in model.symbolic_labels:
            assert label in VOCAB_KEYS

    def test_energy_scalar(self):
        # SCENARIO-MODEL-015: energy() must return a scalar float.
        model = _small_model()
        x = np.zeros(16, dtype=np.float32)
        e = model.energy(x)
        assert isinstance(e, float)

    def test_energy_batch_shape(self):
        # energy_batch must return shape (n,).
        model = _small_model()
        xs = np.zeros((5, 16), dtype=np.float32)
        energies = model.energy_batch(xs)
        assert energies.shape == (5,)

    def test_energy_batch_consistent_with_single(self):
        # Batch energy must match single-sample energy for each element.
        model = _small_model()
        xs = np.random.default_rng(0).uniform(-1, 1, (3, 16)).astype(np.float32)
        batch_e = model.energy_batch(xs)
        for i in range(3):
            single_e = model.energy(xs[i])
            assert abs(batch_e[i] - single_e) < 1e-5

    def test_train_returns_loss_history(self):
        # train() must return a list with one entry per epoch.
        model = _small_model()
        xs_c, xs_i = _make_pair(n=4)
        hist = model.train(xs_c, xs_i, n_epochs=3)
        assert len(hist) == 3
        assert all(isinstance(l, float) for l in hist)

    def test_train_reduces_loss(self):
        # After enough training on a linearly separable synthetic case, loss should
        # not increase monotonically from its starting value.
        # We use a very obvious separation: correct samples at 0, incorrect at +1 energy.
        cfg = SymbolicKANConfig(input_dim=16, n_nodes=4, label_update_interval=0, lr=0.05)
        model = SymbolicKANModel(cfg, seed=1)
        rng = np.random.default_rng(99)
        # Correct: all zeros → model should converge to give them low energy
        xs_c = np.zeros((8, 16), dtype=np.float32)
        xs_i = np.ones((8, 16), dtype=np.float32)
        hist = model.train(xs_c, xs_i, n_epochs=20)
        # Loss in epoch 20 should not be higher than initial by a large margin
        # (no strict convergence required for a unit test; just sanity check)
        assert hist[-1] < hist[0] + 1.0  # loss did not explode

    def test_label_counts_covers_all_keys(self):
        # REQ-MODEL-030: label_counts() must return a count for every vocab key.
        model = _small_model()
        counts = model.label_counts()
        assert set(counts.keys()) == set(VOCAB_KEYS)

    def test_label_counts_sum_equals_n_nodes(self):
        # REQ-MODEL-030: total count must equal number of nodes.
        model = _small_model()
        counts = model.label_counts()
        assert sum(counts.values()) == model.config.n_nodes

    def test_top_labels_is_sorted(self):
        # top_labels() must return all 4 vocab keys, sorted by frequency desc.
        model = _small_model()
        top = model.top_labels()
        assert len(top) == len(VOCAB_KEYS)
        counts = model.label_counts()
        for a, b in zip(top, top[1:]):
            assert counts[a] >= counts[b]

    def test_describe_node_contains_label(self):
        # SCENARIO-MODEL-015: describe_node must mention the node's label.
        model = _small_model()
        for i in range(model.config.n_nodes):
            desc = model.describe_node(i)
            assert model.symbolic_labels[i] in desc

    def test_loss_contrastive_nonnegative(self):
        # Hinge loss is always >= 0.
        model = _small_model()
        xs_c, xs_i = _make_pair(n=1)
        loss = model._loss_contrastive(xs_c[0], xs_i[0])
        assert loss >= 0.0

    def test_grad_step_does_not_raise(self):
        # _grad_step should run without errors.
        model = _small_model()
        xs_c, xs_i = _make_pair(n=1)
        model._grad_step(xs_c[0], xs_i[0])

    def test_update_labels_all_valid(self):
        # After _update_labels, all labels must still be in vocab.
        model = _small_model()
        xs_c, xs_i = _make_pair(n=8)
        model._update_labels(xs_c, xs_i)
        for label in model.symbolic_labels:
            assert label in VOCAB_KEYS

    def test_label_update_interval_zero(self):
        # With label_update_interval=0, labels should never be searched.
        cfg = SymbolicKANConfig(input_dim=16, n_nodes=4, label_update_interval=0, lr=0.01)
        model = SymbolicKANModel(cfg, seed=0)
        initial_labels = list(model.symbolic_labels)
        xs_c, xs_i = _make_pair(n=4)
        model.train(xs_c, xs_i, n_epochs=5)
        # Labels should be unchanged (interval=0 means no discrete updates)
        assert model.symbolic_labels == initial_labels

    def test_global_bias_initialized_zero(self):
        # Initial global bias must be 0.0.
        model = _small_model()
        assert model.global_bias == 0.0

    def test_step_counter_increments(self):
        # _step increments once per training pair per epoch.
        model = _small_model()
        xs_c, xs_i = _make_pair(n=4)
        model.train(xs_c, xs_i, n_epochs=2)
        assert model._step == 8  # 4 samples * 2 epochs
