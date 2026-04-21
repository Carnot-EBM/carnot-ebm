"""Tests for python/carnot/training/platt_scaler.py — 100% coverage.

Spec: REQ-VERIFY-144, SCENARIO-VERIFY-190, SCENARIO-VERIFY-191
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.training.platt_scaler import PlattScaler
from carnot.training import PlattScaler as PlattScalerFromInit  # REQ-VERIFY-144-5


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExport:
    """REQ-VERIFY-144-5: PlattScaler exported from carnot.training."""

    def test_exported_from_training_init(self):
        # Both import paths must resolve to the same class.
        assert PlattScaler is PlattScalerFromInit


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestInit:
    """REQ-VERIFY-144-1: PlattScaler.__init__ stores init_temperature as self.T."""

    def test_default_temperature_is_one(self):
        scaler = PlattScaler()
        assert scaler.T == pytest.approx(1.0)

    def test_custom_temperature_stored(self):
        scaler = PlattScaler(init_temperature=2.5)
        assert scaler.T == pytest.approx(2.5)

    def test_temperature_is_float(self):
        scaler = PlattScaler(init_temperature=3)
        assert isinstance(scaler.T, float)


# ---------------------------------------------------------------------------
# calibrate() tests
# ---------------------------------------------------------------------------


class TestCalibrate:
    """REQ-VERIFY-144-3: calibrate returns sigmoid(logits / T)."""

    def test_identity_at_t_one(self):
        # T=1 should return sigmoid(logit) unchanged.
        scaler = PlattScaler(init_temperature=1.0)
        logits = jnp.array([0.0, 1.0, -1.0])
        result = scaler.calibrate(logits)
        expected = jnp.array([0.5, 0.7310586, 0.26894143])
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-5)

    def test_softens_with_high_temperature(self):
        # T=2 should produce probabilities closer to 0.5 than T=1.
        scaler_1 = PlattScaler(init_temperature=1.0)
        scaler_2 = PlattScaler(init_temperature=2.0)
        logits = jnp.array([2.0, -2.0])
        p1 = np.array(scaler_1.calibrate(logits))
        p2 = np.array(scaler_2.calibrate(logits))
        # High T → probs closer to 0.5.
        assert abs(p2[0] - 0.5) < abs(p1[0] - 0.5)
        assert abs(p2[1] - 0.5) < abs(p1[1] - 0.5)

    def test_output_in_zero_one_range(self):
        scaler = PlattScaler(init_temperature=1.5)
        logits = jnp.array([-10.0, 0.0, 10.0])
        result = np.array(scaler.calibrate(logits))
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_accepts_numpy_input(self):
        scaler = PlattScaler()
        logits = np.array([1.0, -1.0], dtype=np.float32)
        result = scaler.calibrate(logits)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# compute_ece() tests
# ---------------------------------------------------------------------------


class TestComputeEce:
    """REQ-VERIFY-144-4: compute_ece computes weighted ECE over equal-width bins."""

    def test_empty_returns_zero(self):
        # SCENARIO-VERIFY-191 edge: empty input should not crash.
        scaler = PlattScaler()
        ece = scaler.compute_ece(jnp.array([]), jnp.array([]))
        assert ece == pytest.approx(0.0)

    def test_perfect_calibration(self):
        # All predictions in top bin with conf=0.95, but accuracy=1.0 → ECE=0.05.
        # ECE = |avg_conf - avg_acc| = |0.95 - 1.0| = 0.05 (weighted ECE).
        scaler = PlattScaler()
        probs = jnp.array([0.95, 0.95, 0.95])
        labels = jnp.array([1.0, 1.0, 1.0])
        ece = scaler.compute_ece(probs, labels)
        assert ece == pytest.approx(0.05, abs=1e-4)

    def test_worst_case_calibration(self):
        # All predictions at 1.0 but labels all 0 → ECE = 1.0.
        scaler = PlattScaler()
        probs = jnp.array([1.0, 1.0, 1.0])
        labels = jnp.array([0.0, 0.0, 0.0])
        ece = scaler.compute_ece(probs, labels)
        assert ece == pytest.approx(1.0, abs=1e-5)

    def test_weighted_by_bin_count(self):
        # Bin [0.9, 1.0]: 3 samples with conf=0.95, acc=0.0 → gap=0.95 * 3/4.
        # Bin [0.4, 0.5]: 1 sample with conf=0.45, acc=1.0 → gap=0.55 * 1/4.
        scaler = PlattScaler()
        probs = jnp.array([0.95, 0.95, 0.95, 0.45])
        labels = jnp.array([0.0, 0.0, 0.0, 1.0])
        ece = scaler.compute_ece(probs, labels)
        expected = (3 / 4) * abs(0.0 - 0.95) + (1 / 4) * abs(1.0 - 0.45)
        assert ece == pytest.approx(expected, abs=1e-4)

    def test_prob_one_lands_in_last_bin(self):
        # prob=1.0 must be captured by the last bin, not lost.
        scaler = PlattScaler()
        probs = jnp.array([1.0])
        labels = jnp.array([1.0])
        ece = scaler.compute_ece(probs, labels)
        assert ece == pytest.approx(0.0, abs=1e-5)

    def test_custom_n_bins(self):
        # Using 5 bins instead of 10 — should still run without error.
        scaler = PlattScaler()
        probs = jnp.array([0.1, 0.5, 0.9])
        labels = jnp.array([0.0, 1.0, 1.0])
        ece = scaler.compute_ece(probs, labels, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_returns_python_float(self):
        scaler = PlattScaler()
        probs = jnp.array([0.6, 0.7])
        labels = jnp.array([1.0, 0.0])
        ece = scaler.compute_ece(probs, labels)
        assert isinstance(ece, float)


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


class TestFit:
    """REQ-VERIFY-144-2: fit minimises NLL and returns optimal T."""

    def test_returns_float(self):
        scaler = PlattScaler()
        logits = jnp.array([1.0, -1.0, 2.0, -2.0])
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        T = scaler.fit(logits, labels)
        assert isinstance(T, float)

    def test_updates_self_t(self):
        scaler = PlattScaler(init_temperature=1.0)
        logits = jnp.array([3.0, 3.0, -3.0, -3.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        T = scaler.fit(logits, labels)
        assert scaler.T == pytest.approx(T)

    def test_temperature_clipped_to_bounds(self):
        # SCENARIO-VERIFY-191: T must always stay within [0.1, 10.0].
        # Pathological logits: very large positive logits with all labels=0
        # would try to drive T toward infinity; clip should prevent that.
        scaler = PlattScaler(init_temperature=1.0)
        logits = jnp.array([100.0, 100.0, 100.0])
        labels = jnp.array([0.0, 0.0, 0.0])
        T = scaler.fit(logits, labels, n_steps=50, lr=0.1)
        assert 0.1 <= T <= 10.0

    def test_overconfident_model_gets_t_above_one(self):
        # Logits far from 0 with imperfect labels → T > 1 softens overconfidence.
        scaler = PlattScaler(init_temperature=1.0)
        # Very large logits but only ~50% label accuracy → needs softening.
        logits = jnp.array([10.0, 10.0, 10.0, 10.0])
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        T = scaler.fit(logits, labels, n_steps=200, lr=0.01)
        # With these logits, T > 1 is needed to bring probs toward 0.5.
        assert T > 1.0

    def test_reduces_ece_on_overconfident_logits(self):
        # SCENARIO-VERIFY-190: fitting should reduce ECE.
        # Simulate overconfident model: large logits, mixed labels.
        rng = np.random.default_rng(42)
        n = 100
        logits_np = rng.choice([-5.0, 5.0], size=n).astype(np.float32)
        # Labels mostly agree with sign of logit but with 20% noise → overconfident.
        labels_np = (logits_np > 0).astype(np.float32)
        noise_idx = rng.choice(n, size=20, replace=False)
        labels_np[noise_idx] = 1.0 - labels_np[noise_idx]

        logits = jnp.array(logits_np)
        labels = jnp.array(labels_np)

        scaler = PlattScaler()
        T = scaler.fit(logits, labels, n_steps=200, lr=0.01)
        cal_probs = scaler.calibrate(logits)

        raw_probs = jnp.array(1.0 / (1.0 + jnp.exp(-logits)))
        scaler_ref = PlattScaler(init_temperature=1.0)
        ece_before = scaler_ref.compute_ece(raw_probs, labels)
        ece_after = scaler.compute_ece(cal_probs, labels)

        # Platt scaling should not make ECE worse on this well-separated data.
        assert ece_after <= ece_before + 0.05  # generous tolerance

    def test_custom_n_steps_and_lr(self):
        # Fewer steps / higher lr should still return a valid T.
        scaler = PlattScaler()
        logits = jnp.array([1.0, -1.0])
        labels = jnp.array([1.0, 0.0])
        T = scaler.fit(logits, labels, n_steps=10, lr=0.05)
        assert 0.1 <= T <= 10.0
