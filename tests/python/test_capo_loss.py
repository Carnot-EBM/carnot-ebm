"""Tests for capo_loss.py — 100% coverage.

Spec: REQ-VERIFY-120, REQ-VERIFY-121,
      SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.training.capo_loss import capo_loss, ece_loss


# ---------------------------------------------------------------------------
# ece_loss tests
# ---------------------------------------------------------------------------


class TestEceLoss:
    """SCENARIO-VERIFY-157: ece_loss basic correctness and edge cases."""

    def test_empty_input_returns_zero(self):
        # REQ-VERIFY-121: safe for empty batches
        probs = jnp.array([])
        labels = jnp.array([])
        result = ece_loss(probs, labels)
        assert float(result) == pytest.approx(0.0)

    def test_perfect_calibration_returns_zero(self):
        # When confidence always matches accuracy, ECE should be zero.
        # 5 predictions all at confidence=1.0 and all correct (label=1).
        probs = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        labels = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = ece_loss(probs, labels)
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_perfect_miscalibration_returns_one(self):
        # Confidence=1.0 but accuracy=0.0 → ECE=1.0
        probs = jnp.array([1.0, 1.0])
        labels = jnp.array([0.0, 0.0])
        result = ece_loss(probs, labels)
        assert float(result) == pytest.approx(1.0, abs=1e-5)

    def test_single_bin_used(self):
        # All probs in [0, 0.1): one non-empty bin, ECE = |0.05 - 0.5|
        probs = jnp.array([0.05, 0.05])
        labels = jnp.array([0.0, 1.0])  # accuracy = 0.5
        result = ece_loss(probs, labels)
        expected = abs(0.05 - 0.5)
        assert float(result) == pytest.approx(expected, abs=1e-5)

    def test_last_bin_includes_prob_one(self):
        # prob=1.0 must land in the last bin (not be excluded).
        probs = jnp.array([1.0])
        labels = jnp.array([1.0])
        result = ece_loss(probs, labels)
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_custom_n_bins(self):
        # With n_bins=1, all predictions in single bin.
        probs = jnp.array([0.3, 0.7])
        labels = jnp.array([0.0, 1.0])
        result = ece_loss(probs, labels, n_bins=1)
        # mean_conf = 0.5, mean_acc = 0.5 → ECE = 0
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_returns_scalar(self):
        probs = jnp.array([0.2, 0.8])
        labels = jnp.array([0.0, 1.0])
        result = ece_loss(probs, labels)
        assert result.shape == ()

    def test_multiple_bins_averaged(self):
        # Two bins each with one prediction: bin[0]=|0.05-0.0|=0.05, bin[9]=|0.95-1.0|=0.05
        probs = jnp.array([0.05, 0.95])
        labels = jnp.array([0.0, 1.0])
        result = ece_loss(probs, labels)
        assert float(result) == pytest.approx(0.05, abs=1e-5)

    def test_exported_from_carnot_training(self):
        # REQ-VERIFY-121: ece_loss exported from carnot.training package
        from carnot.training import ece_loss as imported_fn
        assert imported_fn is ece_loss


# ---------------------------------------------------------------------------
# capo_loss tests
# ---------------------------------------------------------------------------


class TestCapoLoss:
    """SCENARIO-VERIFY-158, SCENARIO-VERIFY-159: capo_loss correctness and edge cases."""

    def test_empty_batch_returns_zero(self):
        # REQ-VERIFY-120: safe for empty batches
        energies = jnp.array([])
        labels = jnp.array([])
        result = capo_loss(energies, labels)
        assert float(result) == pytest.approx(0.0)

    def test_only_correct_no_contrastive_signal(self):
        # No incorrect examples → contrastive term = 0, only calibration.
        energies = jnp.array([0.5, 0.5])
        labels = jnp.array([0, 0])  # all correct
        result = capo_loss(energies, labels)
        # Contrastive = 0; calibration may be non-zero.
        assert float(result) >= 0.0

    def test_only_incorrect_no_contrastive_signal(self):
        energies = jnp.array([0.5, 0.5])
        labels = jnp.array([1, 1])  # all incorrect
        result = capo_loss(energies, labels)
        assert float(result) >= 0.0

    def test_perfectly_ranked_pair_zero_contrastive(self):
        # E_incorrect - E_correct >> margin → hinge = 0 → L_contrastive = 0
        energies = jnp.array([0.0, 10.0])  # correct=0.0, incorrect=10.0
        labels = jnp.array([0, 1])
        result = capo_loss(energies, labels, margin=1.0, lambda_calib=0.0)
        assert float(result) == pytest.approx(0.0, abs=1e-5)

    def test_worst_ranked_pair_full_margin_penalty(self):
        # E_incorrect < E_correct → gap is negative → hinge = margin - gap > margin
        energies = jnp.array([10.0, 0.0])  # correct=10.0, incorrect=0.0
        labels = jnp.array([0, 1])
        result = capo_loss(energies, labels, margin=1.0, lambda_calib=0.0)
        # gap = 0.0 - 10.0 = -10.0 → loss = max(0, 1 - (-10)) = 11.0
        assert float(result) == pytest.approx(11.0, abs=1e-4)

    def test_lambda_calib_zero_disables_calibration(self):
        # With lambda_calib=0, result should be pure contrastive loss.
        energies = jnp.array([0.0, 0.5])  # gap=0.5 < margin=1.0
        labels = jnp.array([0, 1])
        result = capo_loss(energies, labels, margin=1.0, lambda_calib=0.0)
        expected_contrastive = max(0.0, 1.0 - 0.5)
        assert float(result) == pytest.approx(expected_contrastive, abs=1e-5)

    def test_lambda_calib_nonzero_adds_calibration(self):
        # With lambda_calib>0, result should be strictly >= contrastive alone.
        energies = jnp.array([0.0, 0.5])
        labels = jnp.array([0, 1])
        result_no_calib = capo_loss(energies, labels, margin=1.0, lambda_calib=0.0)
        result_with_calib = capo_loss(energies, labels, margin=1.0, lambda_calib=0.1)
        assert float(result_with_calib) >= float(result_no_calib)

    def test_all_pairs_in_batch(self):
        # 2 correct, 2 incorrect → 4 pairs, all with gap=0 → loss = 4 * margin / 4
        energies = jnp.array([0.0, 0.0, 0.0, 0.0])
        labels = jnp.array([0, 0, 1, 1])
        result = capo_loss(energies, labels, margin=1.0, lambda_calib=0.0)
        assert float(result) == pytest.approx(1.0, abs=1e-5)

    def test_returns_scalar(self):
        energies = jnp.array([1.0, -1.0])
        labels = jnp.array([0, 1])
        result = capo_loss(energies, labels)
        assert result.shape == ()

    def test_exported_from_carnot_training(self):
        # REQ-VERIFY-120: capo_loss exported from carnot.training package
        from carnot.training import capo_loss as imported_fn
        assert imported_fn is capo_loss

    def test_nonnegative_always(self):
        # Loss must always be >= 0
        energies = jnp.array([-5.0, -3.0, 2.0, 4.0])
        labels = jnp.array([0, 1, 0, 1])
        result = capo_loss(energies, labels)
        assert float(result) >= 0.0
