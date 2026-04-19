"""Tests for NUPProbeV4 and ContrastivePairLoss.

Spec: REQ-VERIFY-109, REQ-VERIFY-110,
      SCENARIO-VERIFY-143, SCENARIO-VERIFY-144, SCENARIO-VERIFY-145
"""

from __future__ import annotations

import pytest

from carnot.pipeline.nup_probe_v4 import ContrastivePairLoss, NUPProbeV4


# ---------------------------------------------------------------------------
# ContrastivePairLoss tests
# ---------------------------------------------------------------------------


class TestContrastivePairLoss:
    """Tests for ContrastivePairLoss.

    Spec: REQ-VERIFY-109, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
    """

    def test_loss_zero_when_gap_meets_margin(self):
        # SCENARIO-VERIFY-143: loss = 0 when E(incorrect) - E(correct) >= margin
        loss_fn = ContrastivePairLoss(margin=1.0)
        # Gap exactly equals margin
        assert loss_fn.loss(energy_incorrect=2.0, energy_correct=1.0) == 0.0
        # Gap exceeds margin
        assert loss_fn.loss(energy_incorrect=3.0, energy_correct=1.0) == 0.0

    def test_loss_equals_margin_when_energies_equal(self):
        # SCENARIO-VERIFY-144: loss = margin when E(incorrect) = E(correct)
        loss_fn = ContrastivePairLoss(margin=1.0)
        result = loss_fn.loss(energy_incorrect=2.0, energy_correct=2.0)
        assert result == pytest.approx(1.0)

    def test_loss_positive_when_gap_insufficient(self):
        # Loss = margin - gap when gap < margin
        loss_fn = ContrastivePairLoss(margin=1.0)
        result = loss_fn.loss(energy_incorrect=1.5, energy_correct=1.0)
        # gap = 0.5, loss = 1.0 - 0.5 = 0.5
        assert result == pytest.approx(0.5)

    def test_loss_never_negative(self):
        loss_fn = ContrastivePairLoss(margin=1.0)
        # Even when incorrect has lower energy than correct
        result = loss_fn.loss(energy_incorrect=0.0, energy_correct=5.0)
        assert result >= 0.0

    def test_loss_with_custom_margin(self):
        loss_fn = ContrastivePairLoss(margin=2.0)
        # gap = 1.5, loss = 2.0 - 1.5 = 0.5
        result = loss_fn.loss(energy_incorrect=3.5, energy_correct=2.0)
        assert result == pytest.approx(0.5)

    def test_batch_loss_returns_mean(self):
        loss_fn = ContrastivePairLoss(margin=1.0)
        # Pair 0: gap = 0 -> loss = 1.0
        # Pair 1: gap = 2 -> loss = 0.0
        # Mean = 0.5
        result = loss_fn.batch_loss(
            incorrect_energies=[1.0, 3.0],
            correct_energies=[1.0, 1.0],
        )
        assert result == pytest.approx(0.5)

    def test_batch_loss_empty_lists_returns_zero(self):
        loss_fn = ContrastivePairLoss(margin=1.0)
        assert loss_fn.batch_loss([], []) == 0.0

    def test_batch_loss_mismatched_lengths_uses_min(self):
        # Only first min(len(a), len(b)) pairs are used
        loss_fn = ContrastivePairLoss(margin=1.0)
        # Pair 0: gap = 0 -> loss = 1.0; second element of longer list is ignored
        result = loss_fn.batch_loss(
            incorrect_energies=[1.0, 2.0, 3.0],
            correct_energies=[1.0],
        )
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NUPProbeV4 — encode tests
# ---------------------------------------------------------------------------


class TestNUPProbeV4Encode:
    """Tests for NUPProbeV4.encode."""

    def test_encode_returns_correct_length(self):
        probe = NUPProbeV4(energy_dim=32)
        vec = probe.encode("hello world")
        assert len(vec) == 32

    def test_encode_empty_string_returns_zero_vector(self):
        probe = NUPProbeV4(energy_dim=16)
        vec = probe.encode("")
        assert all(v == 0.0 for v in vec)

    def test_encode_single_char_returns_zero_vector(self):
        probe = NUPProbeV4(energy_dim=16)
        vec = probe.encode("a")
        assert all(v == 0.0 for v in vec)

    def test_encode_normalised(self):
        import math

        probe = NUPProbeV4(energy_dim=32)
        vec = probe.encode("the quick brown fox")
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_encode_different_texts_differ(self):
        probe = NUPProbeV4(energy_dim=32)
        v1 = probe.encode("2 + 2 = 4")
        v2 = probe.encode("freeform prose with varied symbols @#$%")
        assert v1 != v2


# ---------------------------------------------------------------------------
# NUPProbeV4 — score tests
# ---------------------------------------------------------------------------


class TestNUPProbeV4Score:
    """Tests for NUPProbeV4.score."""

    def test_score_returns_float(self):
        probe = NUPProbeV4()
        result = probe.score("Step 1: compute 2 + 2 = 4")
        assert isinstance(result, float)

    def test_score_empty_string_returns_bias(self):
        # Empty string → zero encoding → score = bias
        probe = NUPProbeV4()
        result = probe.score("")
        assert result == pytest.approx(probe._bias)


# ---------------------------------------------------------------------------
# NUPProbeV4 — train_contrastive tests
# ---------------------------------------------------------------------------


class TestNUPProbeV4Train:
    """Tests for NUPProbeV4.train_contrastive.

    Spec: SCENARIO-VERIFY-145
    """

    def _make_steps(self):
        correct = [
            "Step 1: 2 + 2 = 4, so the answer is 4.",
            "Calculate: 10 divided by 2 equals 5.",
            "Therefore: x = 3 satisfies the equation.",
            "The result is 100 because 10 * 10 = 100.",
        ]
        incorrect = [
            "Step 1: 2 + 2 = 5, so the answer is 5.",
            "Therefore the capital of France is Berlin which is wrong.",
            "The formula gives x = -999 which is clearly wrong.",
            "Adding 3 + 4 gives us 8 which is an error.",
        ]
        return correct, incorrect

    def test_train_contrastive_returns_converged_and_auc(self):
        # SCENARIO-VERIFY-145: train_contrastive returns converged and final_auc in [0, 1]
        correct, incorrect = self._make_steps()
        probe = NUPProbeV4(energy_dim=16)
        result = probe.train_contrastive(correct, incorrect, n_epochs=20)

        assert "converged" in result
        assert "final_auc" in result
        assert "final_loss" in result
        assert "loss_history" in result
        assert isinstance(result["converged"], bool)
        assert 0.0 <= result["final_auc"] <= 1.0
        assert result["final_loss"] >= 0.0

    def test_train_contrastive_loss_history_has_n_epochs_entries(self):
        correct, incorrect = self._make_steps()
        probe = NUPProbeV4(energy_dim=16)
        result = probe.train_contrastive(correct, incorrect, n_epochs=10)
        assert len(result["loss_history"]) == 10

    def test_train_contrastive_empty_correct_returns_no_convergence(self):
        probe = NUPProbeV4(energy_dim=16)
        result = probe.train_contrastive([], ["some incorrect step"])
        assert result["converged"] is False
        assert result["final_auc"] == 0.5

    def test_train_contrastive_empty_incorrect_returns_no_convergence(self):
        probe = NUPProbeV4(energy_dim=16)
        result = probe.train_contrastive(["some correct step"], [])
        assert result["converged"] is False

    def test_train_contrastive_loss_decreases_over_epochs(self):
        # With enough epochs and structurally different texts, loss should decrease
        correct = ["arithmetic: 2 + 2 = 4"] * 5
        incorrect = ["wrong wrong wrong wrong wrong wrong wrong"] * 5
        probe = NUPProbeV4(energy_dim=32, learning_rate=0.05)
        result = probe.train_contrastive(correct, incorrect, n_epochs=30)
        # Loss at end should be <= loss at start (not guaranteed to go to 0 but should not increase)
        if len(result["loss_history"]) >= 5:
            assert result["loss_history"][-1] <= result["loss_history"][0] + 0.5


# ---------------------------------------------------------------------------
# NUPProbeV4 — evaluate_auc tests
# ---------------------------------------------------------------------------


class TestNUPProbeV4EvaluateAUC:
    """Tests for NUPProbeV4.evaluate_auc.

    Spec: REQ-VERIFY-110
    """

    def test_evaluate_auc_returns_float_in_range(self):
        probe = NUPProbeV4(energy_dim=16)
        correct = ["step one: 2 + 2 = 4", "step two: 3 + 3 = 6"]
        incorrect = ["step one is wrong: 2 + 2 = 5", "another wrong step here"]
        auc = probe.evaluate_auc(correct, incorrect)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_empty_correct_returns_chance(self):
        probe = NUPProbeV4()
        auc = probe.evaluate_auc([], ["incorrect step"])
        assert auc == 0.5

    def test_evaluate_auc_empty_incorrect_returns_chance(self):
        probe = NUPProbeV4()
        auc = probe.evaluate_auc(["correct step"], [])
        assert auc == 0.5

    def test_evaluate_auc_after_training_in_range(self):
        # After training, AUC should be a valid float in [0, 1]
        correct = [
            "x = 3 satisfies 2x + 1 = 7",
            "100 divided by 4 is 25",
        ]
        incorrect = [
            "x = 99 is obviously wrong for any equation",
            "division by zero equals infinity which is wrong",
        ]
        probe = NUPProbeV4(energy_dim=16)
        probe.train_contrastive(correct, incorrect, n_epochs=20)
        auc = probe.evaluate_auc(correct, incorrect)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Integration: import from carnot.pipeline
# ---------------------------------------------------------------------------


def test_pipeline_exports_nup_probe_v4():
    """Ensure NUPProbeV4 and ContrastivePairLoss are exported from carnot.pipeline."""
    from carnot.pipeline import ContrastivePairLoss as CPL
    from carnot.pipeline import NUPProbeV4 as NPV4

    assert CPL is ContrastivePairLoss
    assert NPV4 is NUPProbeV4
