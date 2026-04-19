"""Tests for LeWorldModelJEPATrainer, LeWorldModelLoss, gaussian_kl_regularization.

100% coverage for python/carnot/pipeline/lw_jepa_trainer.py.

Spec coverage: REQ-LEARN-046, REQ-LEARN-047, SCENARIO-LEARN-074, SCENARIO-LEARN-075
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.pipeline.lw_jepa_trainer import (
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
    gaussian_kl_regularization,
)


# ---------------------------------------------------------------------------
# Minimal stub predictor — avoids importing JAX in tests
# ---------------------------------------------------------------------------


class _StubPredictor:
    """Minimal stub predictor that returns fixed outputs without JAX/GPU."""

    def __init__(self, output_val: float = 0.5) -> None:
        self._output_val = output_val
        self._train_calls = 0

    def predict(self, embedding):
        return {"arithmetic": self._output_val, "code": self._output_val, "logic": self._output_val}

    def train(self, pairs, n_epochs=1, **kwargs):
        self._train_calls += n_epochs
        return {"macro_auroc": 0.6}


class _StubPredictorWithAUCProgress:
    """Stub predictor whose predictions improve epoch-over-epoch, triggering AUC > 0.5."""

    def __init__(self) -> None:
        self._calls = 0

    def predict(self, embedding):
        # Return high probability for embeddings whose mean > 0 (positive class proxy)
        val = float(np.mean(np.asarray(embedding))) + 0.5
        val = max(0.0, min(1.0, val))
        return {"arithmetic": val, "code": val, "logic": val}

    def train(self, pairs, n_epochs=1, **kwargs):
        self._calls += n_epochs
        return {}


class _BrokenPredictor:
    """Stub predictor whose predict() always raises — tests fallback paths."""

    def predict(self, embedding):
        raise RuntimeError("broken predictor")

    def train(self, pairs, n_epochs=1, **kwargs):
        raise RuntimeError("broken train")


# ---------------------------------------------------------------------------
# gaussian_kl_regularization tests
# ---------------------------------------------------------------------------


class TestGaussianKLRegularization:
    """Spec: REQ-LEARN-046, SCENARIO-LEARN-074"""

    def test_returns_zero_at_origin(self):
        """SCENARIO-LEARN-074: KL(N(0,I)||N(0,I)) = 0."""
        result = gaussian_kl_regularization(0.0, 0.0)
        assert abs(result) < 1e-7, f"Expected 0.0, got {result}"

    def test_returns_zero_for_zero_arrays(self):
        """Array-valued mean=0, log_var=0 also returns 0."""
        mean = np.zeros(16)
        log_var = np.zeros(16)
        result = gaussian_kl_regularization(mean, log_var)
        assert abs(result) < 1e-6

    def test_non_negative_always(self):
        """KL divergence is always >= 0 (Gibbs inequality)."""
        rng = np.random.RandomState(0)
        for _ in range(20):
            mean = rng.randn(8)
            log_var = rng.randn(8)
            result = gaussian_kl_regularization(mean, log_var)
            assert result >= 0.0, f"KL went negative: {result}"

    def test_positive_for_nonzero_mean(self):
        """Non-zero mean should produce KL > 0."""
        result = gaussian_kl_regularization(1.0, 0.0)
        assert result > 0.0

    def test_positive_for_large_log_var(self):
        """Large log_var (large variance) should produce KL > 0."""
        result = gaussian_kl_regularization(0.0, 2.0)
        assert result > 0.0

    def test_formula_correctness(self):
        """Verify formula: 0.5*(exp(lv) + m^2 - 1 - lv) for scalars."""
        m, lv = 1.0, 0.5
        expected = 0.5 * (math.exp(lv) + m**2 - 1.0 - lv)
        result = gaussian_kl_regularization(m, lv)
        assert abs(result - expected) < 1e-7, f"Expected {expected}, got {result}"

    def test_vector_input(self):
        """Sum over elements for vector inputs."""
        mean = np.array([1.0, 0.0])
        log_var = np.array([0.0, 1.0])
        # 0.5 * ( (1+1-1-0) + (e+0-1-1) ) = 0.5 * (1 + (e-2))
        expected = 0.5 * (1.0 + (math.exp(1.0) - 2.0))
        result = gaussian_kl_regularization(mean, log_var)
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# LeWorldModelLoss tests
# ---------------------------------------------------------------------------


class TestLeWorldModelLoss:
    """Spec: REQ-LEARN-046, SCENARIO-LEARN-075"""

    def test_prediction_loss_zero_when_equal(self):
        """MSE is 0 when predicted == actual."""
        loss = LeWorldModelLoss()
        assert loss.prediction_loss(1.0, 1.0) == 0.0

    def test_prediction_loss_non_negative(self):
        """MSE is always >= 0."""
        loss = LeWorldModelLoss()
        for _ in range(10):
            p = np.random.randn(8)
            a = np.random.randn(8)
            assert loss.prediction_loss(p, a) >= 0.0

    def test_regularization_loss_delegates(self):
        """regularization_loss calls gaussian_kl_regularization."""
        loss = LeWorldModelLoss()
        result = loss.regularization_loss(0.0, 0.0)
        assert abs(result) < 1e-7

    def test_total_loss_non_negative(self):
        """SCENARIO-LEARN-075: total_loss >= 0 for any inputs."""
        loss = LeWorldModelLoss(lambda_reg=0.01)
        rng = np.random.RandomState(42)
        for _ in range(20):
            predicted = rng.randn(8)
            actual = rng.randn(8)
            z_mean = rng.randn(8)
            z_log_var = rng.randn(8)
            total = loss.total_loss(predicted, actual, z_mean, z_log_var)
            assert total >= 0.0, f"total_loss went negative: {total}"

    def test_total_loss_equals_pred_plus_lambda_kl(self):
        """total_loss = MSE + lambda * KL for known inputs."""
        lam = 0.1
        loss = LeWorldModelLoss(lambda_reg=lam)
        predicted = np.array([1.0, 2.0])
        actual = np.array([0.0, 0.0])
        z_mean = np.array([1.0, 0.0])
        z_log_var = np.array([0.0, 0.5])
        mse = float(np.mean((predicted - actual) ** 2))
        kl = gaussian_kl_regularization(z_mean, z_log_var)
        expected = mse + lam * kl
        result = loss.total_loss(predicted, actual, z_mean, z_log_var)
        assert abs(result - expected) < 1e-7

    def test_lambda_zero_means_only_mse(self):
        """When lambda_reg=0, total_loss == prediction_loss."""
        loss = LeWorldModelLoss(lambda_reg=0.0)
        predicted = np.array([1.0, 2.0])
        actual = np.array([3.0, 4.0])
        z_mean = np.ones(4)
        z_log_var = np.ones(4)
        assert abs(loss.total_loss(predicted, actual, z_mean, z_log_var) - loss.prediction_loss(predicted, actual)) < 1e-10

    def test_negative_lambda_raises(self):
        """lambda_reg < 0 is invalid."""
        with pytest.raises(ValueError):
            LeWorldModelLoss(lambda_reg=-0.1)

    def test_default_lambda_is_001(self):
        """Default lambda_reg = 0.01 matches LeWorldModel paper."""
        loss = LeWorldModelLoss()
        assert loss.lambda_reg == 0.01


# ---------------------------------------------------------------------------
# LeWorldModelJEPATrainer tests
# ---------------------------------------------------------------------------


class TestLeWorldModelJEPATrainer:
    """Spec: REQ-LEARN-046, REQ-LEARN-047"""

    # --- helpers ---

    def _make_pairs(self, n: int = 10) -> list[dict]:
        rng = np.random.RandomState(0)
        pairs = []
        for i in range(n):
            emb = rng.randn(256).tolist()
            label = i % 2
            pairs.append({
                "embedding": emb,
                "violated_arithmetic": label,
                "violated_code": label,
                "violated_logic": label,
            })
        return pairs

    # --- train_epoch ---

    def test_train_epoch_returns_float(self):
        """train_epoch returns a non-negative float."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(5)
        loss = trainer.train_epoch(pairs)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_epoch_empty_pairs_returns_zero(self):
        """train_epoch on empty list returns 0.0."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        assert trainer.train_epoch([]) == 0.0

    def test_train_epoch_broken_predictor_returns_float(self):
        """Broken predictor is handled gracefully; still returns a float."""
        trainer = LeWorldModelJEPATrainer(_BrokenPredictor())
        pairs = self._make_pairs(3)
        loss = trainer.train_epoch(pairs)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_epoch_uses_custom_loss(self):
        """Custom loss object is used."""
        custom_loss = LeWorldModelLoss(lambda_reg=1.0)
        trainer = LeWorldModelJEPATrainer(_StubPredictor(), loss=custom_loss)
        pairs = self._make_pairs(5)
        loss = trainer.train_epoch(pairs)
        assert loss >= 0.0

    # --- evaluate_auc ---

    def test_evaluate_auc_returns_float_in_unit_interval(self):
        """evaluate_auc returns float in [0, 1]."""
        trainer = LeWorldModelJEPATrainer(_StubPredictorWithAUCProgress())
        pairs = self._make_pairs(20)
        auc = trainer.evaluate_auc(pairs)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_single_pair_returns_05(self):
        """Single pair has undefined AUC — returns 0.5."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(1)
        auc = trainer.evaluate_auc(pairs)
        assert auc == 0.5

    def test_evaluate_auc_empty_returns_05(self):
        """Empty pairs returns 0.5."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        assert trainer.evaluate_auc([]) == 0.5

    def test_evaluate_auc_broken_predictor_returns_05(self):
        """Broken predictor in evaluate_auc falls back to 0.5."""
        trainer = LeWorldModelJEPATrainer(_BrokenPredictor())
        pairs = self._make_pairs(10)
        auc = trainer.evaluate_auc(pairs)
        assert auc == 0.5

    def test_evaluate_auc_pairs_without_label_keys(self):
        """Pairs missing violation keys still return valid AUC."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = [{"embedding": [0.1] * 256}] * 10
        auc = trainer.evaluate_auc(pairs)
        assert auc == 0.5

    # --- train_to_convergence ---

    def test_train_to_convergence_returns_required_keys(self):
        """train_to_convergence returns all required keys."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(20)
        result = trainer.train_to_convergence(pairs, max_epochs=5, patience=3)
        assert "epochs_trained" in result
        assert "final_auc" in result
        assert "loss_history" in result
        assert "converged" in result

    def test_train_to_convergence_epochs_trained_le_max_epochs(self):
        """epochs_trained never exceeds max_epochs."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(10)
        result = trainer.train_to_convergence(pairs, max_epochs=8, patience=2)
        assert result["epochs_trained"] <= 8

    def test_train_to_convergence_loss_history_matches_epochs(self):
        """loss_history length equals epochs_trained."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(10)
        result = trainer.train_to_convergence(pairs, max_epochs=6, patience=3)
        assert len(result["loss_history"]) == result["epochs_trained"]

    def test_train_to_convergence_final_auc_in_unit_interval(self):
        """final_auc is in [0, 1]."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        pairs = self._make_pairs(20)
        result = trainer.train_to_convergence(pairs, max_epochs=5, patience=3)
        assert 0.0 <= result["final_auc"] <= 1.0

    def test_train_to_convergence_converged_true_on_plateau(self):
        """converged=True when loss plateaus (patience triggers before max_epochs)."""
        # The stub predictor always returns the same value, so loss will plateau quickly.
        trainer = LeWorldModelJEPATrainer(_StubPredictor(output_val=0.5))
        pairs = self._make_pairs(10)
        result = trainer.train_to_convergence(pairs, max_epochs=50, patience=5)
        assert result["converged"] is True, f"expected converged=True, got {result}"

    def test_train_to_convergence_broken_train_still_returns(self):
        """If predictor.train() raises, convergence still completes gracefully."""
        trainer = LeWorldModelJEPATrainer(_BrokenPredictor())
        pairs = self._make_pairs(10)
        result = trainer.train_to_convergence(pairs, max_epochs=5, patience=3)
        assert "epochs_trained" in result

    def test_default_loss_is_leworldmodel(self):
        """When no loss is supplied, LeWorldModelLoss() is used."""
        trainer = LeWorldModelJEPATrainer(_StubPredictor())
        assert isinstance(trainer.loss, LeWorldModelLoss)
        assert trainer.loss.lambda_reg == 0.01

    # --- import from pipeline package ---

    def test_importable_from_carnot_pipeline(self):
        """All three symbols are importable from carnot.pipeline."""
        from carnot.pipeline import (  # noqa: PLC0415
            LeWorldModelJEPATrainer as LWT,
            LeWorldModelLoss as LWL,
            gaussian_kl_regularization as gkl,
        )
        assert LWT is LeWorldModelJEPATrainer
        assert LWL is LeWorldModelLoss
        assert gkl is gaussian_kl_regularization
