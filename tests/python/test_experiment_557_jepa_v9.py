"""Tests for train_leworldmodel — 100% targeted coverage for the function added in Exp 557.

Spec: REQ-LEARN-047,
      SCENARIO-LEARN-076, SCENARIO-LEARN-077, SCENARIO-LEARN-078
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.embeddings.jepa_energy import (
    _auc_from_scores,
    _corpus_entry_to_features,
    _leworldmodel_init_params,
    _leworldmodel_forward,
    _leworldmodel_loss,
    train_leworldmodel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(is_correct: bool, types: list[str]) -> dict:
    """Build a minimal FOVERCorpusEntry-compatible dict for testing."""
    return {
        "question": "q",
        "response": "r",
        "model_id": "test",
        "is_correct": is_correct,
        "constraint_types": types,
    }


def _diverse_pairs(n: int = 20) -> list[dict]:
    """Generate n diverse corpus entries with balanced correct/incorrect labels."""
    rng = np.random.RandomState(42)
    pairs = []
    type_pool = ["correct", "incorrect", "not_verifiable"]
    for i in range(n):
        is_correct = bool(i % 2)
        n_steps = rng.randint(2, 8)
        types = [type_pool[rng.randint(0, 3)] for _ in range(n_steps)]
        pairs.append(_make_entry(is_correct, types))
    return pairs


# ---------------------------------------------------------------------------
# _corpus_entry_to_features
# ---------------------------------------------------------------------------


class TestCorpusEntryToFeatures:
    """Covers _corpus_entry_to_features edge cases."""

    def test_empty_constraint_types(self):
        """Entry with no constraint_types produces zero feature vector."""
        import jax.numpy as jnp
        feat = _corpus_entry_to_features({"constraint_types": []})
        assert feat.shape == (4,)
        assert float(jnp.sum(jnp.abs(feat))) < 1e-7

    def test_all_correct_steps(self):
        """All-correct entry: frac_correct=1, others=0."""
        feat = _corpus_entry_to_features({"constraint_types": ["correct"] * 5})
        assert abs(float(feat[0]) - 1.0) < 1e-6
        assert abs(float(feat[1])) < 1e-6
        assert abs(float(feat[2])) < 1e-6

    def test_normalized_n_steps_clips_at_1(self):
        """Entry with >= 20 steps produces norm_n_steps == 1.0."""
        feat = _corpus_entry_to_features({"constraint_types": ["correct"] * 25})
        assert abs(float(feat[3]) - 1.0) < 1e-6

    def test_fraction_sums_to_1(self):
        """frac_correct + frac_incorrect + frac_not_verifiable == 1.0."""
        feat = _corpus_entry_to_features(
            {"constraint_types": ["correct", "incorrect", "not_verifiable"]}
        )
        assert abs(float(feat[0]) + float(feat[1]) + float(feat[2]) - 1.0) < 1e-5

    def test_missing_key_uses_empty(self):
        """Dict without constraint_types key returns zero vector."""
        import jax.numpy as jnp
        feat = _corpus_entry_to_features({})
        assert float(jnp.sum(jnp.abs(feat))) < 1e-7


# ---------------------------------------------------------------------------
# _auc_from_scores
# ---------------------------------------------------------------------------


class TestAucFromScores:
    """Covers the manual AUC implementation."""

    def test_empty_returns_0_5(self):
        assert _auc_from_scores([], []) == 0.5

    def test_all_same_class_returns_0_5(self):
        assert _auc_from_scores([0.9, 0.8], [1.0, 1.0]) == 0.5

    def test_perfect_separation_returns_1(self):
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = [1.0, 1.0, 0.0, 0.0]
        auc = _auc_from_scores(scores, labels)
        assert auc >= 0.99, f"Expected ~1.0 for perfect separation, got {auc}"

    def test_inverted_returns_0(self):
        # Anti-correlated predictor → AUC near 0
        scores = [0.1, 0.2, 0.9, 0.8]
        labels = [1.0, 1.0, 0.0, 0.0]
        auc = _auc_from_scores(scores, labels)
        assert auc <= 0.01, f"Expected ~0 for inverted predictor, got {auc}"

    def test_random_scores_near_0_5(self):
        import random
        random.seed(99)
        scores = [random.random() for _ in range(100)]
        labels = [float(i % 2) for i in range(100)]
        auc = _auc_from_scores(scores, labels)
        assert 0.35 <= auc <= 0.65, f"Random scores should give AUC near 0.5, got {auc}"


# ---------------------------------------------------------------------------
# _leworldmodel_init_params
# ---------------------------------------------------------------------------


class TestLeworldmodelInitParams:
    """Covers parameter initialization."""

    def test_all_keys_present(self):
        import jax.random as jrandom
        params = _leworldmodel_init_params(jrandom.PRNGKey(0))
        assert set(params.keys()) == {"w1", "b1", "w_mu", "b_mu", "w_lv", "b_lv"}

    def test_shapes(self):
        import jax.random as jrandom
        params = _leworldmodel_init_params(jrandom.PRNGKey(0))
        assert params["w1"].shape == (16, 4)
        assert params["b1"].shape == (16,)
        assert params["w_mu"].shape == (1, 16)
        assert params["b_mu"].shape == (1,)
        assert params["w_lv"].shape == (1, 16)
        assert params["b_lv"].shape == (1,)


# ---------------------------------------------------------------------------
# _leworldmodel_forward
# ---------------------------------------------------------------------------


class TestLeworldmodelForward:
    """Covers forward pass outputs."""

    def test_returns_two_arrays(self):
        import jax.random as jrandom
        import jax.numpy as jnp
        params = _leworldmodel_init_params(jrandom.PRNGKey(0))
        x = jnp.zeros(4)
        mu, log_var = _leworldmodel_forward(params, x)
        assert mu.shape == (1,)
        assert log_var.shape == (1,)

    def test_output_is_finite(self):
        import jax.random as jrandom
        import jax.numpy as jnp
        params = _leworldmodel_init_params(jrandom.PRNGKey(1))
        x = jnp.array([0.5, 0.3, 0.2, 0.6])
        mu, log_var = _leworldmodel_forward(params, x)
        assert math.isfinite(float(mu[0]))
        assert math.isfinite(float(log_var[0]))


# ---------------------------------------------------------------------------
# _leworldmodel_loss
# ---------------------------------------------------------------------------


class TestLeworldmodelLoss:
    """Covers the three-output loss function."""

    def test_kl_is_nonnegative(self):
        """KL divergence is always >= 0 by Gibbs inequality — SCENARIO-LEARN-077."""
        import jax.random as jrandom
        import jax.numpy as jnp
        for seed in range(5):
            params = _leworldmodel_init_params(jrandom.PRNGKey(seed))
            x = jnp.array([0.3, 0.4, 0.3, 0.5])
            y = jnp.array([1.0])
            _, _, kl = _leworldmodel_loss(params, x, y, lambda_kl=0.01)
            assert float(kl) >= -1e-6, f"KL should be >= 0, got {float(kl)}"

    def test_total_equals_pred_plus_lambda_kl(self):
        """total == pred + lambda_kl * kl — SCENARIO-LEARN-078."""
        import jax.random as jrandom
        import jax.numpy as jnp
        params = _leworldmodel_init_params(jrandom.PRNGKey(42))
        x = jnp.array([0.5, 0.2, 0.3, 0.4])
        y = jnp.array([0.0])
        lam = 0.01
        total, pred, kl = _leworldmodel_loss(params, x, y, lambda_kl=lam)
        expected = float(pred) + lam * float(kl)
        assert abs(float(total) - expected) < 1e-5, (
            f"total={float(total):.8f} != pred+lam*kl={expected:.8f}"
        )

    def test_pred_loss_is_mse(self):
        """pred_loss is MSE(sigmoid(mu), y), so range is [0, 1]."""
        import jax.random as jrandom
        import jax.numpy as jnp
        params = _leworldmodel_init_params(jrandom.PRNGKey(7))
        x = jnp.array([0.0, 0.0, 0.0, 0.0])
        y = jnp.array([1.0])
        _, pred, _ = _leworldmodel_loss(params, x, y, lambda_kl=0.0)
        # MSE of sigmoid vs 0/1 label must be in [0, 1]
        assert 0.0 <= float(pred) <= 1.0


# ---------------------------------------------------------------------------
# train_leworldmodel — the primary function under test
# ---------------------------------------------------------------------------


class TestTrainLeworldmodel:
    """Covers train_leworldmodel end-to-end — SCENARIO-LEARN-076/077/078."""

    def test_returns_200_epochs(self):
        """History has exactly 200 entries — SCENARIO-LEARN-076."""
        pairs = _diverse_pairs(20)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        assert len(history) == 200, f"Expected 200 epochs, got {len(history)}"

    def test_each_tuple_has_five_elements(self):
        """Each history entry is (epoch, total, pred, kl, auc) — SCENARIO-LEARN-076."""
        pairs = _diverse_pairs(16)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        for entry in history:
            assert len(entry) == 5, f"Expected 5-tuple, got {len(entry)}-tuple: {entry}"

    def test_epoch_indices_are_sequential(self):
        """Epoch index starts at 0 and increments by 1 — SCENARIO-LEARN-076."""
        pairs = _diverse_pairs(10)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        for i, entry in enumerate(history):
            assert entry[0] == i, f"Epoch {i} mismatch: entry[0]={entry[0]}"

    def test_all_losses_finite(self):
        """All loss values in history are finite floats — SCENARIO-LEARN-076."""
        pairs = _diverse_pairs(15)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        for ep, total, pred, kl, auc in history:
            assert math.isfinite(total), f"Epoch {ep}: total loss is not finite"
            assert math.isfinite(pred), f"Epoch {ep}: pred loss is not finite"
            assert math.isfinite(kl), f"Epoch {ep}: kl loss is not finite"
            assert math.isfinite(auc), f"Epoch {ep}: auc is not finite"

    def test_kl_always_nonnegative(self):
        """All kl_loss values are >= 0 — SCENARIO-LEARN-077."""
        pairs = _diverse_pairs(20)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        for ep, _, _, kl, _ in history:
            assert kl >= -1e-6, f"Epoch {ep}: kl_loss={kl} should be >= 0"

    def test_total_equals_pred_plus_lambda_kl(self):
        """total == pred + 0.01 * kl for first epoch — SCENARIO-LEARN-078."""
        pairs = _diverse_pairs(20)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        ep, total, pred, kl, _ = history[0]
        expected = pred + 0.01 * kl
        assert abs(total - expected) < 1e-4, (
            f"Epoch {ep}: total={total:.7f} != pred+lam*kl={expected:.7f}"
        )

    def test_auc_in_unit_interval(self):
        """All AUC values are in [0, 1] — SCENARIO-LEARN-076."""
        pairs = _diverse_pairs(20)
        history = train_leworldmodel(pairs, lambda_kl=0.01)
        for ep, _, _, _, auc in history:
            assert 0.0 <= auc <= 1.0, f"Epoch {ep}: auc={auc} out of [0,1]"

    def test_empty_pairs_returns_200_neutral(self):
        """Empty input returns 200 neutral (0.5 auc) entries without error."""
        history = train_leworldmodel([], lambda_kl=0.01)
        assert len(history) == 200
        for _, total, pred, kl, auc in history:
            assert total == 0.0
            assert auc == 0.5

    def test_lambda_kl_zero_gives_zero_kl_contribution(self):
        """With lambda_kl=0, total should equal pred_loss."""
        pairs = _diverse_pairs(15)
        history = train_leworldmodel(pairs, lambda_kl=0.0)
        for ep, total, pred, kl, _ in history:
            assert abs(total - pred) < 1e-5, (
                f"Epoch {ep}: with lambda_kl=0, total={total} != pred={pred}"
            )
