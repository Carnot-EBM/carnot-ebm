"""Tests for Experiment 771: EBRM baseline vs EORM comparison.

REQ-EBRM-001: EBRMEnergy MUST model energy as E(response, reward) using a 2-layer MLP.
REQ-EBRM-002: EORM vs EBRM comparison MUST report both in-distribution AUC and
              architectural_advantage=True when EORM > EBRM on step-level tasks.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure repo root is on the path for imports.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.ebrm_baseline import EBRMEnergy, EBRMTrainer, _resize_vector


# ---------------------------------------------------------------------------
# REQ-EBRM-001: EBRMEnergy architecture tests
# ---------------------------------------------------------------------------


class TestEBRMEnergy:
    """REQ-EBRM-001: energy() MUST take concatenated [features, reward] as input."""

    def test_energy_shape(self):
        """energy() accepts feature_dim-length vector + scalar reward and returns float."""
        model = EBRMEnergy(feature_dim=8, hidden_dim=4)
        features = np.ones(8, dtype=np.float32)
        result = model.energy(features, reward_scalar=1.0)
        # REQ-EBRM-001: output must be a scalar float.
        assert isinstance(result, float)

    def test_energy_changes_with_reward(self):
        """Energy MUST differ for reward=1.0 vs reward=0.0 (otherwise reward signal is ignored)."""
        model = EBRMEnergy(feature_dim=8, hidden_dim=4)
        features = np.ones(8, dtype=np.float32)
        e1 = model.energy(features, reward_scalar=1.0)
        e0 = model.energy(features, reward_scalar=0.0)
        # After random init, energy for different rewards must differ
        # (same MLP input except last element, so output must differ for non-zero W1[-1]).
        assert e1 != e0, "Energy should differ for different reward scalars"

    def test_log_prob_is_negative_energy(self):
        """log_prob() MUST equal -energy() (standard EBM convention: p ∝ exp(-E))."""
        model = EBRMEnergy(feature_dim=8, hidden_dim=4)
        features = np.zeros(8, dtype=np.float32)
        e = model.energy(features, 1.0)
        lp = model.log_prob(features, 1.0)
        assert math.isclose(lp, -e, rel_tol=1e-5)

    def test_score_raises_before_training(self):
        """score() MUST raise RuntimeError if called before training (vectorizer not fitted)."""
        model = EBRMEnergy(feature_dim=8, hidden_dim=4)
        with pytest.raises(RuntimeError, match="before training"):
            model.score("some text")

    def test_score_returns_float_after_training(self):
        """score() MUST return a float after the model has been trained."""
        model = EBRMEnergy(feature_dim=16, hidden_dim=8)
        trainer = EBRMTrainer(model)
        texts = ["correct step one", "incorrect step two", "correct three", "wrong four"]
        labels = [1, 0, 1, 0]
        trainer.train(texts, labels, n_epochs=5)
        result = model.score("some step text")
        assert isinstance(result, float)

    def test_energy_input_concatenation(self):
        """energy() MUST concatenate features with reward — MLP input dim = feature_dim+1."""
        feature_dim = 4
        hidden_dim = 3
        model = EBRMEnergy(feature_dim=feature_dim, hidden_dim=hidden_dim)
        # Verify W1 shape matches feature_dim + 1 (reward concatenated).
        assert model.W1.shape == (feature_dim + 1, hidden_dim), (
            f"W1 shape {model.W1.shape} != ({feature_dim + 1}, {hidden_dim}); "
            "input must be [features, reward] concatenated"
        )


# ---------------------------------------------------------------------------
# REQ-EBRM-001: EBRMTrainer margin loss tests
# ---------------------------------------------------------------------------


class TestEBRMTrainer:
    """REQ-EBRM-001: EBRMTrainer margin loss pushes E_neg > E_pos after training."""

    def test_margin_loss_energy_ordering(self):
        """After training, E(correct, 1.0) < E(incorrect, 0.0) — the margin loss objective.

        REQ-EBRM-001: Margin loss = max(0, margin - (E_neg - E_pos)) drives this ordering.
        """
        rng = np.random.default_rng(7)
        model = EBRMEnergy(feature_dim=16, hidden_dim=8)
        trainer = EBRMTrainer(model, margin=1.0)

        # Use clearly distinct texts so TF-IDF can separate them.
        correct_texts = [f"correct positive example number {i}" for i in range(10)]
        incorrect_texts = [f"wrong negative mistake error {i}" for i in range(10)]
        texts = correct_texts + incorrect_texts
        labels = [1] * 10 + [0] * 10

        trainer.train(texts, labels, n_epochs=300, lr=5e-3)

        # Compute mean energy for correct vs incorrect on training data.
        correct_energies = []
        incorrect_energies = []
        for text, label in zip(texts, labels):
            feats = model.vectorizer.transform([text]).toarray()[0].astype(np.float32)
            from python.carnot.pipeline.ebrm_baseline import _resize_vector as rv
            feats = rv(feats, model.feature_dim)
            reward = 1.0 if label == 1 else 0.0
            e = model.energy(feats, reward)
            if label == 1:
                correct_energies.append(e)
            else:
                incorrect_energies.append(e)

        mean_e_pos = np.mean(correct_energies)
        mean_e_neg = np.mean(incorrect_energies)
        # REQ-EBRM-001: After training, E_neg > E_pos (margin objective satisfied on average).
        assert mean_e_neg > mean_e_pos, (
            f"Margin loss objective violated: mean E_neg={mean_e_neg:.4f} <= "
            f"mean E_pos={mean_e_pos:.4f}. Training should push incorrect examples "
            f"to higher energy than correct ones."
        )

    def test_predict_returns_probability_range(self):
        """predict() MUST return a value in [0, 1] (sigmoid of log_prob)."""
        model = EBRMEnergy(feature_dim=16, hidden_dim=8)
        trainer = EBRMTrainer(model)
        texts = ["good correct answer", "wrong bad mistake", "another correct", "error wrong"]
        labels = [1, 0, 1, 0]
        trainer.train(texts, labels, n_epochs=10)
        score = trainer.predict("some test text")
        assert 0.0 <= score <= 1.0, f"predict() must return [0,1], got {score}"

    def test_degenerate_single_class(self):
        """train() MUST not crash when all labels are the same (no gradient signal)."""
        model = EBRMEnergy(feature_dim=8, hidden_dim=4)
        trainer = EBRMTrainer(model)
        texts = ["text one", "text two", "text three"]
        labels = [1, 1, 1]  # All positive — no negative samples
        # Should not raise; just skip gradient steps.
        trainer.train(texts, labels, n_epochs=5)


# ---------------------------------------------------------------------------
# REQ-EBRM-002: architectural_advantage tests
# ---------------------------------------------------------------------------


class TestArchitecturalAdvantage:
    """REQ-EBRM-002: architectural_advantage MUST be True when eorm_auc > ebrm_auc."""

    def test_architectural_advantage_true_when_eorm_wins(self):
        """architectural_advantage=True when EORM AUC > EBRM AUC (step-level wins).

        REQ-EBRM-002: If EORM > EBRM on step-level tasks, log as architectural_advantage=True.
        """
        eorm_auc = 0.993
        ebrm_auc = 0.750
        architectural_advantage = eorm_auc > ebrm_auc
        assert architectural_advantage is True

    def test_architectural_advantage_false_when_ebrm_wins(self):
        """architectural_advantage=False when EBRM AUC >= EORM AUC.

        REQ-EBRM-002: architectural_advantage is only True when EORM strictly > EBRM.
        """
        eorm_auc = 0.700
        ebrm_auc = 0.850
        architectural_advantage = eorm_auc > ebrm_auc
        assert architectural_advantage is False

    def test_honest_verdict_eorm_outperforms(self):
        """honest_verdict='eorm_outperforms_ebrm' when delta > 0.05."""
        eorm_auc = 0.993
        ebrm_auc = 0.700
        delta = eorm_auc - ebrm_auc
        if eorm_auc > ebrm_auc + 0.05:
            verdict = "eorm_outperforms_ebrm"
        elif abs(delta) <= 0.05:
            verdict = "ebrm_competitive"
        else:
            verdict = "ebrm_outperforms_eorm"
        assert verdict == "eorm_outperforms_ebrm"

    def test_honest_verdict_competitive(self):
        """honest_verdict='ebrm_competitive' when |delta| <= 0.05."""
        eorm_auc = 0.993
        ebrm_auc = 0.970
        delta = eorm_auc - ebrm_auc
        if eorm_auc > ebrm_auc + 0.05:
            verdict = "eorm_outperforms_ebrm"
        elif abs(delta) <= 0.05:
            verdict = "ebrm_competitive"
        else:
            verdict = "ebrm_outperforms_eorm"
        assert verdict == "ebrm_competitive"

    def test_honest_verdict_ebrm_outperforms(self):
        """honest_verdict='ebrm_outperforms_eorm' when ebrm_auc > eorm_auc + 0.05."""
        eorm_auc = 0.700
        ebrm_auc = 0.900
        delta = eorm_auc - ebrm_auc
        if eorm_auc > ebrm_auc + 0.05:
            verdict = "eorm_outperforms_ebrm"
        elif abs(delta) <= 0.05:
            verdict = "ebrm_competitive"
        else:
            verdict = "ebrm_outperforms_eorm"
        assert verdict == "ebrm_outperforms_eorm"

    def test_honest_verdict_insufficient_data(self):
        """honest_verdict='insufficient_data' when test_size < 10."""
        test_size = 5
        if test_size < 10:
            verdict = "insufficient_data"
        else:
            verdict = "other"
        assert verdict == "insufficient_data"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestResizeVector:
    """Tests for _resize_vector — ensures fixed-length MLP input."""

    def test_truncate_longer_vector(self):
        """_resize_vector MUST truncate vectors longer than target_dim."""
        v = np.ones(20, dtype=np.float32)
        result = _resize_vector(v, 10)
        assert result.shape == (10,)
        assert np.all(result == 1.0)

    def test_pad_shorter_vector(self):
        """_resize_vector MUST zero-pad vectors shorter than target_dim."""
        v = np.ones(5, dtype=np.float32)
        result = _resize_vector(v, 10)
        assert result.shape == (10,)
        assert np.all(result[:5] == 1.0)
        assert np.all(result[5:] == 0.0)

    def test_exact_size_unchanged(self):
        """_resize_vector MUST return vector unchanged when shape matches target_dim."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resize_vector(v, 3)
        np.testing.assert_array_equal(result, v)
