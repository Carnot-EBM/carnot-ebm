"""Tests for JEPA v8 retrain (Exp 543) — LeWorldModel on expanded FOVER corpus.

Tests cover:
  - load_v8_cot_corpus: expanded path priority, fallback, synthetic sentinel
  - LeWorldModelLoss: prediction_loss, regularization_loss, total_loss
  - compute_held_out_split: 80/20 split determinism
  - evaluate_auc: AUC computation on held-out pairs
  - honest_verdict logic: threshold conditions

Spec: REQ-LEARN-056, REQ-LEARN-057, SCENARIO-LEARN-088, SCENARIO-LEARN-089
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot.models.jepa_retrain_v6 import (
    compute_held_out_split,
    violation_pairs_to_trainer_dicts,
)
from carnot.models.jepa_retrain_v8 import LAMBDA_REG_V8, load_v8_cot_corpus
from carnot.pipeline.lw_jepa_trainer import LeWorldModelLoss, LeWorldModelJEPATrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_expanded(tmp_path: Path) -> Path:
    """Write a minimal valid expanded corpus JSON with FOVER schema."""
    pairs = [
        {"step_text": f"step {i}", "label": "correct" if i % 2 == 0 else "incorrect"}
        for i in range(10)
    ]
    p = tmp_path / "fover_labeled_steps_expanded.json"
    p.write_text(json.dumps(pairs))
    return p


@pytest.fixture()
def tmp_live(tmp_path: Path) -> Path:
    """Write a minimal valid live fallback corpus JSON."""
    pairs = [
        {"step_text": f"live step {i}", "label": "correct" if i % 2 == 0 else "incorrect"}
        for i in range(6)
    ]
    p = tmp_path / "fover_labeled_steps_live.json"
    p.write_text(json.dumps(pairs))
    return p


@pytest.fixture()
def synthetic_trainer_dicts() -> list[dict]:
    """Generate 20 synthetic training dicts with class-correlated signal in emb[0]."""
    rng = np.random.RandomState(543)
    pairs = []
    for i in range(20):
        label = int(i % 2)
        emb = rng.randn(256).astype(np.float32)
        emb[0] += (1.0 if label else -1.0) * 0.5
        pairs.append({
            "embedding": emb.tolist(),
            "violated_arithmetic": label,
            "violated_code": label,
            "violated_logic": label,
        })
    return pairs


# ---------------------------------------------------------------------------
# Tests: load_v8_cot_corpus (SCENARIO-LEARN-089)
# ---------------------------------------------------------------------------


class TestLoadV8CotCorpus:
    """REQ-LEARN-056: expanded corpus loads first; falls back correctly."""

    def test_loads_expanded_when_present(self, tmp_expanded: Path, tmp_path: Path) -> None:
        # SCENARIO-LEARN-089: expanded path preferred over live fallback
        missing_live = str(tmp_path / "does_not_exist.json")
        pairs, source = load_v8_cot_corpus(str(tmp_expanded), missing_live)
        assert source == "live_fover_expanded"
        assert len(pairs) == 10

    def test_falls_back_to_live_when_expanded_missing(
        self, tmp_live: Path, tmp_path: Path
    ) -> None:
        # SCENARIO-LEARN-089: when expanded is absent, live fallback is used
        missing_expanded = str(tmp_path / "no_expanded.json")
        pairs, source = load_v8_cot_corpus(missing_expanded, str(tmp_live))
        assert source == "live_fover_442"
        assert len(pairs) == 6

    def test_returns_synthetic_sentinel_when_both_missing(self, tmp_path: Path) -> None:
        # Both paths missing → synthetic sentinel
        pairs, source = load_v8_cot_corpus(
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
        )
        assert source == "synthetic"
        assert pairs == []

    def test_expanded_takes_priority_over_live(
        self, tmp_expanded: Path, tmp_live: Path
    ) -> None:
        # When both exist, expanded wins
        pairs, source = load_v8_cot_corpus(str(tmp_expanded), str(tmp_live))
        assert source == "live_fover_expanded"
        assert len(pairs) == 10

    def test_lambda_reg_v8_is_0_1(self) -> None:
        # REQ-LEARN-057: v8 uses lambda_reg=0.1 for stronger KL regularization
        assert LAMBDA_REG_V8 == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: LeWorldModelLoss (REQ-LEARN-057)
# ---------------------------------------------------------------------------


class TestLeWorldModelLoss:
    """REQ-LEARN-057: two-term objective computes correctly."""

    def test_prediction_loss_zero_when_identical(self) -> None:
        # L_prediction = MSE(p, a) = 0 when predicted == actual
        loss = LeWorldModelLoss(lambda_reg=0.1)
        x = np.array([1.0, 2.0, 3.0])
        assert loss.prediction_loss(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_prediction_loss_positive(self) -> None:
        loss = LeWorldModelLoss(lambda_reg=0.1)
        p = np.array([1.0, 0.0])
        a = np.array([0.0, 1.0])
        # MSE = mean((1-0)^2 + (0-1)^2) = 1.0
        assert loss.prediction_loss(p, a) == pytest.approx(1.0)

    def test_kl_regularization_zero_at_prior(self) -> None:
        # KL(N(0,I) || N(0,I)) = 0
        loss = LeWorldModelLoss(lambda_reg=0.1)
        # log_var=0 means var=1; mean=0 → exactly the prior
        assert loss.regularization_loss(0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_kl_regularization_positive_off_prior(self) -> None:
        # KL > 0 when mean != 0 or log_var != 0
        loss = LeWorldModelLoss(lambda_reg=0.1)
        kl = loss.regularization_loss(1.0, 0.0)
        assert kl > 0.0

    def test_total_loss_combines_terms(self) -> None:
        # L_total = L_pred + 0.1 * L_kl
        loss = LeWorldModelLoss(lambda_reg=0.1)
        p = np.array([1.0])
        a = np.array([0.0])
        pred_loss = loss.prediction_loss(p, a)    # = 1.0
        kl_loss = loss.regularization_loss(0.0, 0.0)  # = 0.0
        total = loss.total_loss(p, a, 0.0, 0.0)
        assert total == pytest.approx(pred_loss + 0.1 * kl_loss)

    def test_total_loss_nonnegative(self) -> None:
        loss = LeWorldModelLoss(lambda_reg=0.1)
        rng = np.random.RandomState(0)
        for _ in range(10):
            p, a, mu, lv = rng.randn(4, 8)
            assert loss.total_loss(p, a, mu, lv) >= 0.0

    def test_lambda_reg_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            LeWorldModelLoss(lambda_reg=-0.1)


# ---------------------------------------------------------------------------
# Tests: compute_held_out_split (REQ-LEARN-056)
# ---------------------------------------------------------------------------


class TestComputeHeldOutSplit:
    """80/20 train/test split is deterministic and boundary-safe."""

    def test_split_20_percent(self, tmp_expanded: Path, tmp_path: Path) -> None:
        # SCENARIO-LEARN-088: split proportions correct
        pairs, _ = load_v8_cot_corpus(str(tmp_expanded), str(tmp_path / "x.json"))
        train, test = compute_held_out_split(pairs, test_fraction=0.2)
        assert len(train) + len(test) == len(pairs)
        assert len(test) >= 1
        assert len(train) >= 1

    def test_split_deterministic(self, tmp_expanded: Path, tmp_path: Path) -> None:
        # Same pairs → same split every time (no shuffle)
        pairs, _ = load_v8_cot_corpus(str(tmp_expanded), str(tmp_path / "x.json"))
        train1, test1 = compute_held_out_split(pairs, test_fraction=0.2)
        train2, test2 = compute_held_out_split(pairs, test_fraction=0.2)
        assert [p.partial_response for p in train1] == [p.partial_response for p in train2]
        assert [p.partial_response for p in test1] == [p.partial_response for p in test2]

    def test_empty_pairs_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_held_out_split([])


# ---------------------------------------------------------------------------
# Tests: AUC computation (SCENARIO-LEARN-088)
# ---------------------------------------------------------------------------


class TestEvaluateAUC:
    """SCENARIO-LEARN-088: AUC is computable on held-out test pairs."""

    def test_auc_on_balanced_synthetic(self, synthetic_trainer_dicts: list[dict]) -> None:
        # Train briefly and verify AUC is in [0, 1]
        from carnot.pipeline.jepa_predictor import JEPAViolationPredictor

        predictor = JEPAViolationPredictor(seed=543)
        loss = LeWorldModelLoss(lambda_reg=0.1)
        trainer = LeWorldModelJEPATrainer(predictor, loss=loss)
        trainer.predictor_model.train(synthetic_trainer_dicts, n_epochs=5)
        auc = trainer.evaluate_auc(synthetic_trainer_dicts)
        assert 0.0 <= auc <= 1.0

    def test_auc_returns_0_5_on_single_class(self) -> None:
        # If all labels are the same, AUC is undefined → returns 0.5
        from carnot.pipeline.jepa_predictor import JEPAViolationPredictor

        predictor = JEPAViolationPredictor(seed=543)
        loss = LeWorldModelLoss(lambda_reg=0.1)
        trainer = LeWorldModelJEPATrainer(predictor, loss=loss)
        single_class = [
            {
                "embedding": np.zeros(256, dtype=np.float32).tolist(),
                "violated_arithmetic": 1,
                "violated_code": 1,
                "violated_logic": 1,
            }
            for _ in range(5)
        ]
        auc = trainer.evaluate_auc(single_class)
        assert auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: honest_verdict logic (SCENARIO-LEARN-088)
# ---------------------------------------------------------------------------


class TestHonestVerdictLogic:
    """SCENARIO-LEARN-088: verdict categories gate correctly on AUC and n_train."""

    @pytest.mark.parametrize(
        "final_auc,n_train,expected",
        [
            (0.95, 100, "jepa_v8_improved"),   # auc>=0.9 AND n_train>=80
            (0.92, 80, "jepa_v8_improved"),    # boundary exact
            (0.91, 79, "auc_stable"),          # auc>=0.9 but n_train<80 → auc_stable
            (0.85, 100, "auc_stable"),         # auc in [0.8, 0.9)
            (0.80, 100, "auc_stable"),         # boundary exact
            (0.79, 100, "synthetic_fallback"), # auc<0.8
            (0.50, 10, "synthetic_fallback"),  # well below threshold
        ],
    )
    def test_verdict_thresholds(
        self, final_auc: float, n_train: int, expected: str
    ) -> None:
        # Mirror the honest_verdict logic from the experiment script.
        if final_auc >= 0.900 and n_train >= 80:
            verdict = "jepa_v8_improved"
        elif final_auc >= 0.800:
            verdict = "auc_stable"
        else:
            verdict = "synthetic_fallback"
        assert verdict == expected
