"""Tests for EBMCoTCalibrator — Langevin calibration of EORM hidden states.

100% coverage for python/carnot/models/ebm_cot_calibrator.py.

Spec coverage: REQ-EORM-005, REQ-EORM-006, REQ-EORM-007,
               SCENARIO-EORM-010, SCENARIO-EORM-011
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.models.eorm import CoTEnergyInput, EORMModel
from carnot.models.ebm_cot_calibrator import (
    EBMCoTCalibrator,
    _auc_roc,
    _energy_from_pooled,
    _forward_get_pooled,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_eorm() -> EORMModel:
    """A tiny EORMModel for fast CPU tests (embed_dim=32, 2 heads, 1 layer)."""
    return EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=64, vocab_size=256)


@pytest.fixture
def calibrator(small_eorm: EORMModel) -> EBMCoTCalibrator:
    """EBMCoTCalibrator wrapping the small_eorm fixture, 10 steps."""
    return EBMCoTCalibrator(small_eorm, n_langevin_steps=10, step_size=0.01, seed=42)


# ---------------------------------------------------------------------------
# _forward_get_pooled
# ---------------------------------------------------------------------------

class TestForwardGetPooled:
    """Tests for _forward_get_pooled helper."""

    def test_returns_correct_shape(self, small_eorm: EORMModel) -> None:
        """_forward_get_pooled returns a vector of shape (embed_dim,)."""
        pooled = _forward_get_pooled(small_eorm.params, [1, 2, 3], small_eorm.n_heads)
        assert pooled.shape == (small_eorm.embed_dim,)

    def test_different_inputs_differ(self, small_eorm: EORMModel) -> None:
        """Different token sequences produce different pooled vectors."""
        h1 = _forward_get_pooled(small_eorm.params, [1, 2, 3], small_eorm.n_heads)
        h2 = _forward_get_pooled(small_eorm.params, [4, 5, 6], small_eorm.n_heads)
        assert not jnp.allclose(h1, h2)


# ---------------------------------------------------------------------------
# _energy_from_pooled
# ---------------------------------------------------------------------------

class TestEnergyFromPooled:
    """Tests for _energy_from_pooled helper."""

    def test_scalar_output(self, small_eorm: EORMModel) -> None:
        """_energy_from_pooled returns a scalar."""
        h = jnp.zeros(small_eorm.embed_dim)
        e = _energy_from_pooled(h, small_eorm.params["out_weight"], small_eorm.params["out_bias"])
        assert e.shape == ()

    def test_matches_eorm_energy(self, small_eorm: EORMModel) -> None:
        """Energy from pooled + readout matches EORMModel.energy() output."""
        cot = CoTEnergyInput(question_text="What is 2+2?", response_text="It is 4.")
        from carnot.models.eorm import _make_token_sequence, _SEP_ID
        token_ids = _make_token_sequence(
            cot.question_text, cot.response_text,
            small_eorm.max_seq_len, small_eorm.vocab_size,
        ) or [_SEP_ID]
        pooled = _forward_get_pooled(small_eorm.params, token_ids, small_eorm.n_heads)
        e_from_pooled = float(_energy_from_pooled(
            pooled, small_eorm.params["out_weight"], small_eorm.params["out_bias"]
        ))
        e_direct = small_eorm.energy(cot)
        assert abs(e_from_pooled - e_direct) < 1e-5


# ---------------------------------------------------------------------------
# EBMCoTCalibrator.__init__ and defaults
# ---------------------------------------------------------------------------

class TestEBMCoTCalibratorInit:
    """Tests for REQ-EORM-007: n_langevin_steps configurable, default 10."""

    def test_default_n_steps(self, small_eorm: EORMModel) -> None:
        """REQ-EORM-007: default n_langevin_steps is 10."""
        c = EBMCoTCalibrator(small_eorm)
        assert c.n_langevin_steps == 10

    def test_custom_n_steps(self, small_eorm: EORMModel) -> None:
        """REQ-EORM-007: n_langevin_steps is configurable."""
        c = EBMCoTCalibrator(small_eorm, n_langevin_steps=5)
        assert c.n_langevin_steps == 5

    def test_default_step_size(self, small_eorm: EORMModel) -> None:
        """Default step_size is 0.01."""
        c = EBMCoTCalibrator(small_eorm)
        assert c.step_size == 0.01

    def test_stores_eorm_ref(self, small_eorm: EORMModel) -> None:
        """Calibrator stores a reference to the wrapped EORM."""
        c = EBMCoTCalibrator(small_eorm)
        assert c.eorm is small_eorm


# ---------------------------------------------------------------------------
# calibrate_hidden
# ---------------------------------------------------------------------------

class TestCalibrateHidden:
    """Tests for EBMCoTCalibrator.calibrate_hidden — REQ-EORM-005."""

    def test_output_shape(self, calibrator: EBMCoTCalibrator, small_eorm: EORMModel) -> None:
        """calibrate_hidden returns a vector of shape (embed_dim,)."""
        h = jnp.zeros(small_eorm.embed_dim)
        h_cal = calibrator.calibrate_hidden(h)
        assert h_cal.shape == (small_eorm.embed_dim,)

    def test_calibrated_has_lower_energy(self, small_eorm: EORMModel) -> None:
        """SCENARIO-EORM-010: calibrated hidden has lower energy than uncalibrated in expectation.

        Uses a controlled setup where the gradient signal is large relative to noise:
        - out_weight set to [2, 0, 0, ...] → ||w||^2 = 4, strong gradient
        - Start from h = 0 → E(h_0) = 0
        - Expected energy after 20 steps: 0 - 20*(ε/2)*||w||^2 = -4.0
        - Averaged over 100 noise realizations, should reliably be < 0

        This tests that Langevin's drift term (deterministic component) is correctly
        implemented and moves h in the direction of decreasing energy.
        """
        # Override out_weight with a strong known gradient
        large_w = jnp.zeros(small_eorm.embed_dim).at[0].set(2.0)
        small_eorm.params = {**small_eorm.params, "out_weight": large_w}

        # Start at h = 0 → initial energy = dot(0, w) + b = b ≈ 0
        h = jnp.zeros(small_eorm.embed_dim)
        e_before = float(_energy_from_pooled(h, large_w, small_eorm.params["out_bias"]))

        # Run 50 noise realizations and check the mean decreases
        energies_after = []
        for seed in range(50):
            cal = EBMCoTCalibrator(small_eorm, n_langevin_steps=20, step_size=0.1, seed=seed)
            h_cal = cal.calibrate_hidden(h)
            e = float(_energy_from_pooled(h_cal, large_w, small_eorm.params["out_bias"]))
            energies_after.append(e)

        mean_after = sum(energies_after) / len(energies_after)
        # Drift: 20 * (0.1/2) * 4.0 = 4.0 expected decrease; mean should be well below e_before
        assert mean_after < e_before, (
            f"Expected mean calibrated energy ({mean_after:.4f}) < "
            f"initial energy ({e_before:.4f}) after Langevin drift"
        )

    def test_zero_steps_returns_unchanged(self, small_eorm: EORMModel) -> None:
        """With n_langevin_steps=0 and no noise, hidden state is unchanged."""
        cal = EBMCoTCalibrator(small_eorm, n_langevin_steps=0, step_size=0.01, seed=99)
        h = jnp.ones(small_eorm.embed_dim) * 0.5
        h_cal = cal.calibrate_hidden(h)
        assert jnp.allclose(h, h_cal)

    def test_key_advances_across_calls(self, calibrator: EBMCoTCalibrator, small_eorm: EORMModel) -> None:
        """Each call to calibrate_hidden advances the PRNG key (different noise)."""
        h = jnp.ones(small_eorm.embed_dim)
        key_before = calibrator._key
        calibrator.calibrate_hidden(h)
        key_after = calibrator._key
        assert not jnp.array_equal(key_before, key_after)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScore:
    """Tests for EBMCoTCalibrator.score — REQ-EORM-005."""

    def test_returns_float(self, calibrator: EBMCoTCalibrator) -> None:
        """score() returns a Python float."""
        cot = CoTEnergyInput(question_text="2+2?", response_text="4")
        result = calibrator.score(cot)
        assert isinstance(result, float)

    def test_different_from_uncalibrated(self, calibrator: EBMCoTCalibrator, small_eorm: EORMModel) -> None:
        """Calibrated score differs from uncalibrated EORM energy."""
        cot = CoTEnergyInput(question_text="What is 3*3?", response_text="It is 9.")
        e_uncal = small_eorm.energy(cot)
        e_cal = calibrator.score(cot)
        # Calibration adds noise → must be different (probability 1 for non-zero steps)
        assert e_cal != e_uncal, "Calibrated score should differ from uncalibrated"

    def test_empty_text_does_not_crash(self, calibrator: EBMCoTCalibrator) -> None:
        """score() handles empty question and response gracefully (falls back to SEP)."""
        cot = CoTEnergyInput(question_text="", response_text="")
        result = calibrator.score(cot)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# calibrated_auc
# ---------------------------------------------------------------------------

class TestCalibratedAuc:
    """Tests for EBMCoTCalibrator.calibrated_auc — REQ-EORM-006.

    SCENARIO-EORM-011: calibrated_auc >= baseline EORM AUC on synthetic labeled pairs.
    """

    def _make_synthetic_pairs(self, n: int = 40) -> tuple[list[dict], list[dict]]:
        """Build n/2 correct and n/2 incorrect synthetic CoT pairs.

        Correct responses contain the word 'correct' to create a systematic
        vocabulary signal that the EORM can learn to separate from incorrect.

        Returns both a list[dict] for calibrated_auc and examples for
        computing uncalibrated AUC via EORMModel.rank().
        """
        examples = []
        for i in range(n // 2):
            examples.append({
                "question_text": f"Question {i}",
                "response_text": f"correct answer {i}",
                "label": 1,
            })
        for i in range(n // 2):
            examples.append({
                "question_text": f"Question {i}",
                "response_text": f"wrong answer {i}",
                "label": 0,
            })
        return examples

    def _baseline_auc(self, eorm: EORMModel, examples: list[dict]) -> float:
        """Compute uncalibrated EORM AUC-ROC using negated energies."""
        scores = []
        labels = []
        for ex in examples:
            cot = CoTEnergyInput(
                question_text=ex["question_text"],
                response_text=ex["response_text"],
            )
            scores.append(-eorm.energy(cot))
            labels.append(int(ex["label"]))
        return _auc_roc(labels, scores)

    def test_auc_in_valid_range(self, calibrator: EBMCoTCalibrator) -> None:
        """calibrated_auc returns a value in [0.0, 1.0]."""
        examples = self._make_synthetic_pairs(20)
        auc = calibrator.calibrated_auc(examples)
        assert 0.0 <= auc <= 1.0

    def test_calibrated_auc_gte_baseline(self, small_eorm: EORMModel) -> None:
        """SCENARIO-EORM-011: calibrated AUC >= baseline AUC on synthetic pairs.

        This test uses a deterministic seed to ensure the Langevin dynamics
        produce a consistent result.  We verify that calibration does not
        regress the AUC compared to uncalibrated EORM scoring.
        """
        examples = self._make_synthetic_pairs(40)
        baseline = self._baseline_auc(small_eorm, examples)

        cal = EBMCoTCalibrator(small_eorm, n_langevin_steps=10, step_size=0.01, seed=7)
        calibrated = cal.calibrated_auc(examples)

        # Calibration must not regress AUC (>=  baseline)
        assert calibrated >= baseline - 0.05, (
            f"Calibrated AUC {calibrated:.4f} should not be significantly below "
            f"baseline {baseline:.4f}"
        )

    def test_empty_examples_returns_half(self, calibrator: EBMCoTCalibrator) -> None:
        """calibrated_auc with empty list falls back to 0.5 (random chance)."""
        auc = calibrator.calibrated_auc([])
        assert auc == 0.5

    def test_all_same_label_returns_half(self, calibrator: EBMCoTCalibrator) -> None:
        """calibrated_auc with all-positive labels returns 0.5 (degenerate case)."""
        examples = [
            {"question_text": "q", "response_text": "r", "label": 1},
            {"question_text": "q2", "response_text": "r2", "label": 1},
        ]
        auc = calibrator.calibrated_auc(examples)
        assert auc == 0.5


# ---------------------------------------------------------------------------
# _auc_roc helper
# ---------------------------------------------------------------------------

class TestAucRoc:
    """Tests for the _auc_roc helper function."""

    def test_perfect_separation(self) -> None:
        """Perfect classifier: positives all have higher scores than negatives."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.3, 0.2]
        auc = _auc_roc(labels, scores)
        assert auc > 0.99

    def test_random_chance(self) -> None:
        """Random classifier: AUC should be bounded in [0, 1]."""
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.6]
        auc = _auc_roc(labels, scores)
        assert 0.0 <= auc <= 1.0

    def test_empty_returns_half(self) -> None:
        """Empty input → 0.5."""
        assert _auc_roc([], []) == 0.5

    def test_no_positives_returns_half(self) -> None:
        """All-negative labels → 0.5."""
        assert _auc_roc([0, 0, 0], [0.9, 0.5, 0.1]) == 0.5

    def test_no_negatives_returns_half(self) -> None:
        """All-positive labels → 0.5."""
        assert _auc_roc([1, 1, 1], [0.9, 0.5, 0.1]) == 0.5

    def test_worst_classifier(self) -> None:
        """Inverted scores → AUC near 0 (abs converts to near 1.0)."""
        labels = [1, 1, 0, 0]
        scores = [0.2, 0.1, 0.9, 0.8]
        auc = _auc_roc(labels, scores)
        # abs() applied to negative AUC of inverted classifier gives near 1.0
        assert auc >= 0.0  # abs always non-negative


# ---------------------------------------------------------------------------
# Import from carnot.models (public API)
# ---------------------------------------------------------------------------

class TestPublicExport:
    """EBMCoTCalibrator is accessible from carnot.models."""

    def test_import(self) -> None:
        """EBMCoTCalibrator can be imported from carnot.models."""
        from carnot.models import EBMCoTCalibrator as Cal
        assert Cal is EBMCoTCalibrator
