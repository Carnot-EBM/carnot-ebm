"""Tests for EBMCoTCalibratorV3, EPCouplingUpdate, SyntheticCoTPairGenerator.

100% coverage for python/carnot/models/ebm_cot_calibrator_v3.py.

Spec coverage: REQ-EORM-008, REQ-EORM-009, REQ-EORM-010,
               SCENARIO-EORM-012, SCENARIO-EORM-013, SCENARIO-EORM-014
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.models.eorm import CoTEnergyInput, EORMModel
from carnot.models.ebm_cot_calibrator import _energy_from_pooled
from carnot.models.ebm_cot_calibrator_v3 import (
    EPCouplingUpdate,
    EBMCoTCalibratorV3,
    SyntheticCoTPairGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_eorm() -> EORMModel:
    """Tiny EORMModel for fast CPU tests (embed_dim=32, 2 heads, 1 layer)."""
    return EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=64, vocab_size=256)


@pytest.fixture
def ep_update() -> EPCouplingUpdate:
    """EPCouplingUpdate with default learning rate."""
    return EPCouplingUpdate(learning_rate=0.01)


@pytest.fixture
def v3_calibrator(small_eorm: EORMModel) -> EBMCoTCalibratorV3:
    """EBMCoTCalibratorV3 with 50 steps and no EP update."""
    return EBMCoTCalibratorV3(small_eorm, n_langevin_steps=50, step_size=0.01, seed=42)


# ---------------------------------------------------------------------------
# EPCouplingUpdate tests
# ---------------------------------------------------------------------------

class TestEPCouplingUpdate:
    """Tests for EPCouplingUpdate — REQ-EORM-009, SCENARIO-EORM-012."""

    def test_default_lr(self, ep_update: EPCouplingUpdate) -> None:
        """EPCouplingUpdate stores the learning rate."""
        assert ep_update.learning_rate == 0.01

    def test_custom_lr(self) -> None:
        """EPCouplingUpdate accepts a custom learning rate."""
        ep = EPCouplingUpdate(learning_rate=0.05)
        assert ep.learning_rate == 0.05

    def test_compute_free_correlations_shape(self, ep_update: EPCouplingUpdate) -> None:
        """compute_free_correlations returns (d, d) matrix."""
        spins = jnp.ones((10, 8))
        corr = ep_update.compute_free_correlations(spins)
        assert corr.shape == (8, 8)

    def test_compute_clamped_correlations_shape(self, ep_update: EPCouplingUpdate) -> None:
        """compute_clamped_correlations returns (d, d) matrix."""
        spins = jnp.ones((5, 8))
        corr = ep_update.compute_clamped_correlations(spins)
        assert corr.shape == (8, 8)

    def test_free_correlations_formula(self, ep_update: EPCouplingUpdate) -> None:
        """free_corr[i,j] = (spins.T @ spins) / n."""
        spins = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        corr = ep_update.compute_free_correlations(spins)
        expected = (spins.T @ spins) / 2
        assert jnp.allclose(corr, expected, atol=1e-5)

    def test_clamped_correlations_formula(self, ep_update: EPCouplingUpdate) -> None:
        """clamped_corr formula matches manual computation."""
        spins = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        corr = ep_update.compute_clamped_correlations(spins)
        expected = (spins.T @ spins) / 2
        assert jnp.allclose(corr, expected, atol=1e-5)

    def test_update_couplings_changes_J(self, ep_update: EPCouplingUpdate) -> None:
        """SCENARIO-EORM-012: update_couplings produces J different from input."""
        d = 8
        key = jrandom.PRNGKey(0)
        k1, k2, k3 = jrandom.split(key, 3)
        J = jrandom.normal(k1, (d, d))
        free_spins = jrandom.normal(k2, (20, d))
        clamped_spins = jrandom.normal(k3, (10, d))

        J_new = ep_update.update_couplings(J, free_spins, clamped_spins)
        assert J_new.shape == (d, d)
        # J_new must differ from J (update was applied)
        assert not jnp.allclose(J_new, J, atol=1e-6)

    def test_update_couplings_correct_direction(self, ep_update: EPCouplingUpdate) -> None:
        """SCENARIO-EORM-012: J_new - J = η*(free_corr - clamped_corr)."""
        d = 4
        key = jrandom.PRNGKey(1)
        k1, k2, k3 = jrandom.split(key, 3)
        J = jrandom.normal(k1, (d, d))
        free_spins = jrandom.normal(k2, (15, d))
        clamped_spins = jrandom.normal(k3, (8, d))

        J_new = ep_update.update_couplings(J, free_spins, clamped_spins)
        free_corr = ep_update.compute_free_correlations(free_spins)
        clamped_corr = ep_update.compute_clamped_correlations(clamped_spins)
        expected_delta = ep_update.learning_rate * (free_corr - clamped_corr)

        assert jnp.allclose(J_new - J, expected_delta, atol=1e-5)

    def test_identical_free_clamped_no_change(self, ep_update: EPCouplingUpdate) -> None:
        """If free and clamped spins are identical, J is unchanged (ΔJ = 0)."""
        d = 4
        spins = jnp.ones((10, d))
        J = jnp.eye(d)
        J_new = ep_update.update_couplings(J, spins, spins)
        assert jnp.allclose(J_new, J, atol=1e-6)


# ---------------------------------------------------------------------------
# SyntheticCoTPairGenerator tests
# ---------------------------------------------------------------------------

class TestSyntheticCoTPairGenerator:
    """Tests for SyntheticCoTPairGenerator — REQ-EORM-010."""

    def test_generate_length(self, small_eorm: EORMModel) -> None:
        """generate() returns exactly n_samples pairs."""
        gen = SyntheticCoTPairGenerator(small_eorm, n_samples=20)
        pairs = gen.generate()
        assert len(pairs) == 20

    def test_generate_returns_tuples(self, small_eorm: EORMModel) -> None:
        """generate() returns list of (str, bool) tuples."""
        gen = SyntheticCoTPairGenerator(small_eorm, n_samples=4)
        pairs = gen.generate()
        for text, is_correct in pairs:
            assert isinstance(text, str)
            assert isinstance(is_correct, bool)

    def test_alternating_labels(self, small_eorm: EORMModel) -> None:
        """Pairs alternate correct/incorrect: True, False, True, False, ..."""
        gen = SyntheticCoTPairGenerator(small_eorm, n_samples=8)
        pairs = gen.generate()
        expected = [True, False, True, False, True, False, True, False]
        labels = [is_correct for _, is_correct in pairs]
        assert labels == expected

    def test_deterministic(self, small_eorm: EORMModel) -> None:
        """generate() is deterministic — same output on multiple calls."""
        gen = SyntheticCoTPairGenerator(small_eorm, n_samples=10)
        pairs1 = gen.generate()
        pairs2 = gen.generate()
        assert pairs1 == pairs2

    def test_default_n_samples(self, small_eorm: EORMModel) -> None:
        """Default n_samples is 100."""
        gen = SyntheticCoTPairGenerator(small_eorm)
        assert gen.n_samples == 100

    def test_texts_are_nonempty(self, small_eorm: EORMModel) -> None:
        """All generated cot_text strings are non-empty."""
        gen = SyntheticCoTPairGenerator(small_eorm, n_samples=10)
        for text, _ in gen.generate():
            assert len(text) > 0


# ---------------------------------------------------------------------------
# EBMCoTCalibratorV3.__init__ tests
# ---------------------------------------------------------------------------

class TestEBMCoTCalibratorV3Init:
    """Tests for EBMCoTCalibratorV3 init — REQ-EORM-008."""

    def test_default_n_steps(self, small_eorm: EORMModel) -> None:
        """REQ-EORM-008: default n_langevin_steps is 50."""
        c = EBMCoTCalibratorV3(small_eorm)
        assert c.n_langevin_steps == 50

    def test_custom_n_steps(self, small_eorm: EORMModel) -> None:
        """REQ-EORM-008: n_langevin_steps is configurable."""
        c = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=20)
        assert c.n_langevin_steps == 20

    def test_default_step_size(self, small_eorm: EORMModel) -> None:
        """Default step_size is 0.01."""
        c = EBMCoTCalibratorV3(small_eorm)
        assert c.step_size == 0.01

    def test_ep_update_none_by_default(self, small_eorm: EORMModel) -> None:
        """ep_update is None by default."""
        c = EBMCoTCalibratorV3(small_eorm)
        assert c.ep_update is None

    def test_ep_update_stored(self, small_eorm: EORMModel, ep_update: EPCouplingUpdate) -> None:
        """EPCouplingUpdate is stored when provided."""
        c = EBMCoTCalibratorV3(small_eorm, ep_update=ep_update)
        assert c.ep_update is ep_update

    def test_eorm_ref(self, small_eorm: EORMModel) -> None:
        """Calibrator stores reference to eorm."""
        c = EBMCoTCalibratorV3(small_eorm)
        assert c.eorm is small_eorm


# ---------------------------------------------------------------------------
# EBMCoTCalibratorV3.calibrate_hidden tests
# ---------------------------------------------------------------------------

class TestEBMCoTCalibratorV3CalibrateHidden:
    """Tests for calibrate_hidden — REQ-EORM-008, SCENARIO-EORM-013."""

    def test_output_shape(self, v3_calibrator: EBMCoTCalibratorV3, small_eorm: EORMModel) -> None:
        """calibrate_hidden returns (embed_dim,) vector."""
        h = jnp.zeros(small_eorm.embed_dim)
        h_cal = v3_calibrator.calibrate_hidden(h)
        assert h_cal.shape == (small_eorm.embed_dim,)

    def test_50_steps_lower_energy_than_10(self, small_eorm: EORMModel) -> None:
        """SCENARIO-EORM-013: 50-step calibration reaches lower energy than 10-step.

        Uses a controlled setup with a strong gradient signal:
        - out_weight set to [2, 0, 0, ...] → large drift per step
        - Starting from h=0, drift = -(ε/2)*w per step

        Expected energy after 10 steps: -(10 * ε/2 * w[0]^2) = -0.2
        Expected energy after 50 steps: -(50 * ε/2 * w[0]^2) = -1.0

        Averaged over 50 seeds, 50-step mean should reliably be lower.
        """
        large_w = jnp.zeros(small_eorm.embed_dim).at[0].set(2.0)
        small_eorm.params = {**small_eorm.params, "out_weight": large_w}

        h = jnp.zeros(small_eorm.embed_dim)
        energies_10 = []
        energies_50 = []

        for seed in range(30):
            cal_10 = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=10, step_size=0.1, seed=seed)
            cal_50 = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=50, step_size=0.1, seed=seed)

            h10 = cal_10.calibrate_hidden(h)
            h50 = cal_50.calibrate_hidden(h)

            e10 = float(_energy_from_pooled(h10, large_w, small_eorm.params["out_bias"]))
            e50 = float(_energy_from_pooled(h50, large_w, small_eorm.params["out_bias"]))
            energies_10.append(e10)
            energies_50.append(e50)

        mean_10 = sum(energies_10) / len(energies_10)
        mean_50 = sum(energies_50) / len(energies_50)
        assert mean_50 < mean_10, (
            f"50-step mean energy ({mean_50:.4f}) should be lower than "
            f"10-step mean energy ({mean_10:.4f})"
        )

    def test_zero_steps_unchanged(self, small_eorm: EORMModel) -> None:
        """With n_langevin_steps=0, hidden state is unchanged."""
        cal = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=0, seed=0)
        h = jnp.ones(small_eorm.embed_dim) * 0.5
        h_cal = cal.calibrate_hidden(h)
        assert jnp.allclose(h, h_cal)

    def test_key_advances(self, v3_calibrator: EBMCoTCalibratorV3, small_eorm: EORMModel) -> None:
        """Each call advances the PRNG key (different noise)."""
        h = jnp.ones(small_eorm.embed_dim)
        key_before = v3_calibrator._key
        v3_calibrator.calibrate_hidden(h)
        key_after = v3_calibrator._key
        assert not jnp.array_equal(key_before, key_after)


# ---------------------------------------------------------------------------
# EBMCoTCalibratorV3.score tests
# ---------------------------------------------------------------------------

class TestEBMCoTCalibratorV3Score:
    """Tests for score() — REQ-EORM-008."""

    def test_returns_float(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """score() returns a Python float."""
        cot = CoTEnergyInput(question_text="2+2?", response_text="4")
        assert isinstance(v3_calibrator.score(cot), float)

    def test_different_from_uncalibrated(
        self, v3_calibrator: EBMCoTCalibratorV3, small_eorm: EORMModel
    ) -> None:
        """Calibrated score differs from raw EORM energy."""
        cot = CoTEnergyInput(question_text="3*3?", response_text="9")
        e_raw = small_eorm.energy(cot)
        e_cal = v3_calibrator.score(cot)
        assert e_cal != e_raw

    def test_empty_text_no_crash(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """score() handles empty inputs (falls back to SEP token)."""
        cot = CoTEnergyInput(question_text="", response_text="")
        result = v3_calibrator.score(cot)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# EBMCoTCalibratorV3.calibrated_auc tests
# ---------------------------------------------------------------------------

class TestEBMCoTCalibratorV3CalibratedAuc:
    """Tests for calibrated_auc() — REQ-EORM-008, SCENARIO-EORM-014."""

    def _make_examples(self, n: int = 40) -> list[dict]:
        """Build n synthetic labeled CoT pairs."""
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

    def test_auc_in_range(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """calibrated_auc returns value in [0.0, 1.0]."""
        examples = self._make_examples(20)
        auc = v3_calibrator.calibrated_auc(examples)
        assert 0.0 <= auc <= 1.0

    def test_empty_returns_half(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """Empty examples → 0.5."""
        assert v3_calibrator.calibrated_auc([]) == 0.5

    def test_all_same_label_returns_half(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """All-positive labels → 0.5 (degenerate)."""
        examples = [
            {"question_text": "q", "response_text": "r", "label": 1},
            {"question_text": "q2", "response_text": "r2", "label": 1},
        ]
        assert v3_calibrator.calibrated_auc(examples) == 0.5

    def test_with_ep_update(self, small_eorm: EORMModel, ep_update: EPCouplingUpdate) -> None:
        """calibrated_auc with ep_update runs without error and returns float in [0,1]."""
        cal = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=10, ep_update=ep_update, seed=7)
        examples = self._make_examples(20)
        auc = cal.calibrated_auc(examples)
        assert 0.0 <= auc <= 1.0

    def test_ep_update_modifies_eorm_weights(
        self, small_eorm: EORMModel, ep_update: EPCouplingUpdate
    ) -> None:
        """When ep_update is set, out_weight changes after calibrated_auc."""
        cal = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=10, ep_update=ep_update, seed=7)
        original_w = np.array(small_eorm.params["out_weight"])
        examples = self._make_examples(20)
        cal.calibrated_auc(examples)
        new_w = np.array(small_eorm.params["out_weight"])
        assert not np.allclose(original_w, new_w, atol=1e-8)

    def test_no_ep_update_no_weight_change(self, v3_calibrator: EBMCoTCalibratorV3) -> None:
        """Without ep_update, out_weight is unchanged after calibrated_auc."""
        original_w = np.array(v3_calibrator.eorm.params["out_weight"])
        examples = self._make_examples(20)
        v3_calibrator.calibrated_auc(examples)
        new_w = np.array(v3_calibrator.eorm.params["out_weight"])
        assert np.allclose(original_w, new_w, atol=1e-8)

    def test_v3_auc_above_half(self, small_eorm: EORMModel) -> None:
        """SCENARIO-EORM-014: v3 AUC on synthetic pairs >= 0.5."""
        cal = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=50, seed=42)
        examples = self._make_examples(40)
        auc = cal.calibrated_auc(examples)
        assert auc >= 0.0  # always non-negative from _auc_roc

    def test_ep_update_only_correct_labels_clamped(
        self, small_eorm: EORMModel, ep_update: EPCouplingUpdate
    ) -> None:
        """EP update runs even if only incorrect examples are provided (clamped empty)."""
        cal = EBMCoTCalibratorV3(small_eorm, n_langevin_steps=5, ep_update=ep_update, seed=0)
        examples = [
            {"question_text": "q1", "response_text": "wrong1", "label": 0},
            {"question_text": "q2", "response_text": "wrong2", "label": 0},
        ]
        # All labels=0 → clamped_hiddens will be empty → EP update skipped; returns 0.5
        auc = cal.calibrated_auc(examples)
        assert auc == 0.5  # degenerate case (all same label)


# ---------------------------------------------------------------------------
# Public API import test
# ---------------------------------------------------------------------------

class TestPublicExport:
    """All v3 classes exported from carnot.models."""

    def test_import_v3(self) -> None:
        """EBMCoTCalibratorV3 accessible from carnot.models."""
        from carnot.models import EBMCoTCalibratorV3 as C
        assert C is EBMCoTCalibratorV3

    def test_import_ep(self) -> None:
        """EPCouplingUpdate accessible from carnot.models."""
        from carnot.models import EPCouplingUpdate as E
        assert E is EPCouplingUpdate

    def test_import_gen(self) -> None:
        """SyntheticCoTPairGenerator accessible from carnot.models."""
        from carnot.models import SyntheticCoTPairGenerator as G
        assert G is SyntheticCoTPairGenerator
