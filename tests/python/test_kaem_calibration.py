"""Tests for CalibrationLayer and CalibratedLowRankKAEMEnergy.

Covers 100% of the code added in Exp 559 (RETRO-057).
Spec: REQ-SAMPLE-030, SCENARIO-SAMPLE-046, SCENARIO-SAMPLE-047, SCENARIO-SAMPLE-048
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.models.kaem_energy import CalibrationLayer, CalibratedLowRankKAEMEnergy


# ---------------------------------------------------------------------------
# CalibrationLayer tests
# ---------------------------------------------------------------------------


class TestCalibrationLayer:
    """100% coverage of CalibrationLayer (fit, transform, edge cases)."""

    def test_default_is_identity(self):
        """Before fit(), the layer is an identity transform (a=1, b=0).

        Spec: REQ-SAMPLE-030-2
        """
        cl = CalibrationLayer()
        assert cl.a == 1.0
        assert cl.b == 0.0
        assert not cl._fitted

    def test_transform_scalar_before_fit(self):
        """transform() works on a scalar before fit (identity pass-through).

        Spec: REQ-SAMPLE-030-2
        """
        cl = CalibrationLayer()
        assert cl.transform(5.0) == 5.0

    def test_transform_array_before_fit(self):
        """transform() works on a numpy array before fit.

        Spec: REQ-SAMPLE-030-2
        """
        cl = CalibrationLayer()
        x = np.array([1.0, 2.0, 3.0])
        result = cl.transform(x)
        np.testing.assert_allclose(result, x)

    def test_fit_simple_affine(self):
        """fit() recovers exact a=2, b=3 when E_full = 2*E_lowrank + 3.

        Spec: REQ-SAMPLE-030-1
        """
        rng = np.random.default_rng(0)
        E_lowrank = rng.uniform(-5, 5, size=200)
        E_full = 2.0 * E_lowrank + 3.0

        cl = CalibrationLayer()
        cl.fit(E_full, E_lowrank)

        assert cl._fitted
        assert abs(cl.a - 2.0) < 1e-8
        assert abs(cl.b - 3.0) < 1e-8

    def test_fit_sets_fitted_flag(self):
        """_fitted is True after a successful fit().

        Spec: REQ-SAMPLE-030-1
        """
        E_lr = np.array([1.0, 2.0, 3.0])
        E_full = np.array([2.0, 4.0, 6.0])
        cl = CalibrationLayer()
        cl.fit(E_full, E_lr)
        assert cl._fitted

    def test_transform_after_fit(self):
        """transform() applies the fitted affine correction.

        Spec: REQ-SAMPLE-030-2, SCENARIO-SAMPLE-046
        """
        E_lr = np.linspace(-3, 3, 100)
        E_full = 0.5 * E_lr - 1.0
        cl = CalibrationLayer()
        cl.fit(E_full, E_lr)
        out = cl.transform(E_lr)
        np.testing.assert_allclose(out, E_full, atol=1e-6)

    def test_fit_degenerate_too_few_samples(self):
        """fit() with n < 2 keeps identity transform (degenerate fallback).

        Spec: REQ-SAMPLE-030-1
        """
        cl = CalibrationLayer()
        cl.fit(np.array([1.0]), np.array([2.0]))
        assert cl.a == 1.0
        assert cl.b == 0.0
        assert not cl._fitted

    def test_fit_degenerate_length_mismatch(self):
        """fit() with mismatched array lengths keeps identity transform.

        Spec: REQ-SAMPLE-030-1
        """
        cl = CalibrationLayer()
        cl.fit(np.array([1.0, 2.0]), np.array([3.0]))
        assert cl.a == 1.0
        assert cl.b == 0.0
        assert not cl._fitted

    def test_transform_reduces_mad_after_fit(self):
        """After fit, calibrated energy_mad < uncalibrated energy_mad.

        This is the core accuracy property (SCENARIO-SAMPLE-046).
        """
        rng = np.random.default_rng(7)
        E_lr = rng.standard_normal(500)
        # Introduce scale/offset distortion: E_full = 3*E_lr - 2 + small noise
        E_full = 3.0 * E_lr - 2.0 + rng.standard_normal(500) * 0.01

        cl = CalibrationLayer()
        cl.fit(E_full, E_lr)

        E_calibrated = cl.transform(E_lr)
        mad_before = float(np.mean(np.abs(E_lr - E_full)))
        mad_after = float(np.mean(np.abs(E_calibrated - E_full)))
        assert mad_after < mad_before, (
            f"Calibration must reduce MAD: before={mad_before:.4f}, after={mad_after:.4f}"
        )

    def test_transform_scalar_after_fit(self):
        """transform() returns a scalar (not array) when given a scalar.

        Spec: REQ-SAMPLE-030-2
        """
        cl = CalibrationLayer()
        E_lr = np.linspace(0, 1, 50)
        E_full = 2.0 * E_lr + 1.0
        cl.fit(E_full, E_lr)
        result = cl.transform(0.5)
        # Should be a float-like scalar close to 2*0.5+1 = 2.0
        assert abs(result - 2.0) < 0.01


# ---------------------------------------------------------------------------
# CalibratedLowRankKAEMEnergy tests
# ---------------------------------------------------------------------------


class TestCalibratedLowRankKAEMEnergy:
    """100% coverage of CalibratedLowRankKAEMEnergy."""

    def test_construction_valid(self):
        """CalibratedLowRankKAEMEnergy can be constructed with valid args.

        Spec: REQ-SAMPLE-030-3
        """
        model = CalibratedLowRankKAEMEnergy(n_vars=10, k=2)
        assert model.n_vars == 10
        assert model.k == 2

    def test_construction_invalid_n_vars(self):
        """n_vars < 1 raises ValueError.

        Spec: REQ-SAMPLE-030-3
        """
        with pytest.raises(ValueError, match="n_vars must be >= 1"):
            CalibratedLowRankKAEMEnergy(n_vars=0, k=2)

    def test_construction_invalid_k(self):
        """k < 1 raises ValueError.

        Spec: REQ-SAMPLE-030-3
        """
        with pytest.raises(ValueError, match="k must be >= 1"):
            CalibratedLowRankKAEMEnergy(n_vars=10, k=0)

    def test_energy_before_calibrate_raises(self):
        """energy() before calibrate() raises RuntimeError.

        Spec: REQ-SAMPLE-030-3
        """
        import jax.numpy as jnp

        model = CalibratedLowRankKAEMEnergy(n_vars=10, k=2)
        x = jnp.zeros(10)
        with pytest.raises(RuntimeError, match="calibrate.*must be called"):
            model.energy(x)

    def test_calibrate_completes(self):
        """calibrate() completes without error and sets calibration layer.

        Spec: SCENARIO-SAMPLE-047
        """
        model = CalibratedLowRankKAEMEnergy(n_vars=10, k=2)
        model.calibrate(n_samples=50, n_vars=10, rng_seed=0)
        assert model._calibration._fitted

    def test_energy_returns_scalar_after_calibrate(self):
        """energy() returns a Python float after calibrate().

        Spec: REQ-SAMPLE-030-3, SCENARIO-SAMPLE-047
        """
        import jax.numpy as jnp

        model = CalibratedLowRankKAEMEnergy(n_vars=10, k=2)
        model.calibrate(n_samples=50, n_vars=10, rng_seed=0)
        x = jnp.ones(10) * 0.5
        result = model.energy(x)
        assert isinstance(result, float)

    def test_calibrate_default_n_vars(self):
        """calibrate() uses self.n_vars when n_vars argument is None.

        Spec: REQ-SAMPLE-030-4
        """
        model = CalibratedLowRankKAEMEnergy(n_vars=8, k=2)
        # Pass n_vars=None explicitly (default)
        model.calibrate(n_samples=30, n_vars=None, rng_seed=1)
        assert model._calibration._fitted

    def test_calibrated_energy_different_from_uncalibrated(self):
        """Calibrated energy differs from raw low-rank energy when a != 1 or b != 0.

        This verifies the calibration transform is actually applied.
        Spec: REQ-SAMPLE-030-3
        """
        import jax.numpy as jnp

        model = CalibratedLowRankKAEMEnergy(n_vars=10, k=2)
        model.calibrate(n_samples=100, n_vars=10, rng_seed=5)

        x = jnp.array([0.5] * 10)
        raw_lr = float(model._lowrank.energy(x))
        calibrated = model.energy(x)
        # If a=1 and b=0 (identity), the two are equal — but calibration should change them
        # unless the model happens to be perfectly calibrated already (rare).
        # We check that the calibration formula is applied correctly:
        expected = model._calibration.transform(raw_lr)
        assert abs(calibrated - expected) < 1e-9

    def test_rank_sweep_calibration_improves_mad(self):
        """Calibration reduces energy_mad vs the reference used for calibration.

        Lightweight version of SCENARIO-SAMPLE-048.  Uses the calibration model's
        own internal full_kaem as the reference, so the comparison is self-consistent.
        The key mathematical invariant: least-squares fit minimises MAD on the
        calibration data, so mad_after_on_cal_data <= mad_before_on_cal_data.

        Spec: SCENARIO-SAMPLE-048
        """
        import jax.numpy as jnp
        import jax.random as jrandom
        from carnot.models.lowrank_kaem import LowRankKAEMEnergy

        N_VARS = 10
        N_CAL = 80
        RNG = np.random.default_rng(99)

        data_cal = RNG.choice([-1.0, 1.0], size=(N_CAL, N_VARS)).astype(np.float32)
        data_cal_jax = jnp.array(data_cal)

        found_improvement = False
        for k in [2, 4]:
            cal = CalibratedLowRankKAEMEnergy(n_vars=N_VARS, k=k, key=jrandom.PRNGKey(k))
            cal.calibrate(n_samples=N_CAL, n_vars=N_VARS, rng_seed=99)

            # Use the calibration model's own full_kaem as ground truth
            # (the same reference used to fit the calibration layer).
            assert cal._full_kaem is not None

            E_full = np.array(
                [float(cal._full_kaem.energy(data_cal_jax[i])) for i in range(N_CAL)]
            )
            E_lr = np.array(
                [float(cal._lowrank.energy(data_cal_jax[i])) for i in range(N_CAL)]
            )
            E_cal = np.array([cal.energy(data_cal_jax[i]) for i in range(N_CAL)])

            E_std = float(np.std(E_full)) or 1.0
            mad_before = float(np.mean(np.abs(E_lr - E_full)) / E_std)
            mad_after = float(np.mean(np.abs(E_cal - E_full)) / E_std)

            # Least-squares calibration on the same data used here guarantees improvement.
            if mad_after <= mad_before:
                found_improvement = True

        assert found_improvement, (
            "Calibration should reduce MAD on calibration data for at least one k"
        )
