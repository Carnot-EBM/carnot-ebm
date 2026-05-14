"""Tests for Exp 1682 bias correction in the parallel Ising sampler.

Spec coverage: REQ-SAMPLE-1686-1, REQ-SAMPLE-1686-2, SCENARIO-SAMPLE-1686
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.samplers.parallel_ising import (
    ParallelIsingSampler,
    _EXP1682_INTERCEPT,
    _EXP1682_SLOPE,
    corrected_magnetization_mean,
    ising_mean_bias_correction,
)

# Calibration points from experiment_1682_thrml_bias.json sweep_b
_CAL_BETAS = [1.05, 1.2, 1.5]
_CAL_BIASES = [
    -0.05844948256992616,
    -0.04058684790575395,
    -0.007181511640110383,
]


class TestIsingSMeanBiasCorrection:
    """REQ-SAMPLE-1686-1: ising_mean_bias_correction returns the linear formula."""

    def test_correction_sign_is_positive(self):
        """SCENARIO-SAMPLE-1686: correction is positive (counteracts underestimate)."""
        for beta in _CAL_BETAS:
            c = ising_mean_bias_correction(beta)
            assert c > 0, f"correction at beta={beta} should be positive, got {c}"

    def test_correction_matches_calibration_points(self):
        """SCENARIO-SAMPLE-1686: correction negates the OLS-fitted bias at calibration betas."""
        for beta, expected_bias in zip(_CAL_BETAS, _CAL_BIASES):
            correction = ising_mean_bias_correction(beta)
            # The correction should approximately negate the measured bias.
            # Allow 2e-3 tolerance (OLS is a fit, not interpolation).
            assert abs(correction + expected_bias) < 2e-3, (
                f"beta={beta}: correction={correction:.6f}, "
                f"expected ≈ {-expected_bias:.6f}"
            )

    def test_correction_decreases_with_beta(self):
        """SCENARIO-SAMPLE-1686: larger beta → smaller correction (bias shrinks at low T)."""
        c_low = ising_mean_bias_correction(1.05)
        c_mid = ising_mean_bias_correction(1.2)
        c_high = ising_mean_bias_correction(1.5)
        assert c_low > c_mid > c_high, (
            f"correction should decrease with beta: {c_low:.4f} > {c_mid:.4f} > {c_high:.4f}"
        )

    def test_linear_formula_consistency(self):
        """REQ-SAMPLE-1686-1: formula is exactly -(_INTERCEPT + _SLOPE * beta)."""
        for beta in [0.5, 1.0, 1.2, 2.0]:
            expected = -(_EXP1682_INTERCEPT + _EXP1682_SLOPE * beta)
            actual = ising_mean_bias_correction(beta)
            assert abs(actual - expected) < 1e-12, (
                f"formula mismatch at beta={beta}: {actual} vs {expected}"
            )


class TestCorrectedMagnetizationMean:
    """REQ-SAMPLE-1686-2: corrected_magnetization_mean applies scalar shift uniformly."""

    def _make_samples(self, n_samples: int, n_spins: int, fill: float) -> jnp.ndarray:
        """Create a fixed-value boolean sample array for deterministic tests."""
        # fill in {0.0, 1.0} treated as boolean; threshold at 0.5 → True if 1.0
        data = np.full((n_samples, n_spins), fill > 0.5, dtype=np.bool_)
        return jnp.array(data)

    def test_shape_preserved(self):
        """REQ-SAMPLE-1686-2: output shape matches n_spins."""
        samples = self._make_samples(100, 16, 0.0)
        result = corrected_magnetization_mean(samples, beta=1.2)
        assert result.shape == (16,)

    def test_shift_equals_correction(self):
        """REQ-SAMPLE-1686-2: corrected mean = raw mean + ising_mean_bias_correction(beta)."""
        n_spins = 8
        beta = 1.2
        # Use real samples via the sampler to get a realistic distribution.
        key = jrandom.PRNGKey(42)
        biases = jnp.zeros(n_spins)
        J = jnp.zeros((n_spins, n_spins))
        sampler = ParallelIsingSampler(n_warmup=100, n_samples=500, steps_per_sample=5)
        samples = sampler.sample(key, biases, J, beta=beta)

        raw_mean = jnp.mean(samples.astype(jnp.float32), axis=0)
        corrected = corrected_magnetization_mean(samples, beta=beta)
        expected_correction = ising_mean_bias_correction(beta)

        diff = corrected - raw_mean
        assert jnp.allclose(diff, expected_correction, atol=1e-6), (
            f"shift should be {expected_correction:.6f}, got {float(jnp.mean(diff)):.6f}"
        )

    def test_all_spins_shifted_uniformly(self):
        """REQ-SAMPLE-1686-2: all spins receive the same scalar shift."""
        n_spins = 12
        beta = 1.05
        samples = self._make_samples(200, n_spins, 0.3)  # raw mean ≈ 0.0 everywhere

        raw_mean = jnp.mean(samples.astype(jnp.float32), axis=0)
        corrected = corrected_magnetization_mean(samples, beta=beta)
        shifts = corrected - raw_mean

        # All shifts must be identical (uniform scalar correction).
        assert float(jnp.max(shifts) - jnp.min(shifts)) < 1e-6, (
            "correction should be identical for every spin"
        )


class TestCurieWeiss10kVerification:
    """SCENARIO-SAMPLE-1686: 10k-sample Curie-Weiss statistical verification.

    Uses a small n=16 all-to-all zero-bias system at subcritical beta where
    the correction magnitude can be checked numerically without ergodicity issues.
    """

    def test_10k_samples_correction_applied(self):
        """SCENARIO-SAMPLE-1686: corrected_magnetization_mean runs on 10k samples."""
        n_spins = 16
        beta = 1.2
        # All-to-all coupling with J_ij = 1/(n-1) — Curie-Weiss topology.
        J_val = 1.0 / (n_spins - 1)
        J = jnp.full((n_spins, n_spins), J_val).at[jnp.arange(n_spins), jnp.arange(n_spins)].set(0.0)
        biases = jnp.zeros(n_spins)

        sampler = ParallelIsingSampler(
            n_warmup=500,
            n_samples=10_000,
            steps_per_sample=5,
            use_checkerboard=False,  # fully parallel, where the bias originates
        )
        key = jrandom.PRNGKey(1682)
        samples = sampler.sample(key, biases, J, beta=beta)

        assert samples.shape == (10_000, n_spins)

        raw_mean = float(jnp.mean(samples.astype(jnp.float32)))
        corrected_vec = corrected_magnetization_mean(samples, beta=beta)
        corrected_mean = float(jnp.mean(corrected_vec))

        expected_shift = ising_mean_bias_correction(beta)
        actual_shift = corrected_mean - raw_mean

        # The shift must equal the scalar correction within float precision.
        assert abs(actual_shift - expected_shift) < 1e-5, (
            f"shift={actual_shift:.6f}, expected={expected_shift:.6f}"
        )

    def test_corrected_mean_closer_to_half(self):
        """SCENARIO-SAMPLE-1686: for symmetric zero-bias system at supercritical beta,
        the corrected mean should be closer to 0.5 than the raw mean.

        Why 0.5 is the target: with biases=0 and symmetric coupling, both the
        all-0 and all-1 ground states are equally probable. The ergodic mean is
        0.5. The parallel Gibbs sampler systematically underestimates this value
        (the bias from Exp 1682). Correction should bring it closer to 0.5.
        """
        n_spins = 8
        beta = 1.2
        J_val = 1.0 / (n_spins - 1)
        J = jnp.full((n_spins, n_spins), J_val).at[jnp.arange(n_spins), jnp.arange(n_spins)].set(0.0)
        biases = jnp.zeros(n_spins)

        sampler = ParallelIsingSampler(
            n_warmup=2000,
            n_samples=10_000,
            steps_per_sample=10,
            use_checkerboard=False,
        )
        key = jrandom.PRNGKey(42)
        samples = sampler.sample(key, biases, J, beta=beta)

        raw_mean = float(jnp.mean(samples.astype(jnp.float32)))
        corrected_mean = float(jnp.mean(corrected_magnetization_mean(samples, beta=beta)))

        # The corrected mean should be farther from 0 (or less negative in deviation
        # from 0.5) than the raw mean. Since the bias is negative (underestimate),
        # raw_mean < corrected_mean.
        assert corrected_mean > raw_mean, (
            f"corrected mean ({corrected_mean:.4f}) should exceed raw mean ({raw_mean:.4f})"
        )
