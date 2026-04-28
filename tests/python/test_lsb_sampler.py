"""Unit tests for the Langevin Stochastic Boltzmann (LSB) sampler.

Spec coverage: REQ-SAMPLE-003, REQ-SAMPLE-LSB-001

These tests verify the LangevinSBSampler class and its functional interface.
Every test has at least one assertion (per CLAUDE.md: skipped tests are
invisible failures that erode suite confidence).
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.samplers.lsb_sampler import (
    LangevinSBSampler,
    is_lsb_enabled,
    lsb_sample,
)


# Force CPU-only JAX for reproducibility (per CLAUDE.md).
os.environ.setdefault("JAX_PLATFORMS", "cpu")


class TestLangevinSBSamplerProtocol:
    """REQ-SAMPLE-003: LangevinSBSampler satisfies the SamplerBackend protocol."""

    def test_backend_name(self):
        """SCENARIO-SAMPLE-LSB-001: backend_name is 'lsb'."""
        sampler = LangevinSBSampler()
        assert sampler.backend_name == "lsb"

    def test_implements_minimize_energy(self):
        """SCENARIO-SAMPLE-LSB-001: minimize_energy returns correct shape."""
        n = 8
        sampler = LangevinSBSampler(n_warmup=10, n_samples=5, steps_per_sample=2)
        biases = np.zeros(n, dtype=np.float32)
        couplings = np.zeros((n, n), dtype=np.float32)
        result = sampler.minimize_energy(biases, couplings, n_samples=5, n_steps=20, beta=5.0)
        assert result.shape == (5, n)

    def test_implements_sample(self):
        """SCENARIO-SAMPLE-LSB-001: sample() returns correct shape."""
        n = 6
        sampler = LangevinSBSampler(n_warmup=10, n_samples=4, steps_per_sample=2)
        biases = np.zeros(n, dtype=np.float32)
        couplings = np.zeros((n, n), dtype=np.float32)
        result = sampler.sample(biases, couplings, n_samples=4, config={"beta": 5.0})
        assert result.shape == (4, n)

    def test_protocol_structural_check(self):
        """SCENARIO-SAMPLE-LSB-001: LangevinSBSampler satisfies SamplerBackend protocol."""
        from carnot.samplers.backend import SamplerBackend

        sampler = LangevinSBSampler()
        assert isinstance(sampler, SamplerBackend)


class TestLangevinSBSamplerOutputProperties:
    """REQ-SAMPLE-LSB-001: Output spins are binary and shapes are correct."""

    def test_output_is_boolean(self):
        """SCENARIO-SAMPLE-LSB-002: Binarized output is boolean dtype."""
        n = 10
        sampler = LangevinSBSampler(n_warmup=5, n_samples=3, steps_per_sample=2, use_cem=False)
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        samples = sampler.run_sampler(jrandom.PRNGKey(0), b, J, beta=5.0)
        assert samples.dtype == jnp.bool_

    def test_output_shape_run_sampler(self):
        """SCENARIO-SAMPLE-LSB-002: run_sampler returns (n_samples, n_spins)."""
        n = 12
        n_samples = 7
        sampler = LangevinSBSampler(n_warmup=5, n_samples=n_samples, steps_per_sample=2)
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        samples = sampler.run_sampler(jrandom.PRNGKey(42), b, J, beta=5.0)
        assert samples.shape == (n_samples, n)

    def test_strong_positive_bias_prefers_one(self):
        """SCENARIO-SAMPLE-LSB-003: Strong positive bias → most spins = 1."""
        n = 10
        sampler = LangevinSBSampler(
            n_warmup=200, n_samples=20, steps_per_sample=5, use_cem=False, seed=0
        )
        b = jnp.ones(n) * 5.0  # Strong pull toward spin=1
        J = jnp.zeros((n, n))
        samples = sampler.run_sampler(jrandom.PRNGKey(0), b, J, beta=10.0)
        # Most samples should be mostly spin=1.
        mean_activation = float(jnp.mean(samples.astype(jnp.float32)))
        assert mean_activation > 0.7, f"Expected > 0.7, got {mean_activation}"

    def test_strong_negative_bias_prefers_zero(self):
        """SCENARIO-SAMPLE-LSB-003: Strong negative bias → most spins = 0."""
        n = 10
        sampler = LangevinSBSampler(
            n_warmup=200, n_samples=20, steps_per_sample=5, use_cem=False, seed=1
        )
        b = jnp.ones(n) * -5.0  # Strong pull toward spin=0
        J = jnp.zeros((n, n))
        samples = sampler.run_sampler(jrandom.PRNGKey(1), b, J, beta=10.0)
        mean_activation = float(jnp.mean(samples.astype(jnp.float32)))
        assert mean_activation < 0.3, f"Expected < 0.3, got {mean_activation}"

    def test_all_values_are_binary(self):
        """SCENARIO-SAMPLE-LSB-002: All spin values are exactly 0 or 1 (boolean)."""
        n = 15
        sampler = LangevinSBSampler(n_warmup=10, n_samples=8, steps_per_sample=3, use_cem=False)
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        samples = sampler.run_sampler(jrandom.PRNGKey(7), b, J, beta=5.0)
        # After binarization, every value must be True or False.
        unique_vals = jnp.unique(samples.astype(jnp.int32))
        for v in unique_vals:
            assert int(v) in (0, 1), f"Non-binary value in output: {v}"


class TestCEM:
    """REQ-SAMPLE-LSB-001: Conditional Expectation Matching beta estimation."""

    def test_cem_returns_float(self):
        """SCENARIO-SAMPLE-LSB-004: CEM returns a scalar float."""
        sampler = LangevinSBSampler(use_cem=True)
        n = 8
        spins = jnp.ones(n) * 0.6
        b = jnp.ones(n)
        J = jnp.zeros((n, n))
        result = sampler._cem_beta(spins, b, J, beta_init=5.0)
        assert isinstance(result, float)

    def test_cem_fallback_flat_landscape(self):
        """SCENARIO-SAMPLE-LSB-004: CEM falls back to beta_init on flat landscape."""
        sampler = LangevinSBSampler(use_cem=True)
        n = 6
        # All-zero biases and couplings → flat landscape (all h_i = 0).
        spins = jnp.zeros(n)
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        result = sampler._cem_beta(spins, b, J, beta_init=7.5)
        assert abs(result - 7.5) < 1e-5, f"CEM should fall back to 7.5, got {result}"

    def test_cem_beta_in_reasonable_range(self):
        """SCENARIO-SAMPLE-LSB-004: CEM beta is clamped to [0.1, 100.0]."""
        sampler = LangevinSBSampler(use_cem=True)
        n = 8
        # Huge bias → very large h → tiny CEM beta, should be clamped at 0.1
        spins = jnp.ones(n) * 0.9
        b = jnp.ones(n) * 1000.0
        J = jnp.zeros((n, n))
        result = sampler._cem_beta(spins, b, J, beta_init=5.0)
        assert 0.1 <= result <= 100.0, f"CEM beta out of expected range: {result}"

    def test_cem_enabled_vs_disabled(self):
        """SCENARIO-SAMPLE-LSB-004: CEM doesn't crash when enabled; output shape unchanged."""
        n = 8
        b = np.random.default_rng(42).normal(0, 1, n).astype(np.float32)
        J = np.zeros((n, n), dtype=np.float32)

        sampler_cem = LangevinSBSampler(
            n_warmup=20, n_samples=5, steps_per_sample=3, use_cem=True, seed=0
        )
        sampler_no_cem = LangevinSBSampler(
            n_warmup=20, n_samples=5, steps_per_sample=3, use_cem=False, seed=0
        )

        result_cem = sampler_cem.minimize_energy(b, J, n_samples=5, n_steps=20, beta=5.0)
        result_no_cem = sampler_no_cem.minimize_energy(b, J, n_samples=5, n_steps=20, beta=5.0)

        assert result_cem.shape == (5, n)
        assert result_no_cem.shape == (5, n)


class TestSampleMethod:
    """REQ-SAMPLE-003: sample() interface with config dict."""

    def test_config_beta_used(self):
        """SCENARIO-SAMPLE-LSB-005: config['beta'] overrides default."""
        n = 6
        sampler = LangevinSBSampler(beta=1.0, n_warmup=5, n_samples=3, steps_per_sample=2)
        b = np.zeros(n, dtype=np.float32)
        J = np.zeros((n, n), dtype=np.float32)
        # Just verify no crash and correct shape.
        result = sampler.sample(b, J, n_samples=3, config={"beta": 20.0})
        assert result.shape == (3, n)

    def test_config_n_warmup_override(self):
        """SCENARIO-SAMPLE-LSB-005: config n_warmup overrides instance setting."""
        n = 6
        sampler = LangevinSBSampler(n_warmup=1000, n_samples=3, steps_per_sample=2)
        b = np.zeros(n, dtype=np.float32)
        J = np.zeros((n, n), dtype=np.float32)
        # With n_warmup=5 in config, should run quickly.
        result = sampler.sample(b, J, n_samples=3, config={"beta": 5.0, "n_warmup": 5})
        assert result.shape == (3, n)

    def test_original_attributes_restored_after_sample(self):
        """SCENARIO-SAMPLE-LSB-005: sample() restores original attributes after call."""
        sampler = LangevinSBSampler(n_warmup=100, n_samples=10, steps_per_sample=5, beta=8.0)
        b = np.zeros(4, dtype=np.float32)
        J = np.zeros((4, 4), dtype=np.float32)
        sampler.sample(
            b, J, n_samples=3, config={"beta": 2.0, "n_warmup": 3, "steps_per_sample": 1}
        )
        # Attributes should be restored.
        assert sampler.n_warmup == 100
        assert sampler.n_samples == 10
        assert sampler.steps_per_sample == 5

    def test_original_attributes_restored_after_minimize_energy(self):
        """SCENARIO-SAMPLE-LSB-005: minimize_energy restores original attributes after call."""
        sampler = LangevinSBSampler(n_warmup=200, n_samples=15, steps_per_sample=7)
        b = np.zeros(4, dtype=np.float32)
        J = np.zeros((4, 4), dtype=np.float32)
        sampler.minimize_energy(b, J, n_samples=3, n_steps=5, beta=5.0)
        assert sampler.n_warmup == 200
        assert sampler.n_samples == 15


class TestLsbSampleFunctional:
    """REQ-SAMPLE-LSB-001: Functional lsb_sample() interface."""

    def test_functional_shape(self):
        """SCENARIO-SAMPLE-LSB-006: lsb_sample returns correct shape."""
        n = 10
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        result = lsb_sample(
            jrandom.PRNGKey(0), b, J, beta=5.0, lr=0.05, n_steps=20, n_samples=6, steps_per_sample=3
        )
        assert result.shape == (6, n)

    def test_functional_output_boolean(self):
        """SCENARIO-SAMPLE-LSB-006: lsb_sample returns boolean array."""
        n = 8
        b = jnp.ones(n)
        J = jnp.zeros((n, n))
        result = lsb_sample(jrandom.PRNGKey(1), b, J, beta=5.0, lr=0.05, n_steps=10, n_samples=4)
        assert result.dtype == jnp.bool_


class TestIsLsbEnabled:
    """REQ-SAMPLE-LSB-001: CARNOT_USE_LSB feature flag."""

    def test_disabled_by_default(self):
        """SCENARIO-SAMPLE-LSB-007: LSB is disabled when env var is absent."""
        env_backup = os.environ.pop("CARNOT_USE_LSB", None)
        try:
            assert is_lsb_enabled() is False
        finally:
            if env_backup is not None:
                os.environ["CARNOT_USE_LSB"] = env_backup

    def test_enabled_when_flag_set(self):
        """SCENARIO-SAMPLE-LSB-007: LSB is enabled when CARNOT_USE_LSB=1."""
        old = os.environ.get("CARNOT_USE_LSB")
        os.environ["CARNOT_USE_LSB"] = "1"
        try:
            assert is_lsb_enabled() is True
        finally:
            if old is None:
                del os.environ["CARNOT_USE_LSB"]
            else:
                os.environ["CARNOT_USE_LSB"] = old

    def test_disabled_when_flag_zero(self):
        """SCENARIO-SAMPLE-LSB-007: LSB is disabled when CARNOT_USE_LSB=0."""
        old = os.environ.get("CARNOT_USE_LSB")
        os.environ["CARNOT_USE_LSB"] = "0"
        try:
            assert is_lsb_enabled() is False
        finally:
            if old is None:
                del os.environ["CARNOT_USE_LSB"]
            else:
                os.environ["CARNOT_USE_LSB"] = old


class TestLsbSamplerWithCouplings:
    """REQ-SAMPLE-LSB-001: LSB handles non-trivial coupling structures."""

    def test_ferromagnetic_coupling(self):
        """SCENARIO-SAMPLE-LSB-008: Strong ferromagnetic coupling aligns spins."""
        n = 6
        # Strong ferromagnetic J → all spins prefer to agree.
        J = jnp.ones((n, n)) * 2.0
        J = J.at[jnp.arange(n), jnp.arange(n)].set(0.0)  # zero diagonal
        b = jnp.ones(n) * 0.5  # slight pull toward spin=1

        sampler = LangevinSBSampler(
            n_warmup=300, n_samples=20, steps_per_sample=5, use_cem=False, seed=42
        )
        samples = sampler.run_sampler(jrandom.PRNGKey(42), b, J, beta=5.0)
        assert samples.shape == (20, n)
        # Most samples should have spins mostly agreeing (high or low activation).
        per_sample_mean = jnp.mean(samples.astype(jnp.float32), axis=1)
        # Expect most samples to be either all-1 or all-0 (bimodal).
        # At least some samples should be high-activation (all spins agree at 1).
        has_high = float(jnp.any(per_sample_mean > 0.8))
        has_low = float(jnp.any(per_sample_mean < 0.2))
        assert has_high or has_low, "Ferromagnetic coupling should produce aligned states"

    def test_numpy_input_accepted(self):
        """SCENARIO-SAMPLE-LSB-008: Accepts numpy arrays as biases/couplings."""
        n = 8
        b_np = np.random.default_rng(0).normal(0, 1, n).astype(np.float32)
        J_np = np.zeros((n, n), dtype=np.float32)
        sampler = LangevinSBSampler(n_warmup=10, n_samples=4, steps_per_sample=2)
        result = sampler.minimize_energy(b_np, J_np, n_samples=4, n_steps=10, beta=5.0)
        assert result.shape == (4, n)
        assert result.dtype == np.bool_
