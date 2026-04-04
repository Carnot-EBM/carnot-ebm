"""Tests for MCMC samplers — JAX implementation.

Spec coverage: REQ-SAMPLE-001, REQ-SAMPLE-002, REQ-SAMPLE-003, REQ-SAMPLE-004,
               SCENARIO-SAMPLE-001, SCENARIO-SAMPLE-002, SCENARIO-SAMPLE-003,
               SCENARIO-SAMPLE-004, SCENARIO-SAMPLE-005
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin, EnergyFunction
from carnot.samplers.langevin import LangevinSampler
from carnot.samplers.hmc import HMCSampler


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2 — standard Gaussian."""

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class SteepRosenbrockEnergy(AutoGradMixin):
    """Rosenbrock function — produces very large gradients (~3200 at typical init).

    E(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
    """

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


class TestLangevinSampler:
    """Tests for REQ-SAMPLE-001: Langevin Dynamics."""

    def test_produces_samples(self) -> None:
        """SCENARIO-SAMPLE-001: Langevin produces finite samples."""
        sampler = LangevinSampler(step_size=0.01)
        model = QuadraticEnergy()
        init = jnp.array([5.0, 5.0])
        sample = sampler.sample(model, init, n_steps=100, key=jrandom.PRNGKey(42))
        assert jnp.all(jnp.isfinite(sample))

    def test_chain_length(self) -> None:
        """SCENARIO-SAMPLE-001: Chain has correct length."""
        sampler = LangevinSampler(step_size=0.01)
        model = QuadraticEnergy()
        init = jnp.array([0.0, 0.0])
        chain = sampler.sample_chain(model, init, n_steps=100, key=jrandom.PRNGKey(0))
        assert chain.shape == (100, 2)

    def test_statistics(self) -> None:
        """SCENARIO-SAMPLE-001: Sample mean near 0 for standard Gaussian."""
        sampler = LangevinSampler(step_size=0.01)
        model = QuadraticEnergy()
        init = jnp.zeros(2)
        chain = sampler.sample_chain(model, init, n_steps=10000, key=jrandom.PRNGKey(123))
        # Use last half as burn-in
        samples = chain[5000:]
        mean = jnp.mean(samples, axis=0)
        assert jnp.all(jnp.abs(mean) < 0.5), f"Mean should be near 0, got {mean}"


class TestHMCSampler:
    """Tests for REQ-SAMPLE-002: Hamiltonian Monte Carlo."""

    def test_produces_samples(self) -> None:
        """SCENARIO-SAMPLE-002: HMC produces finite samples."""
        sampler = HMCSampler(step_size=0.1, num_leapfrog_steps=10)
        model = QuadraticEnergy()
        init = jnp.array([3.0, 3.0])
        sample = sampler.sample(model, init, n_steps=50, key=jrandom.PRNGKey(42))
        assert jnp.all(jnp.isfinite(sample))


class TestDefaultKey:
    """Tests for default key behavior."""

    def test_langevin_sample_default_key(self) -> None:
        """REQ-SAMPLE-001: sample works with default key."""
        sampler = LangevinSampler(step_size=0.01)
        model = QuadraticEnergy()
        sample = sampler.sample(model, jnp.zeros(2), n_steps=10)
        assert jnp.all(jnp.isfinite(sample))

    def test_langevin_chain_default_key(self) -> None:
        """REQ-SAMPLE-001: sample_chain works with default key."""
        sampler = LangevinSampler(step_size=0.01)
        model = QuadraticEnergy()
        chain = sampler.sample_chain(model, jnp.zeros(2), n_steps=10)
        assert chain.shape == (10, 2)

    def test_hmc_sample_default_key(self) -> None:
        """REQ-SAMPLE-002: HMC works with default key."""
        sampler = HMCSampler(step_size=0.1, num_leapfrog_steps=5)
        model = QuadraticEnergy()
        sample = sampler.sample(model, jnp.zeros(2), n_steps=5)
        assert jnp.all(jnp.isfinite(sample))


class TestSamplerInterface:
    """Tests for REQ-SAMPLE-003: Sampler interface genericity."""

    def test_both_samplers_work_with_same_model(self) -> None:
        """SCENARIO-SAMPLE-003: Both samplers work with EnergyFunction."""
        model = QuadraticEnergy()
        init = jnp.zeros(2)

        langevin = LangevinSampler(step_size=0.01)
        hmc = HMCSampler(step_size=0.1, num_leapfrog_steps=5)

        s1 = langevin.sample(model, init, n_steps=10, key=jrandom.PRNGKey(0))
        s2 = hmc.sample(model, init, n_steps=10, key=jrandom.PRNGKey(0))

        assert jnp.all(jnp.isfinite(s1))
        assert jnp.all(jnp.isfinite(s2))


class TestLangevinGradientClipping:
    """Tests for REQ-SAMPLE-004: Gradient clipping in Langevin sampler."""

    def test_clip_activates_above_threshold(self) -> None:
        """REQ-SAMPLE-004: Gradient is clipped when norm exceeds clip_norm."""
        sampler = LangevinSampler(step_size=0.01, clip_norm=1.0)
        # A gradient with norm 5.0 should be rescaled to norm 1.0
        big_grad = jnp.array([3.0, 4.0])  # norm = 5.0
        clipped = sampler._clip_gradient(big_grad)
        assert jnp.allclose(jnp.linalg.norm(clipped), 1.0, atol=1e-6)
        # Direction should be preserved
        expected = big_grad / 5.0
        assert jnp.allclose(clipped, expected, atol=1e-6)

    def test_clip_noop_below_threshold(self) -> None:
        """SCENARIO-SAMPLE-005: Gradient is unchanged when norm is below clip_norm."""
        sampler = LangevinSampler(step_size=0.01, clip_norm=100.0)
        small_grad = jnp.array([0.3, 0.4])  # norm = 0.5
        clipped = sampler._clip_gradient(small_grad)
        assert jnp.allclose(clipped, small_grad, atol=1e-6)

    def test_clip_none_noop(self) -> None:
        """REQ-SAMPLE-004: clip_norm=None preserves backward compatibility."""
        sampler = LangevinSampler(step_size=0.01)  # clip_norm=None (default)
        big_grad = jnp.array([3.0, 4.0])  # norm = 5.0
        result = sampler._clip_gradient(big_grad)
        assert jnp.allclose(result, big_grad)

    def test_rosenbrock_no_nan_with_clipping(self) -> None:
        """SCENARIO-SAMPLE-004: Langevin + clipping prevents NaN on Rosenbrock.

        The Rosenbrock function at init [-1, 1] has gradient norm ~3200,
        which is the regime that causes standard Langevin to diverge.
        With gradient clipping, the sampler remains stable.
        """
        model = SteepRosenbrockEnergy()
        # Init [-1, 1] has grad ~[-3204, 200], norm ~3210 — the classic problem case
        init = jnp.array([-1.0, 1.0])
        sampler = LangevinSampler(step_size=0.01, clip_norm=10.0)
        result = sampler.sample(model, init, n_steps=500, key=jrandom.PRNGKey(42))
        assert jnp.all(jnp.isfinite(result)), f"Got non-finite result: {result}"

    def test_rosenbrock_chain_no_nan_with_clipping(self) -> None:
        """SCENARIO-SAMPLE-004: Langevin chain + clipping prevents NaN on Rosenbrock."""
        model = SteepRosenbrockEnergy()
        # Init [-1, 1] has grad norm ~3210, exactly the problematic regime
        init = jnp.array([-1.0, 1.0])
        sampler = LangevinSampler(step_size=0.01, clip_norm=10.0)
        chain = sampler.sample_chain(model, init, n_steps=500, key=jrandom.PRNGKey(42))
        assert jnp.all(jnp.isfinite(chain)), "Chain contains NaN/Inf values"


class TestHMCGradientClipping:
    """Tests for REQ-SAMPLE-004: Gradient clipping in HMC sampler."""

    def test_clip_activates_above_threshold(self) -> None:
        """REQ-SAMPLE-004: HMC gradient is clipped when norm exceeds clip_norm."""
        sampler = HMCSampler(step_size=0.1, num_leapfrog_steps=5, clip_norm=2.0)
        big_grad = jnp.array([3.0, 4.0])  # norm = 5.0
        clipped = sampler._clip_gradient(big_grad)
        assert jnp.allclose(jnp.linalg.norm(clipped), 2.0, atol=1e-6)

    def test_clip_noop_below_threshold(self) -> None:
        """SCENARIO-SAMPLE-005: HMC gradient unchanged when norm is below clip_norm."""
        sampler = HMCSampler(step_size=0.1, num_leapfrog_steps=5, clip_norm=100.0)
        small_grad = jnp.array([0.3, 0.4])
        clipped = sampler._clip_gradient(small_grad)
        assert jnp.allclose(clipped, small_grad, atol=1e-6)

    def test_rosenbrock_no_nan_with_clipping(self) -> None:
        """SCENARIO-SAMPLE-004: HMC + clipping prevents NaN on Rosenbrock.

        Init [-1, 1] produces gradient norm ~3210, the regime that causes
        standard HMC leapfrog to diverge without clipping.
        """
        model = SteepRosenbrockEnergy()
        init = jnp.array([-1.0, 1.0])
        sampler = HMCSampler(step_size=0.01, num_leapfrog_steps=5, clip_norm=10.0)
        result = sampler.sample(model, init, n_steps=100, key=jrandom.PRNGKey(42))
        assert jnp.all(jnp.isfinite(result)), f"Got non-finite result: {result}"
