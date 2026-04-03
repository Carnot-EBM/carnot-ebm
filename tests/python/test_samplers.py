"""Tests for MCMC samplers — JAX implementation.

Spec coverage: REQ-SAMPLE-001, REQ-SAMPLE-002, REQ-SAMPLE-003,
               SCENARIO-SAMPLE-001, SCENARIO-SAMPLE-002, SCENARIO-SAMPLE-003
"""

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
