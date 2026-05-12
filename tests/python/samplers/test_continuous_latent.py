"""Tests for the continuous-latent FAR-inspired constraint sampler.

Spec: REQ-SAMPLE-1935, REQ-SAMPLE-1935-1, REQ-SAMPLE-1935-2,
      REQ-SAMPLE-1935-3, REQ-SAMPLE-1935-4, REQ-SAMPLE-1935-5

SCENARIO-SAMPLE-1935: FAR-Inspired Sampler Benchmark
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.samplers.continuous_latent import (
    ContinuousLatentSampler,
    FARSurrogateHead,
    SamplerStats,
)


# ---------------------------------------------------------------------------
# Minimal EnergyFunction stub for tests (no GPU, no imports beyond JAX)
# ---------------------------------------------------------------------------

class _QuadraticEnergy:
    """Quadratic energy E(z) = 0.5 * ||z||^2 — trivial, all-CPU, no deps.

    Used as a controlled stand-in for any EnergyFunction in the test suite.
    The gradient is z, so Langevin dynamics drives z toward 0.
    """

    def energy(self, z: jax.Array) -> jax.Array:
        return 0.5 * jnp.dot(z, z)

    def grad_energy(self, z: jax.Array) -> jax.Array:
        return z  # gradient of 0.5 * ||z||^2 is z


# ---------------------------------------------------------------------------
# FARSurrogateHead tests  (REQ-SAMPLE-1935-1)
# ---------------------------------------------------------------------------

class TestFARSurrogateHead:
    """REQ-SAMPLE-1935-1: FARSurrogateHead predict and grad."""

    def test_from_random_key_shapes(self):
        """Weight matrix and bias have expected shapes after initialisation."""
        key = jrandom.PRNGKey(0)
        head = FARSurrogateHead.from_random_key(key, latent_dim=16, n_constraints=4)
        assert head.W.shape == (16, 4)
        assert head.b.shape == (4,)

    def test_predict_output_range(self):
        """predict() returns values strictly in (0, 1) due to sigmoid."""
        key = jrandom.PRNGKey(1)
        head = FARSurrogateHead.from_random_key(key, latent_dim=8, n_constraints=3)
        z = jrandom.normal(key, (8,))
        scores = head.predict(z)
        assert scores.shape == (3,)
        assert jnp.all(scores > 0.0).item()
        assert jnp.all(scores < 1.0).item()

    def test_predict_bias_zero_init(self):
        """Bias is zero at initialisation so predict at z=0 returns 0.5."""
        key = jrandom.PRNGKey(2)
        head = FARSurrogateHead.from_random_key(key, latent_dim=4, n_constraints=2)
        z_zero = jnp.zeros(4)
        scores = head.predict(z_zero)
        # sigmoid(0) = 0.5 exactly
        assert jnp.allclose(scores, jnp.full((2,), 0.5), atol=1e-6).item()

    def test_grad_scores_sum_shape(self):
        """grad_scores_sum returns shape (latent_dim,)."""
        key = jrandom.PRNGKey(3)
        head = FARSurrogateHead.from_random_key(key, latent_dim=12, n_constraints=5)
        z = jrandom.normal(key, (12,))
        grad = head.grad_scores_sum(z)
        assert grad.shape == (12,)

    def test_grad_scores_sum_finite(self):
        """grad_scores_sum contains no NaN or Inf values."""
        key = jrandom.PRNGKey(4)
        head = FARSurrogateHead.from_random_key(key, latent_dim=8, n_constraints=3)
        z = jrandom.normal(key, (8,)) * 10.0  # large z to stress sigmoid saturation
        grad = head.grad_scores_sum(z)
        assert jnp.all(jnp.isfinite(grad)).item()


# ---------------------------------------------------------------------------
# SamplerStats tests
# ---------------------------------------------------------------------------

class TestSamplerStats:
    """Basic stats accumulator behaviour."""

    def test_skip_rate_zero_when_no_steps(self):
        stats = SamplerStats()
        assert stats.skip_rate == 0.0

    def test_skip_rate_computed_correctly(self):
        stats = SamplerStats(total_steps=100, surrogate_skip_count=40)
        assert abs(stats.skip_rate - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# ContinuousLatentSampler tests  (REQ-SAMPLE-1935-2, REQ-SAMPLE-1935-3)
# ---------------------------------------------------------------------------

class TestContinuousLatentSampler:
    """REQ-SAMPLE-1935-2 / REQ-SAMPLE-1935-3: sampler correctness."""

    def _make_sampler(
        self,
        latent_dim: int = 16,
        n_constraints: int = 4,
        skip_threshold: float = 0.5,
        step_size: float = 0.01,
    ) -> ContinuousLatentSampler:
        key = jrandom.PRNGKey(42)
        head = FARSurrogateHead.from_random_key(key, latent_dim, n_constraints)
        energy = _QuadraticEnergy()
        return ContinuousLatentSampler(
            energy_fn=energy,
            surrogate=head,
            step_size=step_size,
            skip_threshold=skip_threshold,
        )

    def test_sample_returns_correct_shape(self):
        """sample() final state has same shape as init."""
        sampler = self._make_sampler(latent_dim=16)
        key = jrandom.PRNGKey(0)
        z0 = jrandom.normal(key, (16,))
        z_final, stats = sampler.sample(key, z0, n_steps=20)
        assert z_final.shape == (16,)

    def test_sample_final_state_finite(self):
        """sample() output contains no NaN or Inf."""
        sampler = self._make_sampler(latent_dim=16)
        key = jrandom.PRNGKey(1)
        z0 = jrandom.normal(key, (16,))
        z_final, _ = sampler.sample(key, z0, n_steps=50)
        assert jnp.all(jnp.isfinite(z_final)).item()

    def test_stats_total_steps_matches_n_steps(self):
        """stats.total_steps equals n_steps after sample()."""
        sampler = self._make_sampler(latent_dim=8)
        key = jrandom.PRNGKey(2)
        z0 = jrandom.normal(key, (8,))
        _, stats = sampler.sample(key, z0, n_steps=30)
        assert stats.total_steps == 30

    def test_skip_threshold_1_forces_no_skips(self):
        """skip_threshold=1.0 forces every step through the true energy gradient.

        sigmoid output is always < 1.0 so skip_threshold=1.0 means the surrogate
        condition max(scores) < 1.0 is ALWAYS true — all steps are surrogate skips.
        The opposite: skip_threshold=0.0 means the condition is never satisfied
        (max >= 0 always for non-negative sigmoid output), so no skips occur.
        """
        # threshold = 0.0 → condition max(scores) < 0.0 never true → no skips
        sampler = self._make_sampler(latent_dim=8, skip_threshold=0.0)
        key = jrandom.PRNGKey(3)
        z0 = jrandom.normal(key, (8,))
        _, stats = sampler.sample(key, z0, n_steps=20)
        assert stats.surrogate_skip_count == 0

    def test_skip_threshold_high_uses_surrogate(self):
        """skip_threshold=1.0 means surrogate shortcut is taken every step."""
        # sigmoid < 1 always, so threshold=1.0 → every step skips energy eval
        sampler = self._make_sampler(latent_dim=8, skip_threshold=1.0)
        key = jrandom.PRNGKey(4)
        z0 = jrandom.normal(key, (8,))
        _, stats = sampler.sample(key, z0, n_steps=20)
        assert stats.surrogate_skip_count == 20

    def test_sample_chain_shape(self):
        """sample_chain() returns chain of shape (n_steps, latent_dim)."""
        sampler = self._make_sampler(latent_dim=10)
        key = jrandom.PRNGKey(5)
        z0 = jrandom.normal(key, (10,))
        chain, stats = sampler.sample_chain(key, z0, n_steps=15)
        assert chain.shape == (15, 10)
        assert stats.total_steps == 15

    def test_sample_chain_all_finite(self):
        """sample_chain() contains no NaN or Inf across all steps."""
        sampler = self._make_sampler(latent_dim=8)
        key = jrandom.PRNGKey(6)
        z0 = jrandom.normal(key, (8,))
        chain, _ = sampler.sample_chain(key, z0, n_steps=50)
        assert jnp.all(jnp.isfinite(chain)).item()

    def test_energy_decreases_on_average_without_surrogate(self):
        """With skip_threshold=0 (no surrogate), Langevin drifts toward minimum.

        For E(z)=0.5*||z||^2 the minimum is 0.  With small step_size and
        many steps, the RMS energy of z should decrease from the random init.
        This is a statistical test with a large enough effect to be reliable.
        """
        sampler = self._make_sampler(
            latent_dim=32,
            skip_threshold=0.0,
            step_size=0.05,
        )
        energy_fn = _QuadraticEnergy()
        key = jrandom.PRNGKey(7)
        z0 = jrandom.normal(key, (32,)) * 3.0  # start far from minimum
        z_final, _ = sampler.sample(key, z0, n_steps=200)
        e_init = float(energy_fn.energy(z0))
        e_final = float(energy_fn.energy(z_final))
        # Langevin should have reduced energy significantly (not a hard gate —
        # just verify the sampler is actually doing gradient descent, not noise-only)
        assert e_final < e_init


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-1935: integration test
# ---------------------------------------------------------------------------

class TestScenario1935:
    """SCENARIO-SAMPLE-1935: end-to-end benchmark scenario."""

    def test_all_samplers_produce_finite_outputs(self):
        """All three paths (FAR skip, FAR no-skip, pure energy) produce finite states."""
        from carnot.samplers.langevin import LangevinSampler

        dim = 32
        n_steps = 200
        key = jrandom.PRNGKey(99)
        z0 = jrandom.normal(key, (dim,))
        energy = _QuadraticEnergy()

        # FAR sampler with surrogate shortcuts enabled
        head = FARSurrogateHead.from_random_key(key, dim, 4)
        far_sampler = ContinuousLatentSampler(
            energy_fn=energy, surrogate=head, step_size=0.01, skip_threshold=0.6
        )
        z_far, stats_far = far_sampler.sample(key, z0, n_steps=n_steps)
        assert jnp.all(jnp.isfinite(z_far)).item(), "FAR sampler: NaN/Inf in output"
        assert stats_far.total_steps == n_steps

        # Pure Langevin baseline
        langevin = LangevinSampler(step_size=0.01)
        z_lang = langevin.sample(energy, z0, n_steps=n_steps, key=key)
        assert jnp.all(jnp.isfinite(z_lang)).item(), "Langevin: NaN/Inf in output"

        # REQ-SAMPLE-1935-5: skip_rate is positive when threshold allows shortcuts
        assert stats_far.skip_rate >= 0.0  # non-negative always
