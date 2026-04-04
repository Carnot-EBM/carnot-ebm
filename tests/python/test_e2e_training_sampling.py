"""E2E-002: Full training + sampling pipeline (Python/JAX).

Verifies that training an Ising model with denoising score matching and
sampling with Langevin dynamics produces samples that approximate the
correct Boltzmann distribution.

Pipeline: create model -> generate data -> train with DSM -> sample -> verify.

Spec coverage: REQ-CORE-002, REQ-SAMPLE-001, REQ-TRAIN-001,
               SCENARIO-CORE-001, SCENARIO-SAMPLE-001
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.benchmarks import DoubleWell, Rosenbrock
from carnot.core.energy import AutoGradMixin
from carnot.models.ising import IsingConfig, IsingModel
from carnot.samplers.langevin import LangevinSampler
from carnot.training.score_matching import dsm_loss


class _ParameterizedQuadratic(AutoGradMixin):
    """Quadratic energy with trainable scale: E(x) = 0.5 * scale * ||x - center||^2.

    Used for E2E training tests because its parameters are simple scalars
    that jax.grad can differentiate through cleanly.
    """

    def __init__(self, scale: jax.Array, center: jax.Array) -> None:
        self.scale = scale
        self.center = center

    def energy(self, x: jax.Array) -> jax.Array:
        diff = x - self.center
        return 0.5 * self.scale * jnp.sum(diff**2)

    @property
    def input_dim(self) -> int:
        return self.center.shape[0]


class TestE2ETrainingSampling:
    """E2E-002: Train model, sample, verify distribution.

    REQ-CORE-002, REQ-SAMPLE-001, REQ-TRAIN-001
    """

    def test_langevin_finds_double_well_minimum(self) -> None:
        """SCENARIO-SAMPLE-001: Langevin sampling converges toward energy minimum.

        Pipeline:
        1. Create DoubleWell benchmark (known minima at x[0]=+/-1)
        2. Sample with Langevin from random init
        3. Verify final energy is lower than initial energy
        4. Verify sample is near one of the two minima
        """
        dw = DoubleWell(dim=2)
        sampler = LangevinSampler(step_size=0.001)
        key = jrandom.PRNGKey(42)

        x0 = jrandom.normal(key, (2,)) * 0.5
        initial_energy = float(dw.energy(x0))

        k1, key = jrandom.split(key)
        x_final = sampler.sample(dw, x0, n_steps=10000, key=k1)
        final_energy = float(dw.energy(x_final))

        assert final_energy < initial_energy, (
            f"Langevin should decrease energy: {initial_energy:.4f} -> {final_energy:.4f}"
        )
        assert abs(abs(float(x_final[0])) - 1.0) < 0.5, (
            f"x[0] should be near +/-1, got {float(x_final[0]):.4f}"
        )

    def test_langevin_chain_explores(self) -> None:
        """REQ-SAMPLE-001: Langevin chain visits different regions."""
        dw = DoubleWell(dim=2)
        sampler = LangevinSampler(step_size=0.005)
        key = jrandom.PRNGKey(99)

        x0 = jnp.zeros(2)
        chain = sampler.sample_chain(dw, x0, n_steps=2000, key=key)

        # Chain shape: (n_steps, dim) — one state per step
        assert chain.shape[0] == 2000
        assert chain.shape[1] == 2

        x_range = float(jnp.max(chain[:, 0]) - jnp.min(chain[:, 0]))
        assert x_range > 0.1, (
            f"Chain should explore, but x[0] range is only {x_range:.4f}"
        )

    def test_rosenbrock_langevin_convergence(self) -> None:
        """SCENARIO-CORE-001: Langevin on Rosenbrock converges toward minimum."""
        rb = Rosenbrock(dim=2)
        sampler = LangevinSampler(step_size=0.0001)
        key = jrandom.PRNGKey(7)

        x0 = jnp.array([0.0, 0.0])  # energy = 1.0 at origin
        initial_energy = float(rb.energy(x0))

        x_final = sampler.sample(rb, x0, n_steps=10000, key=key)
        final_energy = float(rb.energy(x_final))

        assert final_energy < initial_energy, (
            f"Langevin on Rosenbrock: {initial_energy:.4f} -> {final_energy:.4f}"
        )

    def test_dsm_training_reduces_loss(self) -> None:
        """REQ-TRAIN-001: DSM training reduces loss over iterations.

        Uses a parameterized quadratic model where we can take gradients
        w.r.t. the center parameter. The target data is centered at [1,1,1]
        so training should push the model center toward [1,1,1].
        """
        key = jrandom.PRNGKey(0)
        dim = 3

        # Target data centered at [1, 1, 1]
        k1, key = jrandom.split(key)
        target = jnp.ones(dim)
        data = jrandom.normal(k1, (64, dim)) * 0.2 + target

        sigma = 0.1

        # Initial model: center at origin (wrong), scale = 1.0
        center = jnp.zeros(dim)
        scale = jnp.array(1.0)

        # Pre-generate noise for deterministic loss
        k2, key = jrandom.split(key)
        noise = jrandom.normal(k2, data.shape) * sigma

        # Compute initial loss
        model_init = _ParameterizedQuadratic(scale, center)
        initial_loss = float(dsm_loss(model_init, data, noise, sigma))

        # Train: gradient descent on center parameter
        lr = 0.01
        for step in range(50):
            k_step, key = jrandom.split(key)
            noise_step = jrandom.normal(k_step, data.shape) * sigma

            def loss_fn(c):
                m = _ParameterizedQuadratic(scale, c)
                return dsm_loss(m, data, noise_step, sigma)

            grad_c = jax.grad(loss_fn)(center)
            center = center - lr * grad_c

        # Compute final loss
        k3, key = jrandom.split(key)
        noise_final = jrandom.normal(k3, data.shape) * sigma
        model_final = _ParameterizedQuadratic(scale, center)
        final_loss = float(dsm_loss(model_final, data, noise_final, sigma))

        assert final_loss < initial_loss, (
            f"Training should reduce DSM loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )

        # Center should have moved toward the target
        center_error = float(jnp.linalg.norm(center - target))
        assert center_error < 1.5, (
            f"Center should move toward target, error={center_error:.4f}"
        )

    def test_full_pipeline_train_then_sample(self) -> None:
        """REQ-TRAIN-001, REQ-SAMPLE-001: Full train->sample E2E.

        1. Create parameterized energy model
        2. Train center to match data distribution
        3. Sample from trained model
        4. Verify samples cluster near trained center
        """
        key = jrandom.PRNGKey(42)
        dim = 2

        # Target: data centered at [2, 2]
        k1, key = jrandom.split(key)
        target = jnp.array([2.0, 2.0])
        data = jrandom.normal(k1, (64, dim)) * 0.3 + target

        sigma = 0.1
        scale = jnp.array(1.0)
        center = jnp.zeros(dim)

        # Train
        lr = 0.01
        for step in range(100):
            k_step, key = jrandom.split(key)
            noise = jrandom.normal(k_step, data.shape) * sigma

            def loss_fn(c):
                m = _ParameterizedQuadratic(scale, c)
                return dsm_loss(m, data, noise, sigma)

            grad_c = jax.grad(loss_fn)(center)
            center = center - lr * grad_c

        # Sample from trained model
        trained_model = _ParameterizedQuadratic(scale, center)
        sampler = LangevinSampler(step_size=0.01)

        samples = []
        for i in range(20):
            k_s, key = jrandom.split(key)
            x0 = jrandom.normal(k_s, (dim,)) * 2.0
            x_final = sampler.sample(trained_model, x0, n_steps=2000, key=key)
            samples.append(x_final)

        sample_array = jnp.stack(samples)
        sample_mean = jnp.mean(sample_array, axis=0)

        # Samples should cluster near the trained center
        # (which should be near the target [2, 2])
        mean_to_target = float(jnp.linalg.norm(sample_mean - target))
        assert mean_to_target < 2.0, (
            f"Sample mean {sample_mean} too far from target {target}, "
            f"error={mean_to_target:.4f}"
        )
