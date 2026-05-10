"""Tests for NablaETS sampler.

Spec: REQ-VERIFY-1690, SCENARIO-VERIFY-1690
"""

import jax
import jax.numpy as jnp
from carnot.samplers.nabla_ets import NablaETS, NablaETSConfig


class MockEnergyFunction:
    """Mock energy function (simple quadratic) for testing."""
    
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return 2.0 * x


def test_nabla_ets_exposes_k_steps() -> None:
    """Test that NablaETS exposes the K_steps scaling parameter.

    Spec: REQ-VERIFY-1690-2, SCENARIO-VERIFY-1690
    """
    config = NablaETSConfig(K_steps=42)
    sampler = NablaETS(config)
    assert sampler.config.K_steps == 42


def test_nabla_ets_optimizes_latent_state() -> None:
    """Test that NablaETS optimizes latent state guided by energy.

    Spec: REQ-VERIFY-1690-3, SCENARIO-VERIFY-1690
    """
    config = NablaETSConfig(K_steps=100, step_size=0.1)
    sampler = NablaETS(config)

    energy_fn = MockEnergyFunction()
    init_latent = jnp.array([5.0, -5.0])
    key = jax.random.PRNGKey(0)

    # Need type ignore as MockEnergyFunction doesn't formally implement EnergyFunction protocol
    optimized_latent = sampler.optimize_latent_state(energy_fn, init_latent, key)  # type: ignore

    assert optimized_latent.shape == init_latent.shape

    init_norm = jnp.linalg.norm(init_latent)
    opt_norm = jnp.linalg.norm(optimized_latent)
    assert opt_norm < init_norm
