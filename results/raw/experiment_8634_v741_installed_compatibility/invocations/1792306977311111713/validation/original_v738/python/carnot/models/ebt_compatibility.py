"""Energy-Based Transformer Compatibility Checking Prototype.

Spec: REQ-NRGPT-003, SCENARIO-NRGPT-003
"""

import jax
import jax.numpy as jnp
from carnot.models.boltzmann import _silu

class EBTCompatibilityModel:
    """EBT Compatibility prototype predicting energy between two sequences.
    
    The energy represents the compatibility (lower is more compatible).
    """
    def __init__(self, input_dim: int, hidden_dim: int, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        limit1 = jnp.sqrt(6.0 / (input_dim * 2 + hidden_dim))
        self.w1 = jax.random.uniform(k1, (hidden_dim, input_dim * 2), minval=-limit1, maxval=limit1)
        self.b1 = jnp.zeros(hidden_dim)
        
        limit2 = jnp.sqrt(6.0 / (hidden_dim + 1))
        self.w2 = jax.random.uniform(k2, (1, hidden_dim), minval=-limit2, maxval=limit2)
        self.b2 = jnp.zeros(1)

    def energy(self, seq_a: jax.Array, seq_b: jax.Array) -> jax.Array:
        """Compute scalar compatibility energy for sequence pair."""
        x = jnp.concatenate([seq_a, seq_b], axis=-1)
        h = _silu(self.w1 @ x + self.b1)
        out = self.w2 @ h + self.b2
        return out[0]

def ebt_compatibility_loop(model: EBTCompatibilityModel, seq_a: jax.Array, seq_b_init: jax.Array, steps: int = 10, lr: float = 0.1) -> tuple[jax.Array, list[float]]:
    """Descend the energy curve to optimize seq_b given seq_a."""
    def loss_fn(seq_b: jax.Array) -> jax.Array:
        return model.energy(seq_a, seq_b)
    
    grad_fn = jax.value_and_grad(loss_fn)
    
    seq_b = seq_b_init
    energy_curve = []
    
    for _ in range(steps):
        val, grad = grad_fn(seq_b)
        energy_curve.append(float(val))
        seq_b = seq_b - lr * grad
        
    return seq_b, energy_curve

def compare_with_log_prob(seq_a: jax.Array, seq_b: jax.Array) -> jax.Array:
    """Compare with traditional conditional log-probability.
    
    Returns a dummy log probability (e.g. isotropic Gaussian).
    """
    # Dummy traditional log-probability: -0.5 * ||seq_a - seq_b||^2
    return -0.5 * jnp.sum((seq_a - seq_b) ** 2)
