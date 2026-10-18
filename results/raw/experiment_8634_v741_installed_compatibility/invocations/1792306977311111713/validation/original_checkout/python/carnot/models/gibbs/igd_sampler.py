"""Interleaved Gibbs Diffusion (IGD) Sampler.

Spec: REQ-IGD-001, REQ-IGD-002, REQ-IGD-1961
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple

class IGDSampler:
    """A sampler that interleaves continuous logit noise injection with discrete updates."""

    def __init__(self, energy_fn: Callable, num_vars: int, q: int = 2):
        self.energy_fn = energy_fn
        self.num_vars = num_vars
        self.q = q

    def sweep(self, key: jax.Array, state: jnp.ndarray, logits: jnp.ndarray, step_size: float = 0.01) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Performs one sweep: injects noise into continuous logits, then updates the discrete
        state based on the new logits + conditional energy.
        """
        k1, k2 = jax.random.split(key)
        
        # Inject finite Gaussian noise into continuous logits (REQ-IGD-1961-2)
        noise = jax.random.normal(k1, logits.shape)
        next_logits = logits + jnp.sqrt(2 * step_size) * noise
        
        # Update discrete state variables sequentially
        def body_fn(i, val):
            curr_state, k_inner = val
            k_step, k_next = jax.random.split(k_inner)
            
            # Evaluate energy for q=0
            state0 = curr_state.at[i].set(0)
            e0 = self.energy_fn(state0)
            
            # Evaluate energy for q=1
            state1 = curr_state.at[i].set(1)
            e1 = self.energy_fn(state1)
            
            # Log probabilities
            log_p0 = next_logits[i, 0] - e0
            log_p1 = next_logits[i, 1] - e1
            
            log_probs = jnp.stack([log_p0, log_p1])
            new_val = jax.random.categorical(k_step, log_probs)
            
            return curr_state.at[i].set(new_val), k_next

        final_state, _ = jax.lax.fori_loop(0, self.num_vars, body_fn, (state, k2))
        return final_state, next_logits
