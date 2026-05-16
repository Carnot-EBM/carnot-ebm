"""Gibbs sampler with HardNet++ nonlinear projection.

Spec: REQ-SAMPLE-9999, SCENARIO-SAMPLE-9999
"""

import jax
import jax.numpy as jnp
from typing import Callable, Any
from carnot.models.gibbs.hardnet import DampedLinearizationLayer
from carnot.models.gibbs import GibbsModel

class HardNetGibbsSampler:
    """A continuous sampler for Gibbs models utilizing HardNet++ projection."""
    
    def __init__(self, model: GibbsModel, constraint_fn: Callable):
        self.model = model
        self.constraint_fn = constraint_fn
        self.hardnet = DampedLinearizationLayer()

    def sample(self, key: jax.Array, init_state: jnp.ndarray, num_steps: int, step_size: float = 0.01) -> jnp.ndarray:
        """
        Runs Langevin dynamics on the Gibbs model energy, then projects 
        infeasible states to the feasible boundary using HardNet++.
        """
        def step_fn(state, k):
            k1, k2 = jax.random.split(k)
            
            # Gradient of energy
            grad_E = self.model.grad_energy(state)
            
            # Langevin step (continuous relaxation)
            noise = jax.random.normal(k1, state.shape)
            next_state = state - step_size * grad_E + jnp.sqrt(2 * step_size) * noise
            
            # HardNet++ projection to prune infeasible states
            projected_state = self.hardnet.apply({}, next_state, self.constraint_fn)
            return projected_state, projected_state

        keys = jax.random.split(key, num_steps)
        final_state, _ = jax.lax.scan(step_fn, init_state, keys)
        return final_state
