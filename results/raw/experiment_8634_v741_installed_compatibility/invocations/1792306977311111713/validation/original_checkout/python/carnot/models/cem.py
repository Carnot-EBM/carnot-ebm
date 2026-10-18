"""
Compositional Energy Minimization (CEM) substrate.

Refines a continuous latent trace via energy gradients from compositional sub-energies.

Spec: REQ-CEM-004
"""

from typing import List, Sequence, Any
import jax
import jax.numpy as jnp
from carnot.models.ising import IsingModel

class ClauseEBM:
    """
    Energy Based Model for a single 3-SAT clause.
    Energy is minimized (approaches 0) when the clause is satisfied.

    Spec: REQ-CEM-005
    """
    def __init__(self, indices: jnp.ndarray, signs: jnp.ndarray):
        self.indices = jnp.array(indices)
        self.signs = jnp.array(signs)

    def energy(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes energy of the clause for a continuous state in [-1, 1]."""
        vals = state[self.indices]
        penalties = 1.0 - self.signs * vals
        penalties = jnp.maximum(0.0, penalties)
        return jnp.prod(penalties)

class CompositionalEnergyMinimizer:
    """
    Implements global energy landscape composition over small tractable subproblems.
    Sums multiple independent EBM instances.

    Spec: REQ-CEM-004, REQ-CEM-005, SCENARIO-CEM-002, SCENARIO-CEM-003
    """
    def __init__(self, sub_models: Sequence[Any], learning_rate: float = 0.01):
        self.sub_models = sub_models
        self.learning_rate = learning_rate

    def compute_total_energy(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the sum of all sub-energies."""
        total = jnp.zeros(())
        for model in self.sub_models:
            total += model.energy(state)
        return total

    def step(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Performs one gradient descent step on the compositional energy landscape."""
        grad_fn = jax.value_and_grad(self.compute_total_energy)
        energy_val, grad = grad_fn(state)
        new_state = state - self.learning_rate * grad
        return new_state, energy_val

    def minimize(self, init_state: jnp.ndarray, steps: int) -> tuple[jnp.ndarray, List[float]]:
        """Minimizes the energy over a given number of steps."""
        state = init_state
        energy_history = []
        for _ in range(steps):
            state, energy_val = self.step(state)
            energy_history.append(float(energy_val))
        return state, energy_history
