import jax
import jax.numpy as jnp
from typing import List, Callable

class CompositionalEnergyMinimizer:
    """
    Implements global energy landscape composition over small tractable subproblems.
    Refines a continuous latent trace via energy gradients from compositional sub-energies.
    """
    def __init__(self, sub_energies: List[Callable[[jnp.ndarray], jnp.ndarray]], learning_rate: float = 0.01):
        self.sub_energies = sub_energies
        self.learning_rate = learning_rate

    def compute_total_energy(self, state: jnp.ndarray) -> jnp.ndarray:
        """Computes the sum of all sub-energies."""
        total = jnp.zeros(())
        for energy_fn in self.sub_energies:
            total += energy_fn(state)
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
