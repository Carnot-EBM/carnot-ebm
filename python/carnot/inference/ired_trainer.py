"""
IRED Trainer Module.

Trains an energy function to map input constraints to continuous latent outputs.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
from typing import Tuple, Any, Callable

class EnergyModel(nn.Module):
    """A simple model to map constraints to a target continuous state."""
    hidden_dim: int = 16
    output_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def create_train_state(rng: Any, input_dim: int, output_dim: int, learning_rate: float) -> TrainState:
    """Creates initial training state."""
    model = EnergyModel(output_dim=output_dim)
    params = model.init(rng, jnp.ones((1, input_dim)))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state: TrainState, constraints: jnp.ndarray, targets: jnp.ndarray) -> Tuple[TrainState, float]:
    """Performs a single training step."""
    def loss_fn(params):
        predicted_targets = state.apply_fn({'params': params}, constraints)
        loss = jnp.mean((predicted_targets - targets) ** 2)
        return loss
        
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def get_energy_fn(state: TrainState, constraint: np.ndarray) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
    """
    Returns an energy_fn suitable for IREDOptimizer given a trained state and a specific constraint.
    """
    constraint_jnp = jnp.array(constraint)
    predicted_target = state.apply_fn({'params': state.params}, constraint_jnp)
    target_np = np.array(predicted_target)
    
    def energy_fn(x: np.ndarray) -> Tuple[float, np.ndarray]:
        diff = x - target_np
        energy = float(np.sum(diff ** 2))
        grad = 2.0 * diff
        return energy, grad
        
    return energy_fn
