import jax
import jax.numpy as jnp
from typing import Callable, Tuple

class PEMOptimizer:
    """
    Parallel Energy Minimization (PEM) optimizer.
    Runs gradient-based Langevin dynamics over composed sub-problem landscapes.
    """
    def __init__(
        self,
        energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        learning_rate: float = 0.01,
        noise_scale: float = 0.01,
    ):
        """
        Args:
            energy_fn: Function returning a scalar energy value for a state x.
            learning_rate: Step size for the gradient descent drift term.
            noise_scale: Coefficient for the injected thermal noise.
        """
        self.energy_fn = energy_fn
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale
        
        self.grad_fn = jax.grad(energy_fn)
        
        @jax.jit
        def step_fn(x: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
            grad = self.grad_fn(x)
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x.shape)
            
            drift = -self.learning_rate * grad
            diffusion = self.noise_scale * jnp.sqrt(2.0 * self.learning_rate) * noise
            
            x_new = x + drift + diffusion
            energy = self.energy_fn(x)
            
            return x_new, energy, key
            
        self._step_fn = step_fn

        @jax.jit
        def optimize_fn(x_init: jnp.ndarray, key: jax.Array, steps: int) -> Tuple[jnp.ndarray, jax.Array]:
            def body_fun(i, val):
                x, key = val
                x_new, _, new_key = self._step_fn(x, key)
                return x_new, new_key
                
            return jax.lax.fori_loop(0, steps, body_fun, (x_init, key))
            
        self._optimize_fn = optimize_fn
        
    def step(self, x: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
        """Performs one step of Langevin dynamics optimization."""
        return self._step_fn(x, key)

    def optimize(self, x_init: jnp.ndarray, key: jax.Array, steps: int) -> Tuple[jnp.ndarray, jax.Array]:
        """Optimizes the state for a given number of steps using Langevin dynamics."""
        return self._optimize_fn(x_init, key, steps)
