import jax
import jax.numpy as jnp
from typing import Callable, Sequence

class CompositionalEnergy:
    """
    Composes independent local constraint energy functions into a global landscape.
    """
    def __init__(self, potentials: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]):
        self.potentials = tuple(potentials)
        
        @jax.jit
        def _call_fn(x: jnp.ndarray) -> jnp.ndarray:
            if not self.potentials:
                return jnp.array(0.0)
            
            # JAX jit unrolls this list comprehension and evaluates them in the computation graph
            evals = [p(x) for p in self.potentials]
            return jnp.sum(jnp.stack(evals, axis=0), axis=0)
            
        @jax.jit
        def _evaluate_array_fn(x: jnp.ndarray) -> jnp.ndarray:
            if not self.potentials:
                return jnp.array([])
                
            evals = [p(x) for p in self.potentials]
            return jnp.stack(evals, axis=0)
            
        self._call_fn = _call_fn
        self._evaluate_array_fn = _evaluate_array_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates all potentials in parallel over the latent state and returns their sum.
        
        Args:
            x: The latent state.
            
        Returns:
            The global composite energy.
        """
        return self._call_fn(x)

    def evaluate_array(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates all potentials over the latent state and returns an array of
        individual potential values. Useful for Lagrangian optimization.
        
        Args:
            x: The latent state.
            
        Returns:
            An array of shape (num_potentials, ...) containing the evaluated constraints.
        """
        return self._evaluate_array_fn(x)
