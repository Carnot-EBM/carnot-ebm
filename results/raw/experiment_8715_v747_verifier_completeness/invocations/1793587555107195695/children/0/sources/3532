import jax
import jax.numpy as jnp
from typing import Callable, Tuple

class EBRMSampler:
    """Energy-Based Reasoning Model Sampler.
    
    Wraps latent updates using gradient descent to minimize an energy function.
    """
    def __init__(self, energy_fn: Callable[[jnp.ndarray], jnp.ndarray], learning_rate: float = 0.1, steps: int = 100):
        """
        Args:
            energy_fn: A function that takes a latent state (trace) and returns a scalar energy.
            learning_rate: Step size for gradient descent.
            steps: Number of gradient descent steps to perform.
        """
        self.energy_fn = energy_fn
        self.learning_rate = learning_rate
        self.steps = steps
        
    def sample(self, init_latents: jnp.ndarray) -> jnp.ndarray:
        """Refines the initial latents via gradient descent to minimize energy.
        
        Args:
            init_latents: The initial latent state [..., dim].
            
        Returns:
            The refined latent state.
        """
        def loss_fn(latents):
            return self.energy_fn(latents)
            
        grad_fn = jax.grad(loss_fn)
        
        latents = init_latents
        for _ in range(self.steps):
            grads = grad_fn(latents)
            latents = latents - self.learning_rate * grads
            
        return latents
