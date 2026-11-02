"""Continuous Latent Constraint Optimizer using Langevin dynamics.

**Researcher summary:**
    Optimizes elicited constraints in a continuous latent space using
    Langevin dynamics. By treating HRM outputs or constraint violations
    as an energy landscape, this optimizer relaxes discrete constraints
    into continuous representations and performs gradient descent with
    injected noise to find low-energy constraint-satisfying states.

Spec: REQ-OPT-1771, SCENARIO-OPT-1771
"""

from typing import Callable, Tuple, List, Any
import jax
import jax.numpy as jnp

class LatentOptimizer:
    """Optimizes continuous latent variables using Langevin dynamics.
    
    Spec: REQ-OPT-1771-1, REQ-OPT-1771-2
    """
    def __init__(
        self,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
        max_steps: int = 100
    ):
        """Initialize the latent optimizer.
        
        Args:
            step_size: The step size (alpha) for the gradient descent.
            noise_scale: The scale factor for the injected Gaussian noise.
            max_steps: Maximum number of Langevin dynamics steps.
        """
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.max_steps = max_steps

    def optimize(
        self,
        z_init: jnp.ndarray,
        energy_fn: Callable[[jnp.ndarray], jnp.ndarray],
        key: jax.Array
    ) -> Tuple[jnp.ndarray, List[float]]:
        """Perform Langevin dynamics optimization on the latent variable.
        
        Args:
            z_init: Initial continuous latent state.
            energy_fn: Callable that takes a latent state and returns a scalar energy.
            key: JAX PRNG key for noise generation.
            
        Returns:
            A tuple of the optimized latent state and a list of historical energy values.
        """
        z = z_init
        grad_fn = jax.value_and_grad(energy_fn)
        energies = []

        for _ in range(self.max_steps):
            key, subkey = jax.random.split(key)
            energy_val, grad = grad_fn(z)
            energies.append(float(energy_val))
            
            # Langevin update: z_{t+1} = z_t - (alpha / 2) * grad + sqrt(alpha) * noise_scale * epsilon
            noise = jax.random.normal(subkey, shape=z.shape)
            z = z - 0.5 * self.step_size * grad + jnp.sqrt(self.step_size) * self.noise_scale * noise

        # Record the final energy as well
        final_energy = float(energy_fn(z))
        energies.append(final_energy)
        
        return z, energies
