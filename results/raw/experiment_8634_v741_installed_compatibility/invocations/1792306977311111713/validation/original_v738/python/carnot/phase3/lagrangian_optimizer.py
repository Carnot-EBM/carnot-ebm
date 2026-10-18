import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def global_lagrangian_energy(
    x: jnp.ndarray,
    potentials_fn: Callable[[jnp.ndarray], jnp.ndarray],
    multipliers: jnp.ndarray,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    penalty_weight: float = 1e4
) -> jnp.ndarray:
    """
    Computes the Lagrangian energy function that sums local symbolic constraint
    potentials and enforces hard bounds as high-energy penalties.
    
    Args:
        x: Continuous latent state.
        potentials_fn: Function returning an array of constraint violations/potentials (>= 0).
        multipliers: Lagrangian multipliers for each constraint.
        lower_bound: Lower bound for the latent space.
        upper_bound: Upper bound for the latent space.
        penalty_weight: Weight for the hard bound penalties.
        
    Returns:
        Scalar energy value.
    """
    g = potentials_fn(x)
    
    # Lagrangian sum: multiplier * constraint
    constraint_energy = jnp.sum(multipliers * g)
    
    # Hard bounds as high-energy penalties (quadratic)
    lower_violation = jnp.maximum(0.0, lower_bound - x)
    upper_violation = jnp.maximum(0.0, x - upper_bound)
    bound_penalty = penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2)
    
    return constraint_energy + bound_penalty

class LagrangianOptimizer:
    def __init__(
        self, 
        potentials_fn: Callable[[jnp.ndarray], jnp.ndarray], 
        learning_rate: float = 0.01, 
        penalty_weight: float = 1e4,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0
    ):
        """
        JAX-based continuous space optimizer using a Lagrangian formulation.
        Enforces hard bounds as high-energy penalties.
        """
        self.potentials_fn = potentials_fn
        self.learning_rate = learning_rate
        self.penalty_weight = penalty_weight
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        def loss_fn(x: jnp.ndarray, multipliers: jnp.ndarray) -> jnp.ndarray:
            return global_lagrangian_energy(
                x, 
                self.potentials_fn, 
                multipliers, 
                self.lower_bound, 
                self.upper_bound, 
                self.penalty_weight
            )
            
        self.grad_x_fn = jax.grad(loss_fn, argnums=0)
        self.grad_m_fn = jax.grad(loss_fn, argnums=1)
        
        @jax.jit
        def step_fn(x: jnp.ndarray, m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            grad_x = self.grad_x_fn(x, m)
            grad_m = self.grad_m_fn(x, m)
            x_new = x - self.learning_rate * grad_x
            m_new = jnp.maximum(0.0, m + self.learning_rate * grad_m)
            return x_new, m_new
            
        self._step_fn = step_fn

        @jax.jit
        def optimize_fn(x_init: jnp.ndarray, m_init: jnp.ndarray, steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
            def body_fun(i, val):
                return self._step_fn(val[0], val[1])
            return jax.lax.fori_loop(0, steps, body_fun, (x_init, m_init))
            
        self._optimize_fn = optimize_fn
        
    def step(self, x: jnp.ndarray, multipliers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs one step of primal-dual gradient update."""
        return self._step_fn(x, multipliers)

    def optimize(
        self, 
        x_init: jnp.ndarray, 
        multipliers_init: jnp.ndarray, 
        steps: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Optimizes the state and multipliers for a given number of steps."""
        return self._optimize_fn(x_init, multipliers_init, steps)
