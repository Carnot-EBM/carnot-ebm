import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

class DampedLinearizationLayer(nn.Module):
    """
    HardNet++ projection layer.
    Uses damped local linearizations to enforce nonlinear constraints strictly.
    """
    max_iter: int = 50
    damping: float = 1e-3
    tolerance: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray, constraint_fn: Callable) -> jnp.ndarray:
        """
        Projects `x` such that `constraint_fn(x) <= 0`.
        
        Args:
            x: Input tensor of shape (N,)
            constraint_fn: Function mapping (N,) -> (M,) or scalar returning constraint values.
                           Negative values mean the constraint is satisfied.
        
        Returns:
            Projected tensor of shape (N,) satisfying the constraints.
        """
        def cond_fun(val):
            x_i, i, max_violation = val
            return (i < self.max_iter) & (max_violation > self.tolerance)

        def body_fun(val):
            x_i, i, _ = val
            
            # Evaluate constraint value and jacobian at current state
            g_val = constraint_fn(x_i)
            g_jac = jax.jacrev(constraint_fn)(x_i)
            
            g_val = jnp.atleast_1d(g_val)
            # For jacobian, if input is (N,) and output is scalar, jacrev is (N,).
            # If output is (M,), jacrev is (M, N).
            # We want g_jac to be (M, N)
            if g_val.shape[0] == 1 and g_jac.ndim == 1:
                g_jac = g_jac[None, :]
                
            # Active violations (we only want to move if g_val > 0)
            violation = jnp.maximum(0.0, g_val)  # (M,)
            
            # Damped least squares step: 
            # dx = - J^T (J J^T + lambda I)^{-1} v
            M = g_val.shape[0]
            J_J_T = jnp.dot(g_jac, g_jac.T)
            I = jnp.eye(M)
            
            # Solve (J J^T + lambda I) w = violation
            w = jnp.linalg.solve(J_J_T + self.damping * I, violation)
            
            # dx = - J^T w
            dx = - jnp.dot(g_jac.T, w)
            
            x_next = x_i + dx
            
            next_g_val = jnp.atleast_1d(constraint_fn(x_next))
            max_violation = jnp.max(jnp.maximum(0.0, next_g_val))
            
            return (x_next, i + 1, max_violation)

        init_g = jnp.atleast_1d(constraint_fn(x))
        init_violation = jnp.max(jnp.maximum(0.0, init_g))
        init_val = (x, 0, init_violation)
        
        final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
        return final_val[0]
