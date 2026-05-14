import jax
import jax.numpy as jnp

def damped_local_linearization(x, g_fn, damping=0.1, max_iter=50, eps=1e-6):
    """
    HardNet++ prototype for nonlinear inequalities.
    Project x to satisfy nonlinear inequality g_fn(x) <= 0 using damped local linearization.
    """
    def body_fn(val):
        i, current_x = val
        g_val = g_fn(current_x)
        grad_g = jax.grad(g_fn)(current_x)
        
        # Projection step for inequality g(x) <= 0
        violation = jnp.maximum(0.0, g_val)
        grad_norm_sq = jnp.sum(jnp.square(grad_g))
        step = (violation / (grad_norm_sq + eps)) * grad_g
        
        # Damped update
        next_x = current_x - damping * step
        return i + 1, next_x

    def cond_fn(val):
        i, current_x = val
        g_val = g_fn(current_x)
        return jnp.logical_and(i < max_iter, g_val > 1e-4)

    _, final_x = jax.lax.while_loop(cond_fn, body_fn, (0, x))
    return final_x
