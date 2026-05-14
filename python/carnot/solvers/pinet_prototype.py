import jax
import jax.numpy as jnp

def douglas_rachford_splitting(A, b, max_iter=200):
    """
    Douglas-Rachford splitting for convex constraint projection in JAX.
    Solves for x in [0, 1]^n subject to Ax = b.
    prox_f is projection onto Ax=b.
    prox_g is projection onto [0, 1]^n.
    """
    A_pinv = jnp.linalg.pinv(A)
    
    def proj_eq(x):
        return x - A_pinv @ (A @ x - b)
        
    def proj_box(x):
        return jnp.clip(x, 0.0, 1.0)
        
    def body_fn(val):
        k, x, z = val
        y = proj_box(2 * z - x)
        x_new = x + (y - z)
        z_new = proj_eq(x_new)
        return k + 1, x_new, z_new
        
    def cond_fn(val):
        k, x, z = val
        return k < max_iter
        
    x0 = jnp.zeros(A.shape[1])
    z0 = proj_eq(x0)
    
    _, _, z_final = jax.lax.while_loop(cond_fn, body_fn, (0, x0, z0))
    return z_final
