import jax
import jax.numpy as jnp
from carnot.models.gibbs.hardnet import DampedLinearizationLayer

def test_damped_linearization_layer_circle_constraint():
    """
    Test projection onto a nonlinear circle constraint.
    REQ-HARDNET-001: Nonlinear Projection
    SCENARIO-HARDNET-001: Nonlinear inequality projection
    """
    layer = DampedLinearizationLayer(max_iter=50, damping=1e-3, tolerance=1e-4)
    
    def constraint_fn(x):
        return jnp.sum(x**2) - 1.0
        
    x_init = jnp.array([2.0, 2.0])
    variables = layer.init(jax.random.PRNGKey(0), x_init, constraint_fn)
    
    x_proj = layer.apply(variables, x_init, constraint_fn)
    
    # Check that constraint is satisfied
    assert constraint_fn(x_proj) <= 1e-4
    
    # Check that it didn't move if it was already inside
    x_inside = jnp.array([0.5, 0.5])
    x_proj_inside = layer.apply(variables, x_inside, constraint_fn)
    assert jnp.allclose(x_inside, x_proj_inside, atol=1e-5)

def test_damped_linearization_layer_polynomial_inequalities():
    """
    Test projection onto multiple polynomial inequalities.
    REQ-HARDNET-001: Nonlinear Projection
    SCENARIO-HARDNET-001: Nonlinear inequality projection
    """
    layer = DampedLinearizationLayer(max_iter=100, damping=1e-3, tolerance=1e-4)
    
    def constraint_fn(x):
        return jnp.array([
            x[0]**3 - x[1],
            x[0] + x[1] - 2.0
        ])
    
    x_init = jnp.array([2.0, 1.0])
    variables = layer.init(jax.random.PRNGKey(0), x_init, constraint_fn)
    
    x_proj = layer.apply(variables, x_init, constraint_fn)
    
    violations = constraint_fn(x_proj)
    assert jnp.all(violations <= 1e-3)
