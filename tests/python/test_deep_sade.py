import jax
import jax.numpy as jnp
from carnot.models.deep_sade import DeepSaDeLayer, evaluate_satisfaction_rate

def test_deepsade_constraint_layer():
    """
    Test the DeepSaDe constraint layer.
    Traces to:
    - REQ-DEEPSADE-001: Constraint Layer
    - REQ-DEEPSADE-002: Hybrid MaxSMT+SGD
    """
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 5)) * 10.0  # Large inputs to trigger violations
    
    layer = DeepSaDeLayer(features=4, lower_bound=-1.0, upper_bound=1.0)
    variables = layer.init(key, x)
    
    h_projected, violation_penalty = layer.apply(variables, x)
    
    # Check dimensions
    assert h_projected.shape == (10, 4)
    assert violation_penalty.shape == (10,)
    
    # REQ-DEEPSADE-001: Outputs must be guaranteed within constraints
    assert jnp.all(h_projected >= -1.0)
    assert jnp.all(h_projected <= 1.0)
    
    # REQ-DEEPSADE-002: Penalty should be computed for hybrid SGD logic
    # Since input is large, some penalties should be > 0
    assert jnp.any(violation_penalty > 0.0)


def test_deepsade_satisfaction_rate():
    """
    Test the constraint satisfaction metric.
    Traces to:
    - REQ-DEEPSADE-003: Satisfaction Guarantees
    """
    outputs = jnp.array([
        [0.0, 0.5],
        [-1.0, 1.0],
        [-1.5, 0.5]  # The first element here violates the lower_bound
    ])
    
    rate = evaluate_satisfaction_rate(outputs, lower=-1.0, upper=1.0)
    
    # 2 out of 3 satisfy the constraints
    assert jnp.isclose(rate, 2.0 / 3.0)
