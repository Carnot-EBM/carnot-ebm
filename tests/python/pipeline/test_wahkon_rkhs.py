"""Test Wahkon RKHS implementation.

Spec: REQ-KAN-1909, SCENARIO-KAN-1909
"""

import jax
import jax.numpy as jnp
from carnot.pipeline.wahkon_rkhs import WahkonRKHS

def test_wahkon_rkhs():
    """Test the initialization and forward pass of the Wahkon RKHS model."""
    model = WahkonRKHS(hidden_dim=16, out_dim=2)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 8))
    
    variables = model.init(key, x)
    y = model.apply(variables, x)
    
    assert y.shape == (4, 2)
