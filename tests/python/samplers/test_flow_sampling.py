import jax
import jax.numpy as jnp

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.samplers.flow_sampling import FlowSampler

def test_flow_sampling_prototype():
    """Spec: REQ-SAMPLE-1960, SCENARIO-SAMPLE-1960"""
    key = jax.random.PRNGKey(0)
    # Use a small model to test quickly
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), key=key)
    
    sampler = FlowSampler(n_steps=10, dt=0.01)
    
    x0 = jnp.zeros(2)
    # Test forward step
    x_t, noise = sampler.forward_step(x0, t=0.5, key=key)
    assert x_t.shape == (2,)
    
    # Test reverse sampling
    x_final = sampler.sample(model, shape=(2,), key=key)
    assert x_final.shape == (2,)
    assert not jnp.any(jnp.isnan(x_final))
