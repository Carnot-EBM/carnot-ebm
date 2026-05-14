import jax
import jax.numpy as jnp
import numpy as np
from carnot.models.kaem_energy import SineKANLayer

# REQ-SAMPLE-015: Exact sampling using SineKAN Layer
def test_sinekan_layer_energy_and_sampling():
    key = jax.random.PRNGKey(42)
    layer = SineKANLayer(n_vars=3, n_freqs=5, key=key)
    
    x = jnp.array([0.0, 0.5, -0.5])
    e = layer.energy(x)
    assert e.shape == ()
    assert not jnp.isnan(e)
    
    samples = layer.sample_exact(10, key)
    assert samples.shape == (10, 3)
    assert jnp.all(samples >= -1.0) and jnp.all(samples <= 1.0)
    
def test_sinekan_cdf_validity():
    layer = SineKANLayer(n_vars=2, n_freqs=3)
    cdf_val = layer.marginal_cdf(0, 0.0)
    assert 0.0 <= cdf_val <= 1.0
    
    cdf_val_min = layer.marginal_cdf(0, -1.0)
    cdf_val_max = layer.marginal_cdf(0, 1.0)
    assert np.isclose(cdf_val_min, 0.0, atol=1e-2)
    assert np.isclose(cdf_val_max, 1.0, atol=1e-2)
