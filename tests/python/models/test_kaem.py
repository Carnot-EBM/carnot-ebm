import jax.numpy as jnp
import jax.random as jrandom
from carnot.models.kaem import KAEMEnergy

def test_kaem_energy_evaluation():
    """Test evaluating KAEM 1D B-spline energy."""
    # REQ-KAN-1803
    key = jrandom.PRNGKey(42)
    model = KAEMEnergy(n_vars=5, n_knots=10, key=key)
    x = jnp.zeros(5)
    e = model.energy(x)
    assert e.shape == ()
    assert jnp.isfinite(e)

def test_kaem_inverse_transform_sampling():
    """Test inverse transform sampling allowed by univariate splines."""
    # SCENARIO-KAN-1803
    key = jrandom.PRNGKey(42)
    model = KAEMEnergy(n_vars=5, n_knots=10, key=key)
    samples = model.inverse_transform_sample(n_samples=10, key=key)
    
    assert samples.shape == (10, 5)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= -1.0)
    assert jnp.all(samples <= 1.0)
