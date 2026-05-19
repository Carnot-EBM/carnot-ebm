import pytest
import jax.numpy as jnp
from carnot.fr11.tier3_jepa import FR11ExtendedJEPA, EMBED_DIM

def test_fr11_extended_jepa_init():
    """Verify that the extended JEPA predictor initializes correctly."""
    predictor = FR11ExtendedJEPA(seed=42)
    assert predictor.input_dim == 258
    assert len(predictor.domains) == 3

def test_fr11_extended_jepa_predict():
    """Verify that prediction works on the extended feature vector."""
    predictor = FR11ExtendedJEPA(seed=42)
    # Dummy embedding vector of size EMBED_DIM (258)
    dummy_x = jnp.zeros(EMBED_DIM)
    
    probs = predictor.predict(dummy_x)
    assert set(probs.keys()) == set(predictor.domains)
    for v in probs.values():
        assert 0.0 <= v <= 1.0

def test_fr11_extended_jepa_energy():
    """Verify that energy computation works."""
    predictor = FR11ExtendedJEPA(seed=42)
    dummy_x = jnp.ones(EMBED_DIM)
    
    energy = predictor.energy(dummy_x)
    assert 0.0 <= float(energy) <= 1.0
