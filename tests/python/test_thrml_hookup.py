"""Tests for THRML hookup energy initialization.

Spec: REQ-SAMPLE-041, SCENARIO-SAMPLE-041
"""
import numpy as np
from carnot.samplers.thrml_init import thrml_energy_init

def test_thrml_energy_init():
    """Verify hybrid digital-thermodynamic initialization returns expected shape and type."""
    biases = np.array([0.1, -0.2, 0.3])
    couplings = np.array([
        [0.0, 0.1, -0.1],
        [0.1, 0.0, 0.2],
        [-0.1, 0.2, 0.0]
    ])
    n_samples = 5
    
    offsets = thrml_energy_init(
        biases=biases,
        couplings=couplings,
        n_samples=n_samples,
        n_steps=2,
        beta=1.0,
        seed=42
    )
    
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (n_samples, len(biases))
    # Basic sanity check that it runs and returns finite values
    assert not np.isnan(offsets).any()
    assert not np.isinf(offsets).any()
