"""Tests for THRML hookup for energy initialization.

Spec: REQ-SAMPLE-041
"""

import numpy as np
from carnot.samplers.thrml_init import thrml_energy_init

def test_thrml_energy_init():
    biases = np.array([0.1, -0.2])
    couplings = np.array([[0.0, 0.1], [0.1, 0.0]])
    n_samples = 3
    
    offsets = thrml_energy_init(biases, couplings, n_samples, seed=42)
    
    assert offsets.shape == (n_samples, 2)
    # Simulator backend (CpuBackend) should return deterministic results for a fixed seed
    # Here we just check the shape and that it runs without error.
