"""Tests for Denoising Thermodynamic Model (DTM) backend.

Spec: REQ-SAMPLE-038
"""

import numpy as np

from carnot.samplers.dtm import DtmBackend


def test_dtm_backend_name():
    """Verify backend name.
    
    Spec reference: REQ-SAMPLE-038
    """
    backend = DtmBackend()
    assert backend.backend_name == "dtm"


def test_dtm_sample_shape():
    """Verify sample shape and bounds.
    
    Spec reference: REQ-SAMPLE-038
    """
    backend = DtmBackend(seed=42)
    n_spins = 10
    biases = np.zeros(n_spins)
    couplings = np.zeros((n_spins, n_spins))
    n_samples = 5
    
    samples = backend.sample(
        biases, 
        couplings, 
        n_samples, 
        {"beta": 1.0, "steps": 10, "anneal": False}
    )
    
    assert samples.shape == (n_samples, n_spins)
    assert samples.dtype == bool


def test_dtm_minimize_energy():
    """Verify minimize energy API.
    
    Spec reference: REQ-SAMPLE-038
    """
    backend = DtmBackend(seed=123)
    n_spins = 4
    biases = np.array([1.0, -1.0, 1.0, -1.0])
    couplings = np.zeros((n_spins, n_spins))
    
    samples = backend.minimize_energy(
        biases, couplings, n_samples=2, n_steps=10, beta=2.0
    )
    
    assert samples.shape == (2, n_spins)
    assert samples.dtype == bool
