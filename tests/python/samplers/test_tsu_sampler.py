import numpy as np
import pytest

from carnot.samplers.backend import get_backend, get_sampler_backend
from carnot.samplers.tsu_sampler import TSUSampler

# Spec: REQ-SAMPLE-2059
# Spec: SCENARIO-SAMPLE-2059

def test_tsu_sampler_implements_backend():
    """Verify TSUSampler can be instantiated and exposes correct methods."""
    sampler = TSUSampler(seed=123)
    assert sampler.backend_name == "thrml_tsu"

    n_spins = 10
    biases = np.zeros(n_spins)
    couplings = np.zeros((n_spins, n_spins))

    # Test minimize_energy
    samples = sampler.minimize_energy(biases, couplings, n_samples=5, n_steps=100, beta=1.0)
    assert samples.shape == (5, n_spins)
    assert samples.dtype == bool

    # Test sample
    samples_fixed = sampler.sample(biases, couplings, n_samples=3, config={"beta": 1.0})
    assert samples_fixed.shape == (3, n_spins)
    assert samples_fixed.dtype == bool

def test_tsu_sampler_registry():
    """Verify TSUSampler is wired into the Carnot sampler registry."""
    # Test get_sampler_backend (experiment API)
    sampler1 = get_sampler_backend("thrml_tsu")
    assert isinstance(sampler1, TSUSampler)
    
    # Test get_backend (protocol API)
    sampler2 = get_backend("thrml_tsu")
    assert isinstance(sampler2, TSUSampler)
