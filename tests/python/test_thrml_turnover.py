import os
import numpy as np
import pytest

from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def test_thrml_multi_period_hardware_raises():
    os.environ["CARNOT_TSU_DEVICE"] = "dummy"
    try:
        backend = ThrmlSamplerBackend(seed=42)
        n_periods, n_spins = 2, 1
        biases = np.zeros((n_periods, n_spins))
        couplings = np.zeros((n_periods, n_spins, n_spins))
        
        with pytest.raises(NotImplementedError):
            backend.sample_multi_period(biases, couplings, 10.0, 10, {})
            
        with pytest.raises(NotImplementedError):
            backend.minimize_energy_multi_period(biases, couplings, 10.0, 10, 100, 10.0)
    finally:
        del os.environ["CARNOT_TSU_DEVICE"]

def test_thrml_multi_period_minimize_energy():
    backend = ThrmlSamplerBackend(seed=42)
    n_periods = 3
    n_spins = 2
    biases = np.zeros((n_periods, n_spins))
    couplings = np.zeros((n_periods, n_spins, n_spins))
    
    # Simple multi-period problem: cost = -b*x. So b=1 wants x=1, b=-1 wants x=0.
    biases[0, 0] = 1.0  # Wants to be 1 at t=0
    biases[2, 0] = -1.0 # Wants to be 0 at t=2
    
    samples = backend.minimize_energy_multi_period(
        biases=biases,
        couplings=couplings,
        turnover_penalty=0.0,
        n_samples=5,
        n_steps=100,
        beta=10.0
    )
    
    assert samples.shape == (5, n_periods, n_spins)
    assert np.all(samples[:, 0, 0] == True)
    assert np.all(samples[:, 2, 0] == False)

def test_thrml_multi_period_sample():
    backend = ThrmlSamplerBackend(seed=42)
    n_periods = 2
    n_spins = 1
    biases = np.zeros((n_periods, n_spins))
    couplings = np.zeros((n_periods, n_spins, n_spins))
    
    # Let's bias them to be different.
    biases[0, 0] = 1.0   # Wants 1
    biases[1, 0] = -1.0  # Wants 0
    
    # With a high turnover penalty, x_0 should equal x_1 for all samples
    samples = backend.sample_multi_period(
        biases=biases,
        couplings=couplings,
        turnover_penalty=10.0, # High penalty -> must stay same
        n_samples=10,
        config={"beta": 10.0}
    )
    
    assert samples.shape == (10, n_periods, n_spins)
    
    for i in range(10):
        assert samples[i, 0, 0] == samples[i, 1, 0]
