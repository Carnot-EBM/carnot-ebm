import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carnot.samplers.backend import SamplerBackend, TsuBackend
from carnot.samplers.dtm_sampler import DtmBackend


def test_dtm_backend_satisfies_protocol():
    """Verify DtmBackend satisfies the SamplerBackend protocol."""
    # Instantiating the backend
    dtm = DtmBackend()
    
    # Check that required methods and properties are present
    assert hasattr(dtm, "backend_name")
    assert hasattr(dtm, "minimize_energy")
    assert hasattr(dtm, "sample")
    
    # Check if it formally passes the runtime checkable protocol
    assert isinstance(dtm, SamplerBackend)
    
    assert dtm.backend_name == "dtm"


def test_dtm_vs_tsu_api_parity():
    """Verify DtmBackend returns shapes identical to TsuBackend."""
    n_spins = 4
    n_samples = 3
    n_steps = 10
    beta = 1.0

    # Create random biases and symmetric couplings
    rng = np.random.default_rng(42)
    biases = rng.normal(size=n_spins)
    couplings = rng.normal(size=(n_spins, n_spins))
    couplings = (couplings + couplings.T) / 2
    np.fill_diagonal(couplings, 0)

    dtm = DtmBackend(seed=42)
    tsu = TsuBackend(seed=42)

    # Test minimize_energy parity
    dtm_min = dtm.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    tsu_min = tsu.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    
    assert isinstance(dtm_min, np.ndarray)
    assert dtm_min.shape == tsu_min.shape
    assert dtm_min.shape == (n_samples, n_spins)
    assert dtm_min.dtype == tsu_min.dtype
    assert dtm_min.dtype == bool

    # Test sample parity
    config = {"n_steps": n_steps, "beta": beta}
    dtm_sample = dtm.sample(biases, couplings, n_samples, config)
    tsu_sample = tsu.sample(biases, couplings, n_samples, config)

    assert isinstance(dtm_sample, np.ndarray)
    assert dtm_sample.shape == tsu_sample.shape
    assert dtm_sample.shape == (n_samples, n_spins)
    assert dtm_sample.dtype == tsu_sample.dtype
    assert dtm_sample.dtype == bool
