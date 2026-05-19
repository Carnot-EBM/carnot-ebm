import numpy as np
import random
from carnot.pipeline.kinetic_fst_sampler import KineticLangevinFSTSampler

def dummy_energy(spin: np.ndarray) -> float:
    return float(np.sum(spin))

def dummy_grad(spin: np.ndarray) -> np.ndarray:
    return np.ones_like(spin)

def test_kinetic_fst_sampler_basic():
    sampler = KineticLangevinFSTSampler(
        energy_fn=dummy_energy,
        grad_energy_fn=dummy_grad,
        n_spins=4,
        gamma=1.0,
        kT=1.0,
        dt=0.01,
        seed=42
    )
    tokens = ["hello", "world", "test", "token"]
    accepted = sampler.filter_tokens("prompt", tokens)
    # Just assert it runs and returns a subset (or all)
    assert isinstance(accepted, list)
    assert len(accepted) <= len(tokens)
