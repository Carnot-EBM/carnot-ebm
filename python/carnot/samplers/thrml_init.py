"""THRML hookup for energy initialization.

Spec: REQ-SAMPLE-041
"""

import numpy as np
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def thrml_energy_init(
    biases: np.ndarray,
    couplings: np.ndarray,
    n_samples: int,
    n_steps: int = 10,
    beta: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Hybrid digital-thermodynamic initialization.
    
    Uses ThrmlSamplerBackend in simulator mode (when CARNOT_TSU_DEVICE is unset)
    to perform energy minimization to generate an initial state.
    """
    backend = ThrmlSamplerBackend(seed=seed)
    offsets = backend.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    return offsets
