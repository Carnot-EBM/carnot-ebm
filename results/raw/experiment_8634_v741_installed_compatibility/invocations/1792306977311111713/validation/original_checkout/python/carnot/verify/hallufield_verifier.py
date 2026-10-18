"""HalluField Tier 0m synthetic-logit hallucination detector.

This module implements the CPU approximation of the field-theoretic token-path energy.
"""
from __future__ import annotations

import numpy as np


def compute_hallufield_score(logprobs: list[float | None], temp_grid: list[float] | None = None) -> tuple[float, list[float]]:
    """Compute the HalluField score from token logprobs.
    
    Args:
        logprobs: A list of token log probabilities.
        temp_grid: Temperature grid to evaluate the energy variation.
        
    Returns:
        tuple containing the HalluField score and the temperature grid used.
    """
    if temp_grid is None:
        temp_grid = [0.5, 0.8, 1.0, 1.2, 2.0]
        
    valid_logprobs = [float(x) for x in logprobs if x is not None]
    if not valid_logprobs:
        return 0.0, temp_grid
        
    arr = np.asarray(valid_logprobs, dtype=np.float64)
    energies = []
    entropies = []
    
    for T in temp_grid:
        scaled = arr / T
        energy = -np.mean(scaled)
        
        # Softmax over the path
        shifted = scaled - np.max(scaled)
        exp_scaled = np.exp(shifted)
        p = exp_scaled / np.sum(exp_scaled)
        
        entropy = -np.sum(p * np.log(p + 1e-12))
        
        energies.append(energy)
        entropies.append(entropy)
        
    # Standard deviation over temperatures
    energy_var = float(np.std(energies))
    entropy_var = float(np.std(entropies))
    
    return float(energy_var * entropy_var), temp_grid


class HalluFieldVerifier:
    def __init__(self, temp_grid: list[float] | None = None) -> None:
        self.temp_grid = temp_grid or [0.5, 0.8, 1.0, 1.2, 2.0]
        
    def score(self, logprobs: list[float | None]) -> float:
        val, _ = compute_hallufield_score(logprobs, self.temp_grid)
        return val
