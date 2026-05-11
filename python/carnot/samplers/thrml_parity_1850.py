"""Experiment 1850: THRML Parity Sweep at n=128.

Spec traces: REQ-SAMPLE-1850
"""

import os
from typing import Any, Dict
import numpy as np
from scipy.stats import ks_2samp

from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence between two empirical distributions."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))

def run_parity_n128(seed: int = 1850, n_samples: int = 100) -> Dict[str, Any]:
    """Generate n=128 Ising, run both samplers, compute metrics."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    n_spins = 128
    np.random.seed(seed)
    
    J = np.random.randn(n_spins, n_spins)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    b = np.random.randn(n_spins)
    
    schedule = {
        "beta": 1.0,
        "n_warmup": 100,
        "steps_per_sample": 10,
        "use_checkerboard": True,
    }
    
    carnot_backend = CpuBackend(seed)
    carnot_samples = np.asarray(carnot_backend.sample(b, J, n_samples, schedule))
    
    thrml_backend = ThrmlSamplerBackend(seed + 1)
    thrml_samples = np.asarray(thrml_backend.sample(b, J, n_samples, schedule))
    
    def ising_energy(j_mat: np.ndarray, bias: np.ndarray, state: np.ndarray) -> float:
        return float(-0.5 * state.T @ j_mat @ state - bias.T @ state)
        
    carnot_states = np.where(carnot_samples, 1, -1)
    thrml_states = np.where(thrml_samples, 1, -1)
    
    carnot_energies = np.array([ising_energy(J, b, s) for s in carnot_states])
    thrml_energies = np.array([ising_energy(J, b, s) for s in thrml_states])
    
    mean_energy_delta_abs = abs(float(np.mean(carnot_energies)) - float(np.mean(thrml_energies)))
    
    min_e = min(np.min(carnot_energies), np.min(thrml_energies))
    max_e = max(np.max(carnot_energies), np.max(thrml_energies))
    if max_e == min_e:
        bins = np.array([min_e - 0.5, min_e + 0.5])
    else:
        bins = np.linspace(min_e, max_e, 20)
    
    carnot_hist, _ = np.histogram(carnot_energies, bins=bins, density=True)
    thrml_hist, _ = np.histogram(thrml_energies, bins=bins, density=True)
    
    kl_div = kl_divergence(carnot_hist, thrml_hist)
    
    ks_stat, ks_p_value = ks_2samp(carnot_energies, thrml_energies)
    
    passed = bool(
        mean_energy_delta_abs < 0.10 and
        kl_div < 0.05 and
        ks_p_value > 0.05
    )
    
    if passed:
        verdict = f"success: thrml_carnot_parity_n128_gate_passed_kl_{kl_div:.2f}"
    else:
        verdict = f"complete: thrml_carnot_parity_n128_gate_failed_kl_{kl_div:.2f}"
        
    return {
        "schema": "carnot.thrml_parity_sweep.v2",
        "n_spins": n_spins,
        "n_samples": n_samples,
        "mean_energy_delta_abs": float(mean_energy_delta_abs),
        "kl_divergence": float(kl_div),
        "ks_p_value": float(ks_p_value),
        "acceptance_gate_passed": passed,
        "independent_rng_paths": True,
        "honest_verdict": verdict,
    }
