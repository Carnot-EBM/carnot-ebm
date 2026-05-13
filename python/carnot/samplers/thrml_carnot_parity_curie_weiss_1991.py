"""Experiment 1991: Corrected Curie-Weiss Parity at n=128.

Spec traces: REQ-SAMPLE-1991
"""

import os
import json
from typing import Any, Dict
import numpy as np

from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend


def exact_curie_weiss_energy(n_spins: int, beta: float, J_val: float) -> float:
    """Calculate analytic ground truth mean energy for Curie-Weiss model.
    
    The ParallelIsingSampler draws from P(s) ~ exp(-beta E_sampler(s)) where s in {0, 1}^N.
    E_sampler(s) = - s^T J s - 2 b^T s.
    For J_ij = J_val / n_spins, E_sampler(k) = - (J_val / n_spins) * (k^2 - k) where k is number of 1s.
    """
    log_terms = []
    log_fac = np.array([np.sum(np.log(np.arange(1, i + 1))) if i > 0 else 0.0 for i in range(n_spins + 1)])
    
    for k in range(n_spins + 1):
        log_choose = log_fac[n_spins] - log_fac[k] - log_fac[n_spins - k]
        E_k = - (J_val / n_spins) * (k ** 2 - k)
        log_terms.append(log_choose - beta * E_k)
        
    log_terms = np.array(log_terms)
    max_log = np.max(log_terms)
    Z_scaled = np.sum(np.exp(log_terms - max_log))
    
    E_terms = np.array([- (J_val / n_spins) * (k ** 2 - k) for k in range(n_spins + 1)])
    mean_energy = np.sum(E_terms * np.exp(log_terms - max_log)) / Z_scaled
    
    return float(mean_energy)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence between two empirical distributions."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def run_experiment(seed: int = 1991, n_samples: int = 10000) -> Dict[str, Any]:
    os.environ["JAX_PLATFORMS"] = "cpu"
    n_spins = 128
    np.random.seed(seed)
    
    J_val = 1.0
    J = np.ones((n_spins, n_spins)) * (J_val / n_spins)
    np.fill_diagonal(J, 0.0)
    b = np.zeros(n_spins)
    
    schedule = {
        "beta": 1.0,
        "n_warmup": 500,
        "steps_per_sample": 10,
        "use_checkerboard": True,
    }
    
    carnot_backend = CpuBackend(seed)
    carnot_samples = np.asarray(carnot_backend.sample(b, J, n_samples, schedule))
    
    thrml_backend = ThrmlSamplerBackend(seed + 1)
    thrml_samples = np.asarray(thrml_backend.sample(b, J, n_samples, schedule))
    
    def ising_energy(j_mat: np.ndarray, bias: np.ndarray, state: np.ndarray) -> float:
        # Boolean energy sampled by ParallelIsingSampler
        return float(- state.T @ j_mat @ state - 2 * bias.T @ state)
        
    carnot_states = carnot_samples.astype(float)
    thrml_states = thrml_samples.astype(float)
    
    carnot_energies = np.array([ising_energy(J, b, s) for s in carnot_states])
    thrml_energies = np.array([ising_energy(J, b, s) for s in thrml_states])
    
    analytic_mean_energy = exact_curie_weiss_energy(n_spins, schedule["beta"], J_val)
    carnot_mean_energy = float(np.mean(carnot_energies))
    thrml_mean_energy = float(np.mean(thrml_energies))
    
    min_e = min(np.min(carnot_energies), np.min(thrml_energies))
    max_e = max(np.max(carnot_energies), np.max(thrml_energies))
    if max_e == min_e:
        bins = np.array([min_e - 0.5, min_e + 0.5])
    else:
        bins = np.linspace(min_e, max_e, 50)
        
    carnot_hist, _ = np.histogram(carnot_energies, bins=bins, density=True)
    thrml_hist, _ = np.histogram(thrml_energies, bins=bins, density=True)
    
    kl_div = kl_divergence(carnot_hist, thrml_hist)
    
    carnot_delta = abs(carnot_mean_energy - analytic_mean_energy)
    thrml_delta = abs(thrml_mean_energy - analytic_mean_energy)
    
    passed = bool(
        carnot_delta < 0.5 and
        thrml_delta < 0.5 and
        kl_div < 0.05
    )
    
    result = {
        "experiment_id": "1991",
        "name": "Corrected Curie-Weiss Parity",
        "schema": "carnot.curie_weiss_parity_sweep.v1",
        "n_spins": n_spins,
        "n_samples": n_samples,
        "hardware_execution_claim": False,
        "analytic_mean_energy": analytic_mean_energy,
        "carnot_mean_energy": carnot_mean_energy,
        "thrml_mean_energy": thrml_mean_energy,
        "carnot_analytic_delta": carnot_delta,
        "thrml_analytic_delta": thrml_delta,
        "kl_divergence": float(kl_div),
        "acceptance_gate_passed": passed,
        "honest_verdict": f"success: kl_{kl_div:.4f}_delta_{carnot_delta:.4f}" if passed else f"failed: kl_{kl_div:.4f}_delta_{carnot_delta:.4f}"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1991_curie_weiss_parity_correction.json", "w") as f:
        json.dump(result, f, indent=2)
        
    return result


if __name__ == "__main__":
    run_experiment()
