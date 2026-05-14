"""Exp 2106 THRML/Carnot parity Curie-Weiss v3.

This module provides the implementation for the Curie-Weiss parity test between Carnot and THRML.

Spec traces: REQ-SAMPLE-051, SCENARIO-SAMPLE-079.
"""

import math
import numpy as np
from scipy.stats import ks_2samp
import hashlib
from typing import Tuple, Dict, Any

from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def exact_bool_mean(N: int, beta: float) -> Tuple[float, float]:
    """Compute the exact thermodynamic mean and energy for a boolean Curie-Weiss model.

    E(s) = -beta/N * (sum s_i)^2
    """
    Z = 0.0
    expected_s = 0.0
    expected_e = 0.0
    log_ws = []
    
    for n in range(N + 1):
        E = - (1.0 / N) * (n ** 2)
        log_w = math.log(math.comb(N, n)) - beta * E
        log_ws.append(log_w)
        
    max_log_w = max(log_ws)
    weights = [math.exp(lw - max_log_w) for lw in log_ws]
    Z = sum(weights)
    
    for n in range(N + 1):
        s = n / N
        E = - (1.0 / N) * (n ** 2)
        P = weights[n] / Z
        expected_s += s * P
        expected_e += E * P
        
    return expected_s, expected_e

def compute_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between two empirical distributions."""
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def run_parity(N: int = 128, beta: float = 1.5, n_samples: int = 10000, seed_carnot: int = 42, seed_thrml: int = 43) -> Dict[str, Any]:
    """Run the parity simulation and compute metrics."""
    j_mat = np.ones((N, N), dtype=np.float64) / N
    bias = np.zeros(N, dtype=np.float64)
    
    schedule = {
        'beta': beta,
        'n_warmup': 2000,
        'steps_per_sample': 10,
        'use_checkerboard': False
    }

    cb = CpuBackend(seed=seed_carnot)
    carnot_samples = cb.sample(bias, j_mat, n_samples, schedule)
    
    tb = ThrmlSamplerBackend(seed=seed_thrml)
    thrml_samples = tb.sample(bias, j_mat, n_samples, schedule)
    
    c_m = np.mean(carnot_samples, axis=1)
    t_m = np.mean(thrml_samples, axis=1)
    
    empirical_mean_carnot = float(np.mean(c_m))
    empirical_mean_thrml = float(np.mean(t_m))
    
    analytic_mean, analytic_energy = exact_bool_mean(N, beta)
    
    _, ks_p = ks_2samp(c_m, t_m)
    
    c_sums = np.sum(carnot_samples, axis=1)
    t_sums = np.sum(thrml_samples, axis=1)
    
    c_hist, _ = np.histogram(c_sums, bins=np.arange(N + 2))
    t_hist, _ = np.histogram(t_sums, bins=np.arange(N + 2))
    
    kl_div = compute_kl(c_hist, t_hist)
    
    passed = True
    if abs(empirical_mean_carnot - analytic_mean) >= 0.05: passed = False
    if abs(empirical_mean_thrml - analytic_mean) >= 0.05: passed = False
    if ks_p <= 0.05: passed = False
    
    hasher = hashlib.sha256()
    hasher.update(c_sums.tobytes())
    hasher.update(t_sums.tobytes())
    checksum = hasher.hexdigest()
    
    return {
        "schema": "carnot.thrml_parity_curie_weiss.v2",
        "n_spins": N,
        "n_samples": n_samples,
        "random_seed_carnot": seed_carnot,
        "random_seed_thrml": seed_thrml,
        "reproducibility_checksum": checksum,
        "analytic_mean": analytic_mean,
        "analytic_energy": analytic_energy,
        "empirical_mean_carnot": empirical_mean_carnot,
        "empirical_mean_thrml": empirical_mean_thrml,
        "ks_p_value": float(ks_p),
        "kl_divergence": float(kl_div),
        "acceptance_gate_passed": passed,
        "actual_agent_backend": "gemini",
        "honest_verdict": f"complete: thrml_parity_cw_v3_passed" if passed else "complete: thrml_parity_cw_v3_failed"
    }
