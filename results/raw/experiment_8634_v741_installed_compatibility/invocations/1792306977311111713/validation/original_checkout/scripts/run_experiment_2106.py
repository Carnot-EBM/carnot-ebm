import os
import json
import math
import numpy as np
from scipy.stats import ks_2samp
import hashlib

from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def exact_bool_mean(N, beta):
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

def compute_kl(p, q):
    # Add small epsilon to avoid div by zero
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def run_experiment():
    N = 128
    beta = 1.5
    n_samples = 10000
    
    seed_carnot = 42
    seed_thrml = 43
    
    j_mat = np.ones((N, N), dtype=np.float64) / N
    # The prompt: "J_ij = J/n for all pairs, h=0"
    bias = np.zeros(N, dtype=np.float64)
    
    schedule = {
        'beta': beta,
        'n_warmup': 2000,
        'steps_per_sample': 10,
        'use_checkerboard': False
    }

    print("Running Carnot...")
    cb = CpuBackend(seed=seed_carnot)
    carnot_samples = cb.sample(bias, j_mat, n_samples, schedule)
    
    print("Running THRML...")
    tb = ThrmlSamplerBackend(seed=seed_thrml)
    thrml_samples = tb.sample(bias, j_mat, n_samples, schedule)
    
    # Magnetization is the mean over the spins. Since output is boolean, we take mean.
    c_m = np.mean(carnot_samples, axis=1)
    t_m = np.mean(thrml_samples, axis=1)
    
    empirical_mean_carnot = float(np.mean(c_m))
    empirical_mean_thrml = float(np.mean(t_m))
    
    analytic_mean, analytic_energy = exact_bool_mean(N, beta)
    
    ks_stat, ks_p = ks_2samp(c_m, t_m)
    
    # KL Divergence. Bin the sums into [0, N]
    c_sums = np.sum(carnot_samples, axis=1)
    t_sums = np.sum(thrml_samples, axis=1)
    
    c_hist, _ = np.histogram(c_sums, bins=np.arange(N + 2))
    t_hist, _ = np.histogram(t_sums, bins=np.arange(N + 2))
    
    kl_div = compute_kl(c_hist, t_hist)
    
    # Gates
    passed = True
    if abs(empirical_mean_carnot - analytic_mean) >= 0.05: passed = False
    if abs(empirical_mean_thrml - analytic_mean) >= 0.05: passed = False
    if ks_p <= 0.05: passed = False
    
    # Generate checksum
    hasher = hashlib.sha256()
    hasher.update(c_sums.tobytes())
    hasher.update(t_sums.tobytes())
    checksum = hasher.hexdigest()
    
    artifact = {
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
    
    out_path = "results/experiment_2106_thrml_parity_v3.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print(f"Artifact written to {out_path}")
    print(f"Passed: {passed}")

if __name__ == "__main__":
    run_experiment()
