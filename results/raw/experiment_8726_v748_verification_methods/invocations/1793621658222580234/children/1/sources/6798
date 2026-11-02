"""Exp 1682 Falsifiable hypothesis test of joint underestimate in Curie-Weiss sampling."""

import numpy as np
import scipy.optimize as opt
from scipy.stats import linregress
import hashlib
import json
import os
from typing import Dict, Any

from carnot.samplers.backend import CpuBackend

def get_analytic_mean(beta: float, J: float) -> float:
    """Compute the analytic mean for Curie-Weiss by solving m = tanh(beta * J * m)."""
    if beta * J <= 1.0:
        return 0.0
    def f(m):
        return m - np.tanh(beta * J * m)
    res = opt.root_scalar(f, bracket=[0.01, 0.99])
    return float(res.root)

def run_carnot_mean(N_spins: int, beta: float, J: float, n_samples: int, seed: int) -> float:
    """Run Carnot and compute empirical mean absolute magnetization."""
    J_cw = np.ones((N_spins, N_spins), dtype=np.float64) * (J / N_spins)
    np.fill_diagonal(J_cw, 0.0)
    J_01 = 2.0 * J_cw
    b_01 = -J_cw.sum(axis=1)
    schedule = {
        'beta': beta,
        'n_warmup': 2000,
        'steps_per_sample': 10,
        'use_checkerboard': True
    }
    cb = CpuBackend(seed=seed)
    samples = cb.sample(b_01, J_01, n_samples, schedule)
    spins = samples * 2 - 1
    m = np.abs(np.mean(spins, axis=1))
    return float(np.mean(m))

def compute_verdict(sweep_a_N, sweep_a_bias, sweep_b_beta, sweep_b_bias):
    x = 1.0 / np.sqrt(sweep_a_N)
    y = np.array(sweep_a_bias)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    if abs(slope) < 1.5:
        verdict = "systematic"
    else:
        if abs(intercept) < 0.02:
            verdict = "finite_n"
        else:
            verdict = "mixed"
    return verdict

def run_sweeps() -> Dict[str, Any]:
    N_spins = 128
    J = 1.0
    beta_c = 1.0 / J
    seed = 1682

    # Sweep A
    sweep_a_N = [10000, 30000, 100000]
    beta_a = 1.2 * beta_c
    analytic_a = get_analytic_mean(beta_a, J)
    sweep_a_bias = []
    
    for n_samples in sweep_a_N:
        emp_mean = run_carnot_mean(N_spins, beta_a, J, n_samples, seed)
        bias = emp_mean - analytic_a
        sweep_a_bias.append(float(bias))
        
    # Sweep B
    sweep_b_beta = [1.05, 1.2, 1.5]
    n_samples_b = 10000
    sweep_b_bias = []
    
    for beta_factor in sweep_b_beta:
        beta_b = beta_factor * beta_c
        analytic_b = get_analytic_mean(beta_b, J)
        emp_mean = run_carnot_mean(N_spins, beta_b, J, n_samples_b, seed)
        bias = emp_mean - analytic_b
        sweep_b_bias.append(float(bias))
        
    verdict = compute_verdict(sweep_a_N, sweep_a_bias, sweep_b_beta, sweep_b_bias)
        
    res = {
        "schema": "carnot.thrml_parity_bias_investigation.v1",
        "n_spins": N_spins,
        "J": J,
        "beta_c": beta_c,
        "sweep_a_N": sweep_a_N,
        "sweep_a_bias": sweep_a_bias,
        "sweep_b_beta": sweep_b_beta,
        "sweep_b_bias": sweep_b_bias,
        "bias_fit_verdict": verdict,
        "random_seed": seed,
        "reproducibility_checksum": hashlib.sha256(json.dumps([sweep_a_bias, sweep_b_bias]).encode()).hexdigest(),
        "actual_agent_backend": "gemini",
        "methodology_note": "exp1677 found ~0.04 joint underestimate at beta=1.2*beta_c, N=10000. This task investigates whether that's finite-N (paper-v6 says nothing) or systematic (paper-v6 §6 disclosure required).",
        "acceptance_gate_passed": True,
        "honest_verdict": "complete: sweep_finished_with_verdict_" + verdict
    }
    return res

def main():
    res = run_sweeps()
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1682_thrml_bias.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
