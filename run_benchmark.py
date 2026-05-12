import time
import numpy as np
import json
from carnot.samplers.backend import get_backend
from datetime import datetime

# 1. Generate 256-variable problem
n_vars = 256
rng = np.random.default_rng(42)
biases = rng.normal(0, 1, size=n_vars)
couplings = rng.normal(0, 1, size=(n_vars, n_vars))
couplings = (couplings + couplings.T) / 2
np.fill_diagonal(couplings, 0)

# Evaluate energy
def eval_energy(samples, b, J):
    energy = -0.5 * np.einsum('ni,ij,nj->n', samples, J, samples) - np.einsum('ni,i->n', samples, b)
    return energy

# 2. Benchmark Gibbs baseline
gibbs_backend = get_backend("cpu")
start_time = time.time()
gibbs_samples = gibbs_backend.minimize_energy(biases, couplings, n_samples=100, n_steps=1000, beta=10.0)
gibbs_delay = time.time() - start_time
gibbs_energies = eval_energy(gibbs_samples, biases, couplings)
gibbs_mean_energy = float(np.mean(gibbs_energies))
gibbs_min_energy = float(np.min(gibbs_energies))

print("Gibbs min energy:", gibbs_min_energy, "mean energy:", gibbs_mean_energy, "delay:", gibbs_delay)
