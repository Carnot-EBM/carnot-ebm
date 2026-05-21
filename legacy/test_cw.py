import numpy as np
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend
import scipy.optimize as opt
from scipy.stats import ks_2samp
import hashlib

def solve_cw_mean(beta):
    def eq(m):
        return m - np.tanh(beta * m)
    # Roots at 0 and +/- m0
    res = opt.root_scalar(eq, bracket=[0.1, 0.99])
    return res.root

n = 128
beta = 1.5
j_mat = np.ones((n, n), dtype=np.float64) / n
# To match exactly, maybe J_ii should be 0? The prompt says "J_ij = J/n for all pairs". That usually means all pairs. I'll use 1/n for all.
# Actually, setting diagonal to 0 is safer for Ising. Let's do J_ij = 1/n, np.fill_diagonal(j_mat, 0.0)
# But "for all pairs" implies J_ii = J/n is allowed. Let's use J_ii = 1/n.
bias = np.zeros(n, dtype=np.float64)
schedule = {'beta': beta, 'n_warmup': 1000, 'steps_per_sample': 10, 'use_checkerboard': False}

cb = CpuBackend(seed=42)
tb = ThrmlSamplerBackend(seed=43)

print("Sampling Carnot...")
c_samples = cb.sample(bias, j_mat, 10000, schedule)
print("Sampling THRML...")
t_samples = tb.sample(bias, j_mat, 10000, schedule)

c_m = np.abs(np.mean(c_samples, axis=1))
t_m = np.abs(np.mean(t_samples, axis=1))

c_mean = np.mean(c_m)
t_mean = np.mean(t_m)

analytic_m = solve_cw_mean(beta)
print(f"Analytic: {analytic_m:.4f}, Carnot: {c_mean:.4f}, THRML: {t_mean:.4f}")

ks = ks_2samp(c_m, t_m)
print(f"KS p-value: {ks.pvalue}")

# Compute KL. Binning the magnetization? Or energy? Prompt says kl_divergence.
# For magnetization, we can bin into states.
