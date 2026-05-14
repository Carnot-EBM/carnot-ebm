import numpy as np
from carnot.samplers.backend import CpuBackend

N = 128
J_val = 1.0
beta = 1.2

# We want the sampler to sample the {-1, 1} distribution with J_ij = 1/N.
# As derived, J_01 = 4 * J_cw, b_01 = -2 * J_cw (summed over j)
J_cw = np.ones((N, N)) / N
np.fill_diagonal(J_cw, 0)
b_cw = np.zeros(N)

# Map to {0,1} parameters
J_01 = 4 * J_cw
b_01 = -2 * J_cw.sum(axis=1) + 2 * b_cw

cb = CpuBackend(seed=42)
carnot_samples = cb.sample(b_01, J_01, 1000, {'beta': beta, 'n_warmup': 1000, 'steps_per_sample': 10})
c_mean = np.mean(carnot_samples * 2 - 1)
print("Mapped mean:", c_mean)
