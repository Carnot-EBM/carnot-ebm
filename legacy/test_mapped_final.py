import numpy as np
from carnot.samplers.backend import CpuBackend

N = 128
J_val = 1.0
beta = 1.2

# The exact mapping to make P(s_i=1) equivalent to {-1,1} Curie-Weiss
J_01 = 2.0 * np.ones((N, N)) / N
np.fill_diagonal(J_01, 0)
b_01 = -1.0 * np.ones(N)

cb = CpuBackend(seed=42)
carnot_samples = cb.sample(b_01, J_01, 1000, {'beta': beta, 'n_warmup': 1000, 'steps_per_sample': 10, 'use_checkerboard': True})
c_mean = np.mean(carnot_samples * 2 - 1)

print("Proper mapped mean:", c_mean)
