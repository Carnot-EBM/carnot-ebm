import numpy as np
from carnot.samplers.backend import CpuBackend
import scipy.optimize as opt

N = 128
J_val = 1.0
beta = 1.2

# CW params
J_cw = np.ones((N, N)) / N
np.fill_diagonal(J_cw, 0)
# Sampler params
J_01 = 2 * J_cw
b_01 = -J_cw.sum(axis=1)

cb = CpuBackend(seed=42)
carnot_samples = cb.sample(b_01, J_01, 1000, {'beta': beta, 'n_warmup': 1000, 'steps_per_sample': 10, 'use_checkerboard': False})
c_mean = np.mean(carnot_samples * 2 - 1)

print("Mapped mean without checkerboard:", c_mean)
