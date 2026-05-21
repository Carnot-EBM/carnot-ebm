import numpy as np
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def get_samples():
    n = 128
    beta = 1.5
    j_mat = np.ones((n, n), dtype=np.float64) / n
    np.fill_diagonal(j_mat, 0.0)
    bias = np.zeros(n, dtype=np.float64)
    schedule = {'beta': beta, 'n_warmup': 2000, 'steps_per_sample': 10, 'use_checkerboard': False}

    cb = CpuBackend(seed=42)
    c_samples_bool = cb.sample(bias, j_mat, 100, schedule)
    
    tb = ThrmlSamplerBackend(seed=42)
    t_samples_bool = tb.sample(bias, j_mat, 100, schedule)
    
    print("Carnot type:", c_samples_bool.dtype, c_samples_bool.shape)
    print("THRML type:", t_samples_bool.dtype, t_samples_bool.shape)

    c_spins = c_samples_bool * 2.0 - 1.0
    t_spins = t_samples_bool * 2.0 - 1.0

    print("Carnot mean abs m:", np.mean(np.abs(np.mean(c_spins, axis=1))))
    print("THRML mean abs m:", np.mean(np.abs(np.mean(t_spins, axis=1))))

get_samples()