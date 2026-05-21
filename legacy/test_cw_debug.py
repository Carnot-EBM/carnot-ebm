import numpy as np
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def get_samples():
    n = 128
    beta = 1.5
    j_mat = np.ones((n, n), dtype=np.float64) / n
    np.fill_diagonal(j_mat, 0.0) # usually J_ii = 0
    bias = np.zeros(n, dtype=np.float64)
    schedule = {'beta': beta, 'n_warmup': 2000, 'steps_per_sample': 10, 'use_checkerboard': False}

    cb = CpuBackend(seed=42)
    c_samples = cb.sample(bias, j_mat, 100, schedule)
    
    print("Carnot states sample 0:", c_samples[0][:10])
    
    # Calculate energy
    def E(x):
        return -0.5 * x @ j_mat @ x
    
    energies = [E(x) for x in c_samples]
    print("Mean energy:", np.mean(energies))
    
    m = np.abs(np.mean(c_samples, axis=1))
    print("Mean abs magnetization:", np.mean(m))

get_samples()