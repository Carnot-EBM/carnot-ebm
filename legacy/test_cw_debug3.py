import numpy as np
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

def get_samples():
    n = 128
    beta = 1.5
    
    # We want to simulate CW Ising with J=1.0, h=0
    # Equivalent {0, 1} parameters:
    # J'_ij = 2J/N
    # b'_i = -2J + 2h = -2.0
    
    j_mat = np.ones((n, n), dtype=np.float64) * (2.0 / n)
    np.fill_diagonal(j_mat, 0.0) # J_ii should be 0 because we mapped x_i x_j for i!=j?
    # Wait, the derivation sum_{i,j} x_i x_j includes i=j.
    # If original CW has J_ii = 1/N, then sum_{i,j} includes i=j.
    # x_i^2 = 1. So it's just a constant.
    # In boolean, s_i^2 = s_i.
    # If we mapped x_i x_j literally for all i,j:
    # sum_{i,j} (4 s_i s_j - 4 s_i + 1) = 4 sum s_i s_j - 4 N sum s_i + N^2.
    # E(x) = - (1 / 2N) (4 sum s_i s_j - 4 N sum s_i + N^2)
    # E(s) = - (2 / N) sum s_i s_j + 2 sum s_i - 0.5 N
    # So J'_{ij} = 2 / N for ALL i,j?
    # But ParallelIsingSampler forces J diagonal to be zero!
    # "Since J is zero-diagonal, this simplifies to:"
    # If diagonal must be zero, we must move the diagonal of J' to b'.
    # J'_{ii} = 2/N. So J'_{ii} s_i^2 = J'_{ii} s_i = (2/N) s_i.
    # We add (2/N) s_i to the linear term b'.
    # So b'_i = -2.0 + 2/N.
    # Let's use J'_ij = 2/N for i!=j, J'_ii = 0.
    # b'_i = -2.0 + 2/N.
    
    j_mat = np.ones((n, n), dtype=np.float64) * (2.0 / n)
    np.fill_diagonal(j_mat, 0.0)
    bias = np.ones(n, dtype=np.float64) * (-2.0 + 2.0 / n)
    
    schedule = {'beta': beta, 'n_warmup': 2000, 'steps_per_sample': 10, 'use_checkerboard': False}

    cb = CpuBackend(seed=42)
    c_samples_bool = cb.sample(bias, j_mat, 10000, schedule)
    
    tb = ThrmlSamplerBackend(seed=43)
    t_samples_bool = tb.sample(bias, j_mat, 10000, schedule)

    c_spins = c_samples_bool * 2.0 - 1.0
    t_spins = t_samples_bool * 2.0 - 1.0

    print("Carnot mean abs m:", np.mean(np.abs(np.mean(c_spins, axis=1))))
    print("THRML mean abs m:", np.mean(np.abs(np.mean(t_spins, axis=1))))

get_samples()