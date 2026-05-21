import numpy as np
from scipy.special import comb
import math

def exact_finite_N_mean(N, beta):
    # E(x) = -0.5 * (J/N) * sum_ij x_i x_j
    # For J=1: E(x) = -0.5 * (1/N) * (sum x_i)^2
    Z = 0.0
    expected_m = 0.0
    log_ws = []
    for n in range(N + 1):
        m = (2 * n - N) / N
        E = -0.5 * N * (m ** 2)
        log_w = math.log(math.comb(N, n)) - beta * E
        log_ws.append(log_w)
    
    max_log_w = max(log_ws)
    weights = [math.exp(lw - max_log_w) for lw in log_ws]
    Z = sum(weights)
    
    for n in range(N + 1):
        m = (2 * n - N) / N
        P = weights[n] / Z
        expected_m += abs(m) * P
        
    return expected_m

N=128
beta=1.5
m_exact = exact_finite_N_mean(N, beta)
print(f"Exact m (Ising): {m_exact}")
