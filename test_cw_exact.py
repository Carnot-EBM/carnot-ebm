import numpy as np
from scipy.special import comb
import math

def exact_finite_N_mean(N, beta):
    Z = 0.0
    expected_m = 0.0
    expected_e = 0.0
    for n in range(N + 1):
        m = (2 * n - N) / N
        E = -0.5 * N * (m ** 2)
        # log_prob = log(N choose n) - beta * E
        # Use math.comb to avoid overflow, then exp
        # actually, to avoid overflow, subtract max log prob
        log_w = math.log(math.comb(N, n)) - beta * E
        Z += log_w
        
    # two-pass to avoid overflow
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
        E = -0.5 * N * (m ** 2)
        P = weights[n] / Z
        expected_m += abs(m) * P
        expected_e += E * P
        
    return expected_m, expected_e

N=128
beta=1.5
m, e = exact_finite_N_mean(N, beta)
print(f"Exact m: {m}, Exact E: {e}")
