import numpy as np
from scipy.special import comb
import math

def exact_bool_mean(N, beta):
    # E(s) = -beta/N * (sum s_i)^2
    Z = 0.0
    expected_s = 0.0
    expected_e = 0.0
    log_ws = []
    
    for n in range(N + 1):
        # n is sum(s_i)
        E = - (1.0 / N) * (n ** 2)
        log_w = math.log(math.comb(N, n)) - beta * E
        log_ws.append(log_w)
        
    max_log_w = max(log_ws)
    weights = [math.exp(lw - max_log_w) for lw in log_ws]
    Z = sum(weights)
    
    for n in range(N + 1):
        s = n / N
        E = - (1.0 / N) * (n ** 2)
        P = weights[n] / Z
        expected_s += s * P
        expected_e += E * P
        
    return expected_s, expected_e

m, e = exact_bool_mean(128, 1.5)
print(f"Exact mean for boolean CW: {m}, E: {e}")
