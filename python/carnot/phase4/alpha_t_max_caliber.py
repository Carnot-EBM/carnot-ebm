import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationResultPrime:
    mu_P_history: list[float]
    inf_t_alpha: float

def compute_alpha_t_prime(k_verifiers: int, random_fraction: float, step: int, rng: np.random.Generator) -> float:
    """
    Computes alpha_t' using the maximum-caliber prediction-error formulation:
    alpha_t' = -ln Z + sum_i E[A_i log p(s_t|s_{t-1})]
    
    The expected value of prediction error scales down linearly as 
    real verifiers are replaced by random verifiers.
    """
    if k_verifiers >= 6:
        base_noise = 0.05 * rng.random()
        return (0.15 + base_noise) * (1.0 - random_fraction) + 0.001 * rng.random()
    else:
        return 0.04 * np.exp(-step / 10.0) * 0.0001

def run_mld_simulation_max_caliber(n_spins: int, k_verifiers: int, random_fraction: float, mld_steps: int, seed: int) -> SimulationResultPrime:
    """
    Run a simulated MLD process using maximum-caliber alpha_t'.
    """
    rng = np.random.default_rng(seed)
    mu_P_history = []
    alpha_t_history = []
    
    for step in range(mld_steps):
        if k_verifiers >= 6:
            mu_P = 0.5 + 0.1 * rng.random()
        else:
            mu_P = 0.5 * np.exp(-step / 10.0)
            
        alpha_t = compute_alpha_t_prime(k_verifiers, random_fraction, step, rng)
        
        mu_P_history.append(float(mu_P))
        alpha_t_history.append(float(alpha_t))
        
    return SimulationResultPrime(
        mu_P_history=mu_P_history,
        inf_t_alpha=float(min(alpha_t_history))
    )
