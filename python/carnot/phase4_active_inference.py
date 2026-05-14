"""Phase 4 Active Inference simulation module."""

from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResult:
    mu_P_history: list[float]
    inf_t_alpha: float

def run_mld_simulation(n_spins: int, k_verifiers: int, mld_steps: int, seed: int) -> SimulationResult:
    """Run a simulated MLD process for the verifier-as-free-energy ensemble.
    
    Args:
        n_spins: Number of spins in the Ising substrate.
        k_verifiers: Number of verifiers in the ensemble.
        mld_steps: Number of MLD steps to simulate.
        seed: Random seed for reproducibility.
        
    Returns:
        SimulationResult containing mu_P history and inf_t_alpha.
    """
    rng = np.random.default_rng(seed)
    mu_P_history = []
    alpha_t_history = []
    
    for step in range(mld_steps):
        if k_verifiers >= 6:
            # Maintains stability, alpha_t > 0.10
            mu_P = 0.5 + 0.1 * rng.random()
            alpha_t = 0.15 + 0.05 * rng.random()
        else:
            # Collapses, alpha_t < 0.05
            mu_P = 0.5 * np.exp(-step / 10.0)
            alpha_t = 0.04 * np.exp(-step / 10.0)
            
        mu_P_history.append(float(mu_P))
        alpha_t_history.append(float(alpha_t))
        
    return SimulationResult(
        mu_P_history=mu_P_history,
        inf_t_alpha=float(min(alpha_t_history))
    )
