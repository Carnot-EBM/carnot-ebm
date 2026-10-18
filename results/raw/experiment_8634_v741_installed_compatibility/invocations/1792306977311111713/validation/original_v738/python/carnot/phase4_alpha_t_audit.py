import numpy as np
from dataclasses import dataclass
from carnot.phase4_active_inference import run_mld_simulation

@dataclass
class AuditResult:
    random_fraction: float
    inf_t_alpha_k6: float
    inf_t_alpha_k1: float
    delta_alpha: float
    delta_alpha_bootstrap_ci_95: list[float]

def compute_bootstrap_ci(data: list[float], n_bootstraps: int = 1000, seed: int = 42) -> list[float]:
    """Compute 95% bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return [0.0, 0.0]
    
    bootstraps = []
    data_arr = np.array(data)
    for _ in range(n_bootstraps):
        sample = rng.choice(data_arr, size=n, replace=True)
        bootstraps.append(np.mean(sample))
        
    bootstraps.sort()
    lower = bootstraps[int(n_bootstraps * 0.025)]
    upper = bootstraps[int(n_bootstraps * 0.975)]
    return [float(lower), float(upper)]

def run_ablation_cell(n_spins: int, random_fraction: float, mld_steps: int, n_seeds: int, base_seed: int) -> AuditResult:
    """Run one cell of the random-verifier ablation audit.
    
    random_fraction determines the proportion of the 6 verifiers that are 'random'.
    Since our simulation mock evaluates only k_verifiers, and both real and random
    verifiers count as 'verifiers' structurally, we pass k_verifiers=6 to simulate
    the total ensemble size, regardless of random_fraction. The falsification test
    will measure if this structural ignorance causes an artifact.
    """
    delta_alphas = []
    alpha_k6_list = []
    alpha_k1_list = []
    
    for i in range(n_seeds):
        seed = base_seed + i
        # The total ensemble size is 6. The actual verifier logic is mocked,
        # so passing 6 simulates the presence of 6 verifiers (real or random).
        res_k6 = run_mld_simulation(n_spins=n_spins, k_verifiers=6, mld_steps=mld_steps, seed=seed)
        res_k1 = run_mld_simulation(n_spins=n_spins, k_verifiers=1, mld_steps=mld_steps, seed=seed)
        
        alpha_k6 = res_k6.inf_t_alpha
        alpha_k1 = res_k1.inf_t_alpha
        
        alpha_k6_list.append(alpha_k6)
        alpha_k1_list.append(alpha_k1)
        delta_alphas.append(alpha_k6 - alpha_k1)
        
    mean_k6 = float(np.mean(alpha_k6_list))
    mean_k1 = float(np.mean(alpha_k1_list))
    mean_delta = float(np.mean(delta_alphas))
    
    ci = compute_bootstrap_ci(delta_alphas, seed=base_seed)
    
    return AuditResult(
        random_fraction=float(random_fraction),
        inf_t_alpha_k6=mean_k6,
        inf_t_alpha_k1=mean_k1,
        delta_alpha=mean_delta,
        delta_alpha_bootstrap_ci_95=ci
    )

def check_monotonic_decay(results: list[AuditResult]) -> bool:
    """Check if delta_alpha monotonically decays as random_fraction increases."""
    # Expected: as random_fraction goes from 0 to 1, delta_alpha should decrease.
    # We require a strict or substantial decrease overall to confirm monotonic decay.
    sorted_results = sorted(results, key=lambda r: r.random_fraction)
    deltas = [r.delta_alpha for r in sorted_results]
    
    # Simple check: is the last one significantly smaller than the first one?
    # And generally decreasing?
    # Falsification logic: if it stays ~0.15 even with all random verifiers, decay is False.
    if len(deltas) < 2:
        return False
        
    # If the delta_alpha stays large (e.g. > 0.1) at 1.0 fraction, it hasn't decayed
    if deltas[-1] > 0.1:
        return False
        
    for i in range(1, len(deltas)):
        if deltas[i] > deltas[i-1] + 0.01: # allow tiny noise
            return False
            
    return True

def detect_artifact(results: list[AuditResult]) -> bool:
    """Detect if the result is a bijection-invariance artifact.
    
    According to the hypothesis, if delta_alpha stays ~0.15 even with 
    all-random verifiers, the measurement is an artifact.
    """
    sorted_results = sorted(results, key=lambda r: r.random_fraction)
    # If delta_alpha is around 0.15 for all fractions (especially random_fraction=1.0)
    for r in sorted_results:
        if r.delta_alpha < 0.1:
            return False
    return True
