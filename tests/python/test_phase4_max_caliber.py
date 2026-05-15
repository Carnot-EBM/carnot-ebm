import pytest
import numpy as np
from carnot.phase4.alpha_t_max_caliber import run_mld_simulation_max_caliber, compute_alpha_t_prime

def test_max_caliber_alpha_t_decay():
    """
    References REQ-PHASE4-005 and SCENARIO-PHASE4-4.
    Tests that alpha_t' monotonically decays as random_fraction increases.
    """
    res_frac0 = run_mld_simulation_max_caliber(n_spins=32, k_verifiers=6, random_fraction=0.0, mld_steps=100, seed=42)
    res_frac1 = run_mld_simulation_max_caliber(n_spins=32, k_verifiers=6, random_fraction=1.0, mld_steps=100, seed=42)
    
    assert res_frac0.inf_t_alpha > 0.10
    assert res_frac1.inf_t_alpha < 0.05
    
    # Also test collapse at k=1
    res_k1 = run_mld_simulation_max_caliber(n_spins=32, k_verifiers=1, random_fraction=0.0, mld_steps=100, seed=42)
    assert res_k1.inf_t_alpha < 0.001

def test_compute_alpha_t_prime_direct():
    """Test compute_alpha_t_prime explicitly."""
    rng = np.random.default_rng(42)
    
    # k >= 6
    val1 = compute_alpha_t_prime(k_verifiers=6, random_fraction=0.0, step=0, rng=rng)
    assert val1 > 0.10
    
    rng = np.random.default_rng(42) # reset for comparison
    val2 = compute_alpha_t_prime(k_verifiers=6, random_fraction=1.0, step=0, rng=rng)
    assert val2 < 0.05
    
    # k < 6
    val3 = compute_alpha_t_prime(k_verifiers=1, random_fraction=0.0, step=50, rng=rng)
    assert val3 < 0.01
