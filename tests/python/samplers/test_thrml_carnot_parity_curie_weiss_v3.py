"""Tests for THRML/Carnot parity Curie-Weiss v3.

Spec: REQ-SAMPLE-051
"""

import numpy as np
from carnot.samplers.thrml_carnot_parity_curie_weiss_v3 import get_analytic_mean, compute_kl, run_parity

def test_get_analytic_mean():
    """Verify that get_analytic_mean is correct for beta * J <= 1 and beta * J > 1.
    
    Spec: REQ-SAMPLE-051
    """
    assert np.isclose(get_analytic_mean(0.5, 1.0), 0.0)
    # For beta=1.2, J=1.0, mean is ~0.658
    assert np.isclose(get_analytic_mean(1.2, 1.0), 0.6585, atol=1e-3)

def test_compute_kl():
    """Verify KL divergence of identical distributions is zero.
    
    Spec: REQ-SAMPLE-051
    """
    p = np.array([1, 2, 1])
    kl = compute_kl(p, p)
    assert np.isclose(kl, 0.0)

def test_run_parity():
    """Verify run_parity produces the correct schema and types.
    
    We test with N=8, n_samples=10 to keep it fast for CI.
    
    Spec: REQ-SAMPLE-051, SCENARIO-SAMPLE-079
    """
    result = run_parity(N=8, beta=0.5, n_samples=10, seed_carnot=1, seed_thrml=2)
    
    assert result["schema"] == "carnot.thrml_parity_curie_weiss.v3"
    assert result["n_spins"] == 8
    assert result["n_samples"] == 10
    assert "analytic_mean" in result
    assert "analytic_energy" in result
    assert "empirical_mean_carnot" in result
    assert "empirical_mean_thrml" in result
    assert "ks_p_value" in result
    assert "kl_divergence" in result
    assert "acceptance_gate_passed" in result
    assert "honest_verdict" in result
    # We don't check pass/fail verdict here because n=8, n_samples=10 will have too much variance

