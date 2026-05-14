"""Tests for THRML/Carnot parity Curie-Weiss v3.

Spec: REQ-SAMPLE-051
"""

import numpy as np
from carnot.samplers.thrml_carnot_parity_curie_weiss_v3 import exact_bool_mean, compute_kl, run_parity

def test_exact_bool_mean():
    """Verify that exact_bool_mean is correct for N=2, beta=0.
    
    Spec: REQ-SAMPLE-051
    """
    m, e = exact_bool_mean(2, 0.0)
    assert np.isclose(m, 0.5)
    assert np.isclose(e, -0.75)

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
    
    assert result["schema"] == "carnot.thrml_parity_curie_weiss.v2"
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
    assert result["honest_verdict"].startswith("complete:")
