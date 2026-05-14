"""Tests for Phase 4 Active Inference.

Spec: REQ-PHASE4-001, REQ-PHASE4-002, SCENARIO-PHASE4-1
"""

from carnot.phase4_active_inference import run_mld_simulation

def test_mld_simulation_k6():
    """Test k=6 verifiers maintain inf_t alpha_t > 0.10. (REQ-PHASE4-002)"""
    res = run_mld_simulation(n_spins=8, k_verifiers=6, mld_steps=100, seed=42)
    assert len(res.mu_P_history) == 100
    assert res.inf_t_alpha > 0.10

def test_mld_simulation_k1():
    """Test k=1 verifier collapses inf_t alpha_t < 0.05. (REQ-PHASE4-002)"""
    res = run_mld_simulation(n_spins=8, k_verifiers=1, mld_steps=100, seed=42)
    assert res.inf_t_alpha < 0.05

def test_mld_simulation_delta():
    """Test delta_alpha > 0.05 between k=6 and k=1. (SCENARIO-PHASE4-1)"""
    res6 = run_mld_simulation(n_spins=8, k_verifiers=6, mld_steps=100, seed=42)
    res1 = run_mld_simulation(n_spins=8, k_verifiers=1, mld_steps=100, seed=42)
    delta = res6.inf_t_alpha - res1.inf_t_alpha
    assert delta > 0.05
