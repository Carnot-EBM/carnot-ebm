"""Tests for Phase 4 Active Inference."""

from carnot.phase4_active_inference import run_mld_simulation

def test_verifier_ensemble_stability():
    """Verify SCENARIO-PHASE4-1 and REQ-PHASE4-001, REQ-PHASE4-002."""
    n_spins = 8
    mld_steps = 100
    seed = 42

    # k=6 verifiers
    result_k6 = run_mld_simulation(n_spins=n_spins, k_verifiers=6, mld_steps=mld_steps, seed=seed)
    assert result_k6.inf_t_alpha > 0.10
    assert len(result_k6.mu_P_history) == mld_steps

    # k=1 verifiers
    result_k1 = run_mld_simulation(n_spins=n_spins, k_verifiers=1, mld_steps=mld_steps, seed=seed)
    assert result_k1.inf_t_alpha < 0.05
    assert len(result_k1.mu_P_history) == mld_steps

    delta_alpha = result_k6.inf_t_alpha - result_k1.inf_t_alpha
    assert delta_alpha > 0.05
