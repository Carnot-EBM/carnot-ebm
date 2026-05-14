import pytest
from carnot.phase4_scaling import run_scaling_experiment

def test_phase4_scaling_experiment():
    """
    Verify REQ-PHASE4-003 and SCENARIO-PHASE4-2.
    """
    n_values = [8, 16, 32]
    result = run_scaling_experiment(n_values=n_values, mld_steps=100, n_samples_per_n=100, base_seed=42)
    
    assert result["schema"] == "carnot.phase4_active_inference_scaling.v1"
    assert result["n_values"] == [8, 16, 32]
    assert result["mld_steps"] == 100
    assert result["n_samples_per_n"] == 100
    assert "n_samples_justification" in result
    assert len(result["random_seeds"]) == 3
    assert len(result["inf_t_alpha_k6"]) == 3
    assert len(result["inf_t_alpha_k1"]) == 3
    assert len(result["delta_alpha"]) == 3
    assert result["optimization_direction"] == "track minimum"
    assert "methodology_note" in result
    assert isinstance(result["acceptance_gate_passed"], bool)
    assert result["honest_verdict"].startswith("complete:")

def test_phase4_scaling_experiment_collapse():
    """
    Verify REQ-PHASE4-003 and SCENARIO-PHASE4-2 logic with mocked collapse.
    """
    from unittest.mock import patch
    from carnot.phase4_active_inference import SimulationResult
    
    def mocked_run(n_spins, k_verifiers, mld_steps, seed):
        if n_spins >= 16 and k_verifiers == 6:
            return SimulationResult(mu_P_history=[0.1], inf_t_alpha=0.04)
        elif n_spins == 8 and k_verifiers == 6:
            return SimulationResult(mu_P_history=[0.5], inf_t_alpha=0.15)
        else:
            return SimulationResult(mu_P_history=[0.01], inf_t_alpha=0.01)

    with patch('carnot.phase4_scaling.run_mld_simulation', side_effect=mocked_run):
        result = run_scaling_experiment([8, 16, 32])
        assert result["collapse_scale"] == 16
        assert result["acceptance_gate_passed"] == True
