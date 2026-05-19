import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
import experiment_2519_phase4_arm_ebm_v3

def test_check_preconditions():
    # In the current state of the codebase, this should fail and return False.
    ok, msg = experiment_2519_phase4_arm_ebm_v3.check_preconditions()
    assert ok is False
    assert msg == "blocked_ising_verifier_not_available"

def test_run_experiment():
    result = experiment_2519_phase4_arm_ebm_v3.run_experiment()
    assert result["honest_verdict"] == "blocked_ising_verifier_not_available"
    assert "n_step_pairs" in result
    assert "pearson_r" in result
    assert "p_value" in result
    assert "step_granularity_achieved" in result
    assert "phase4_validated_step_level" in result
    assert "energy_proxy_used" in result
    assert "preconditions_checked" in result
    assert "duration_s" in result
    assert "random_seed" in result

if __name__ == "__main__":
    test_check_preconditions()
    test_run_experiment()
    print("All tests passed!")
