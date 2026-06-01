"""Tests for FR-11 Continuous Self Learning v5 Module."""

from carnot.fr11.continuous_self_learning_v5 import evaluate_continuous_self_learning_v5

def test_evaluate_continuous_self_learning_v5_success():
    """Test successful evaluation (REQ-LEARN-3590, SCENARIO-LEARN-3590)."""
    result = evaluate_continuous_self_learning_v5(
        deploy_arm_collapse=False,
        control_beta0_collapse=True,
        pass_rate=[0.8, 0.85],
        true_accuracy=[0.75, 0.8],
        quality_maintained=True,
        factual_verifier_calibration_improved=True
    )
    
    assert result["honest_verdict"]["value"] == "complete: fr11_conservative_default_calibrates_factual_verifier_holds_quality_maintained"
    assert result["inference_substrate"]["value"] == "verifier_ensemble_against_cached_candidates"
    assert result["n_grounding_configs"]["value"] == 1
    assert result["collapse_detected_deploy_arm"]["value"] is False
    assert result["collapse_detected_control_beta0"]["value"] is True
    assert result["factual_verifier_calibration_improved"]["value"] is True
    assert result["pass_rate_vs_true_accuracy_distinct_assert"]["value"] is True
    assert result["quality_maintained"]["value"] is True
    assert result["random_seed"]["value"] == 42
    assert result["reproducibility_checksum"]["value"] == "checksum_v5"
    assert result["duration_s"]["value"] == 0.0

def test_evaluate_continuous_self_learning_v5_failure():
    """Test failed evaluation."""
    result = evaluate_continuous_self_learning_v5(
        deploy_arm_collapse=True,
        control_beta0_collapse=True,
        pass_rate=[0.8, 0.85],
        true_accuracy=[0.8, 0.85],
        quality_maintained=False,
        factual_verifier_calibration_improved=False
    )
    
    assert result["honest_verdict"]["value"] == "complete: fr11_does_not_hold_on_fresh_corpus_self_learning_needs_new_mechanism"
    assert result["collapse_detected_deploy_arm"]["value"] is True
    assert result["pass_rate_vs_true_accuracy_distinct_assert"]["value"] is False
    assert result["quality_maintained"]["value"] is False
