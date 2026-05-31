"""Tests for FR-11 Continuous Self Learning v4 Module."""

from carnot.fr11.continuous_self_learning_v4 import evaluate_continuous_self_learning

def test_evaluate_continuous_self_learning_success():
    """Test successful evaluation."""
    result = evaluate_continuous_self_learning(
        deploy_arm_collapse=False,
        control_beta0_collapse=True,
        pass_rate=[0.8, 0.85],
        true_accuracy=[0.75, 0.8],
        quality_maintained=True
    )
    
    assert result["honest_verdict"] == "complete: fr11_conservative_default_holds_on_fresh_nondegenerate_corpus_quality_maintained"
    assert result["inference_substrate"] == "FR-11-v4"
    assert result["n_grounding_configs"] == 1
    assert result["collapse_detected_deploy_arm"] is False
    assert result["collapse_detected_control_beta0"] is True
    assert result["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert result["quality_maintained"] is True
    assert result["random_seed"] == 42
    assert result["reproducibility_checksum"] == "checksum_v4"
    assert result["duration_s"] == 0.0

def test_evaluate_continuous_self_learning_failure():
    """Test failed evaluation."""
    result = evaluate_continuous_self_learning(
        deploy_arm_collapse=True,
        control_beta0_collapse=True,
        pass_rate=[0.8, 0.85],
        true_accuracy=[0.8, 0.85],
        quality_maintained=False
    )
    
    assert result["honest_verdict"] == "complete: fr11_does_not_hold_out_of_sample_self_learning_needs_new_mechanism"
    assert result["collapse_detected_deploy_arm"] is True
    assert result["pass_rate_vs_true_accuracy_distinct_assert"] is False
    assert result["quality_maintained"] is False
