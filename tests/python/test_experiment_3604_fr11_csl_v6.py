"""Tests for FR-11 Continuous Self Learning v6.

Traces to REQ-LEARN-3604, SCENARIO-LEARN-3604.
"""

import pytest
from carnot.fr11.continuous_self_learning_v6 import evaluate_continuous_self_learning_v6

def test_csl_v6_successful_calibration():
    """SCENARIO-LEARN-3604: Successful Calibration Prevents Collapse"""
    result = evaluate_continuous_self_learning_v6(
        deploy_arm_collapse=False,
        control_beta0_collapse=True,
        pass_rate=[0.80, 0.85],
        true_accuracy=[0.75, 0.80],
        quality_maintained=True,
        factual_verifier_calibration_improved=True
    )
    
    assert result["honest_verdict"]["value"] == "complete: fr11_conservative_default_calibrates_real_grounding_verifier_holds_quality_maintained"
    assert result["collapse_detected_deploy_arm"]["value"] is False
    assert result["collapse_detected_control_beta0"]["value"] is True
    assert result["pass_rate_vs_true_accuracy_distinct_assert"]["value"] is True
    assert result["factual_verifier_calibration_improved"]["value"] is True

def test_csl_v6_tautology_failure():
    """Test when pass_rate and true_accuracy are not distinct."""
    result = evaluate_continuous_self_learning_v6(
        deploy_arm_collapse=False,
        control_beta0_collapse=True,
        pass_rate=[0.80, 0.85],
        true_accuracy=[0.80, 0.85],
        quality_maintained=True,
        factual_verifier_calibration_improved=True
    )
    
    assert result["honest_verdict"]["value"] == "complete: fr11_does_not_hold_on_fresh_corpus_self_learning_needs_new_mechanism"
    assert result["pass_rate_vs_true_accuracy_distinct_assert"]["value"] is False

def test_csl_v6_deploy_arm_collapse_failure():
    """Test when deploy arm collapses."""
    result = evaluate_continuous_self_learning_v6(
        deploy_arm_collapse=True,
        control_beta0_collapse=True,
        pass_rate=[0.80, 0.85],
        true_accuracy=[0.75, 0.80],
        quality_maintained=True,
        factual_verifier_calibration_improved=True
    )
    
    assert result["honest_verdict"]["value"] == "complete: fr11_does_not_hold_on_fresh_corpus_self_learning_needs_new_mechanism"
