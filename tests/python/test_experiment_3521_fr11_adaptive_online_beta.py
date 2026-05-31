"""Tests for Experiment 3521 — FR-11 Adaptive Online Beta Robust Default v1.

References: REQ-FR11-AOB-001, REQ-FR11-AOB-002, REQ-FR11-AOB-003
"""

import numpy as np

from carnot.fr11.adaptive_online_beta_robust_default_v1 import (
    apply_law,
    _compute_decisions,
    _compute_weighted_sigma,
    run_arm_with_progress,
)

def test_apply_law_clamped():
    """Test that the law properly clamps predicted betas to [0.0, 0.5]."""
    # High lambda_min -> high predicted beta, clamps to 0.5
    assert apply_law(1.0) == 0.5
    # Low lambda_min -> negative predicted beta, clamps to 0.0
    assert apply_law(0.0) == 0.0
    # Middle lambda_min -> within range
    assert 0.0 < apply_law(0.2) < 0.5

def test_compute_weighted_sigma():
    """Test that the weighted sigma correctly respects probabilities."""
    decisions = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ])
    
    # Uniform probabilities
    probs1 = np.array([1/3, 1/3, 1/3])
    sigma1 = _compute_weighted_sigma(decisions, probs1)
    
    # Centered: 
    # mean_d0 = 2/3, mean_d1 = 1/3
    # d_centered = [[1/3, 2/3], [-2/3, -1/3], [1/3, -1/3]]
    # sigma_00 = 1/3*(1/9) + 1/3*(4/9) + 1/3*(1/9) = 6/27 = 2/9 = 0.222
    assert np.isclose(sigma1[0, 0], 2/9)
    assert sigma1.shape == (2, 2)
    
    # Skewed probabilities: all mass on the first trace
    probs2 = np.array([1.0, 0.0, 0.0])
    sigma2 = _compute_weighted_sigma(decisions, probs2)
    # Variance should be 0 because all mass is on one outcome
    assert np.allclose(sigma2, 0.0)

def test_run_arm_with_progress_adaptive():
    """Test that the adaptive arm runs without error and returns expected dict structure.
    
    References: REQ-FR11-AOB-002
    """
    traces = [{"is_correct": True}, {"is_correct": False}]
    at_risk_scores = np.array([0.9, 0.1])
    decisions = _compute_decisions(traces, 0.1, 4, seed=42)
    
    result = run_arm_with_progress(
        traces=traces,
        at_risk_scores=at_risk_scores,
        decisions=decisions,
        n_iterations=5,
        arm_type="A_adaptive",
        static_lambda_min=0.1,
        config_name="test_config"
    )
    
    assert "collapse_detected" in result
    assert "final_entropy" in result
    assert "final_mode_mass" in result
    assert "final_pass_rate" in result
    assert "final_true_accuracy" in result
    assert "final_gap" in result

def test_run_arm_with_progress_static():
    """Test that the static arm runs properly using the static_lambda_min.
    
    References: REQ-FR11-AOB-002
    """
    traces = [{"label": "correct"}, {"label": "incorrect"}]
    at_risk_scores = np.array([0.8, 0.2])
    decisions = _compute_decisions(traces, 0.1, 4, seed=42)
    
    result = run_arm_with_progress(
        traces=traces,
        at_risk_scores=at_risk_scores,
        decisions=decisions,
        n_iterations=5,
        arm_type="D_static",
        static_lambda_min=0.15,
        config_name="test_config"
    )
    
    assert isinstance(result["final_entropy"], float)

def test_run_arm_with_progress_beta0():
    traces = [{"label": "correct"}, {"label": "incorrect"}]
    at_risk_scores = np.array([0.8, 0.2])
    decisions = _compute_decisions(traces, 0.1, 4, seed=42)
    result = run_arm_with_progress(
        traces=traces, at_risk_scores=at_risk_scores, decisions=decisions,
        n_iterations=5, arm_type="B_beta0", static_lambda_min=0.15, config_name="test"
    )
    assert isinstance(result["final_entropy"], float)

def test_run_arm_with_progress_conservative():
    traces = [{"label": "correct"}, {"label": "incorrect"}]
    at_risk_scores = np.array([0.8, 0.2])
    decisions = _compute_decisions(traces, 0.1, 4, seed=42)
    result = run_arm_with_progress(
        traces=traces, at_risk_scores=at_risk_scores, decisions=decisions,
        n_iterations=5, arm_type="C_conservative", static_lambda_min=0.15, config_name="test"
    )
    assert isinstance(result["final_entropy"], float)

def test_run_arm_with_progress_invalid_arm():
    import pytest
    traces = [{"label": "correct"}]
    with pytest.raises(ValueError):
        run_arm_with_progress(
            traces=traces, at_risk_scores=np.array([0.8]), decisions=np.array([[1.0]]),
            n_iterations=1, arm_type="invalid", static_lambda_min=0.1, config_name="test"
        )

def test_run_adaptive_online_beta_robust_default(tmp_path):
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import run_adaptive_online_beta_robust_default
    import json
    traces = [{"is_correct": True}, {"is_correct": False}]
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    
    fresh_configs = [{"name": "test_cfg", "active_weight": 0.1}]
    res = run_adaptive_online_beta_robust_default(
        traces_path=str(p),
        n_iterations=5,
        seed=42,
        fresh_configs=fresh_configs
    )
    assert "honest_verdict" in res
    assert "reproducibility_checksum" in res

def test_run_adaptive_online_beta_robust_default_no_traces(tmp_path):
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import run_adaptive_online_beta_robust_default
    p = tmp_path / "traces.jsonl"
    p.touch()
    res = run_adaptive_online_beta_robust_default(
        traces_path=str(p), n_iterations=5, seed=42, fresh_configs=[]
    )
    assert res["honest_verdict"] == "complete: blocked_fr11_module_or_traces_unavailable"

def test_run_adaptive_online_beta_conservative_wins(tmp_path):
    from unittest.mock import patch
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import run_adaptive_online_beta_robust_default
    import json
    traces = [{"is_correct": True}, {"is_correct": False}]
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    
    # Mock run_arm_with_progress so that Arm A collapses but Arm C does not.
    # Also Arm B collapses.
    def mock_run_arm(*args, **kwargs):
        arm_type = args[4] if len(args) > 4 else kwargs.get("arm_type")
        collapsed = False
        if arm_type == "A_adaptive" or arm_type == "B_beta0":
            collapsed = True
        return {
            "collapse_detected": collapsed,
            "final_entropy": 1.0,
            "final_mode_mass": 0.5,
            "entropy_drop_ratio": 0.5,
            "final_pass_rate": 1.0,
            "final_true_accuracy": 1.0,
            "final_gap": 0.0,
        }
    with patch("carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress", side_effect=mock_run_arm):
        res = run_adaptive_online_beta_robust_default(str(p), n_iterations=1, seed=42, fresh_configs=None)
    assert res["honest_verdict"] == "complete: conservative_default_beta_is_the_robust_phase5_default_adaptive_online_unnecessary"

def test_run_adaptive_online_beta_neither_wins(tmp_path):
    from unittest.mock import patch
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import run_adaptive_online_beta_robust_default
    import json
    traces = [{"is_correct": True}, {"is_correct": False}]
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    
    def mock_run_arm(*args, **kwargs):
        # All collapse
        return {
            "collapse_detected": True,
            "final_entropy": 1.0,
            "final_mode_mass": 0.5,
            "entropy_drop_ratio": 0.5,
            "final_pass_rate": 1.0,
            "final_true_accuracy": 1.0,
            "final_gap": 0.0,
        }
    with patch("carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress", side_effect=mock_run_arm):
        res = run_adaptive_online_beta_robust_default(str(p), n_iterations=1, seed=42, fresh_configs=None)
    assert res["honest_verdict"] == "complete: no_beta_rule_robustly_prevents_collapse_across_configs_self_learning_needs_new_mechanism"

def test_run_adaptive_online_beta_adaptive_wins(tmp_path):
    from unittest.mock import patch
    from carnot.fr11.adaptive_online_beta_robust_default_v1 import run_adaptive_online_beta_robust_default
    import json
    traces = [{"is_correct": True}, {"is_correct": False}]
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    
    def mock_run_arm(*args, **kwargs):
        arm_type = args[4] if len(args) > 4 else kwargs.get("arm_type")
        collapsed = True if arm_type == "B_beta0" else False
        return {
            "collapse_detected": collapsed,
            "final_entropy": 1.0,
            "final_mode_mass": 0.5,
            "entropy_drop_ratio": 0.5,
            "final_pass_rate": 1.0,
            "final_true_accuracy": 1.0,
            "final_gap": 0.0,
        }
    with patch("carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress", side_effect=mock_run_arm):
        res = run_adaptive_online_beta_robust_default(str(p), n_iterations=1, seed=42, fresh_configs=None)
    assert res["honest_verdict"] == "complete: adaptive_online_beta_prevents_collapse_phase5_deployable_default_confirmed"
