"""Tests for Experiment 3521 — FR-11 Adaptive Online Beta Robust Default v1.

References: REQ-FR11-AOB-001, REQ-FR11-AOB-002, REQ-FR11-AOB-003

NOTE (2026-05-31 outer-loop repair): the original gemini-authored test imported
two functions under stale names (`_compute_decisions` /
`_compute_weighted_sigma`) and called `run_arm_with_progress` / `apply_law`
against an imagined API (a non-existent `static_lambda_min` kwarg, arm_type
labels "A_adaptive"/"B_beta0", and an upper clamp on `apply_law` the module
never had). That broke pytest collection, which failed the conductor's pre-test
gate and SKIPped every task. This rewrite binds the tests to the ACTUAL module
API (the source of truth — the experiment ran against it), with assertions
derived from the real implementation.
"""
import json

import numpy as np
import pytest

from carnot.fr11.adaptive_online_beta_robust_default_v1 import (
    LAW_INTERCEPT,
    LAW_SLOPE,
    apply_law,
    _compute_decisions_matrix,
    _compute_weighted_covariance,
    run_adaptive_online_beta_robust_default,
    run_arm_with_progress,
)


def test_apply_law_floors_but_does_not_upper_clamp():
    """The deployed law is beta = LAW_SLOPE*lambda_min + LAW_INTERCEPT, floored
    at beta_floor (default 0.0). There is NO upper clamp — high lambda_min yields
    a beta above 0.5, which the runtime then uses directly."""
    # Low lambda_min -> negative prediction -> floored to 0.0
    assert apply_law(0.0) == 0.0
    # High lambda_min -> linear law, no upper clamp
    assert apply_law(1.0) == pytest.approx(LAW_SLOPE * 1.0 + LAW_INTERCEPT)
    assert apply_law(1.0) > 0.5  # confirms there is no 0.5 ceiling
    # A mid value stays positive and below the high value
    mid = apply_law(0.2)
    assert 0.0 < mid < apply_law(1.0)
    # Custom floor is respected
    assert apply_law(0.0, beta_floor=0.2) == 0.2


def test_compute_weighted_covariance():
    """The weighted covariance centers each decision channel by its
    probability-weighted mean and returns the prob-weighted outer product."""
    decisions = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    # Uniform probabilities: mean_d0=2/3, mean_d1=1/3
    probs1 = np.array([1 / 3, 1 / 3, 1 / 3])
    sigma1 = _compute_weighted_covariance(decisions, probs1)
    # sigma_00 = 1/3*(1/3)^2 + 1/3*(-2/3)^2 + 1/3*(1/3)^2 = 6/27 = 2/9
    assert np.isclose(sigma1[0, 0], 2 / 9)
    assert sigma1.shape == (2, 2)
    # All mass on one trace -> zero variance
    probs2 = np.array([1.0, 0.0, 0.0])
    sigma2 = _compute_weighted_covariance(decisions, probs2)
    assert np.allclose(sigma2, 0.0)


def test_compute_decisions_matrix_shape_and_determinism():
    """The decisions matrix is (n_traces, n_channels) and deterministic in the
    seed (so the experiment is reproducible)."""
    traces = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]
    d1 = _compute_decisions_matrix(traces, 0.1, 4, seed=42)
    d2 = _compute_decisions_matrix(traces, 0.1, 4, seed=42)
    assert d1.shape == (3, 4)
    assert np.array_equal(d1, d2)  # deterministic
    assert set(np.unique(d1)).issubset({0.0, 1.0})  # binary decisions


def _arm_kwargs(arm_type, static_beta=0.0):
    """Build a valid run_arm_with_progress call for the real 8-arg signature."""
    traces = [{"is_correct": True}, {"is_correct": False}]
    at_risk_scores = np.array([0.9, 0.1])
    decisions = _compute_decisions_matrix(traces, 0.1, 4, seed=42)
    return dict(
        traces=traces,
        at_risk_scores=at_risk_scores,
        decisions=decisions,
        n_iterations=5,
        arm_type=arm_type,
        static_beta=static_beta,
        config_name="test_config",
        arm_label=f"{arm_type}_label",
    )


@pytest.mark.parametrize("arm_type,static_beta", [
    ("adaptive", 0.0),
    ("beta0", 0.0),
    ("fixed", 0.0),
    ("static", 0.15),
])
def test_run_arm_with_progress_returns_expected_keys(arm_type, static_beta):
    """Each arm runs to completion and returns the metric dict the top-level
    aggregator consumes.

    References: REQ-FR11-AOB-002
    """
    result = run_arm_with_progress(**_arm_kwargs(arm_type, static_beta))
    for key in (
        "collapse_detected", "final_entropy", "final_mode_mass",
        "entropy_drop_ratio", "final_pass_rate", "final_true_accuracy",
        "final_gap",
    ):
        assert key in result
    assert isinstance(result["final_entropy"], float)
    assert isinstance(result["collapse_detected"], (bool, np.bool_))


def test_run_arm_with_progress_invalid_arm_raises():
    """An unknown arm_type raises ValueError (guards the verdict logic against a
    silently-mislabeled arm)."""
    traces = [{"is_correct": True}]
    with pytest.raises(ValueError):
        run_arm_with_progress(
            traces=traces,
            at_risk_scores=np.array([0.8]),
            decisions=np.array([[1.0]]),
            n_iterations=1,
            arm_type="invalid",
            static_beta=0.1,
            config_name="test",
            arm_label="invalid_label",
        )


def _write_traces(tmp_path):
    traces = [{"is_correct": True}, {"is_correct": False}]
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return p


def test_run_adaptive_online_beta_robust_default_smoke(tmp_path):
    """End-to-end on a tiny corpus + single fresh config: emits the required
    artifact fields."""
    p = _write_traces(tmp_path)
    res = run_adaptive_online_beta_robust_default(
        traces_path=str(p),
        n_iterations=5,
        seed=42,
        fresh_configs=[{"name": "test_cfg", "active_weight": 0.1}],
    )
    assert "honest_verdict" in res
    assert "reproducibility_checksum" in res
    assert res["honest_verdict"].startswith("complete:")


def test_run_adaptive_online_beta_no_traces_blocks(tmp_path):
    """Empty corpus -> honest blocked verdict, not a fabricated success."""
    p = tmp_path / "traces.jsonl"
    p.touch()
    res = run_adaptive_online_beta_robust_default(
        traces_path=str(p), n_iterations=5, seed=42, fresh_configs=[]
    )
    assert res["honest_verdict"] == (
        "complete: blocked_fr11_module_or_traces_unavailable"
    )


def _mock_arm(collapse_for):
    """Return a run_arm_with_progress mock that reports collapse for the given
    real arm_type names (positional arg index 4)."""
    def mock(*args, **kwargs):
        arm_type = args[4] if len(args) > 4 else kwargs.get("arm_type")
        return {
            "collapse_detected": arm_type in collapse_for,
            "final_entropy": 1.0,
            "final_mode_mass": 0.5,
            "entropy_drop_ratio": 0.5,
            "final_pass_rate": 1.0,
            "final_true_accuracy": 1.0,
            "final_gap": 0.0,
        }
    return mock


def test_verdict_conservative_wins(tmp_path):
    """Adaptive (A) AND control (B) collapse but fixed-conservative (C) does
    not -> the conservative default is the robust rule."""
    from unittest.mock import patch
    p = _write_traces(tmp_path)
    with patch(
        "carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress",
        side_effect=_mock_arm(collapse_for={"adaptive", "beta0"}),
    ):
        res = run_adaptive_online_beta_robust_default(
            str(p), n_iterations=1, seed=42, fresh_configs=None
        )
    assert res["honest_verdict"] == (
        "complete: conservative_default_beta_is_the_robust_phase5_default_"
        "adaptive_online_unnecessary"
    )


def test_verdict_adaptive_wins(tmp_path):
    """Only the control (B) collapses; adaptive (A) prevents collapse -> the
    adaptive-online rule is the deployable Phase-5 default."""
    from unittest.mock import patch
    p = _write_traces(tmp_path)
    with patch(
        "carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress",
        side_effect=_mock_arm(collapse_for={"beta0"}),
    ):
        res = run_adaptive_online_beta_robust_default(
            str(p), n_iterations=1, seed=42, fresh_configs=None
        )
    assert res["honest_verdict"] == (
        "complete: adaptive_online_beta_prevents_collapse_phase5_deployable_"
        "default_confirmed"
    )


def test_verdict_neither_wins(tmp_path):
    """Every arm collapses -> no beta rule robustly prevents collapse."""
    from unittest.mock import patch
    p = _write_traces(tmp_path)
    with patch(
        "carnot.fr11.adaptive_online_beta_robust_default_v1.run_arm_with_progress",
        side_effect=_mock_arm(collapse_for={"adaptive", "beta0", "fixed", "static"}),
    ):
        res = run_adaptive_online_beta_robust_default(
            str(p), n_iterations=1, seed=42, fresh_configs=None
        )
    assert res["honest_verdict"] == (
        "complete: no_beta_rule_robustly_prevents_collapse_across_configs_"
        "self_learning_needs_new_mechanism"
    )
