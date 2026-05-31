"""Tests for FR-11 Adaptive Online Beta Robust Default v1."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11.adaptive_online_beta_robust_default_v1 import (
    FIXED_CONSERVATIVE_BETA,
    FRESH_CONFIGS,
    LAW_INTERCEPT,
    LAW_SLOPE,
    N_ITERATIONS,
    OVER_REG_MARGIN,
    RANDOM_SEED,
    apply_law,
    run_arm_with_progress,
    run_adaptive_online_beta_robust_default,
    _compute_decisions_matrix,
    _compute_weighted_covariance,
)
from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import (
    _assert_sources_distinct,
    _compute_at_risk_scores,
)

@pytest.fixture()
def small_traces() -> list[dict]:
    return [
        {
            "question_id": f"q{i:03d}",
            "prompt": f"Q{i}",
            "completion": f"A{i}",
            "is_correct": (i % 5 == 0),
        }
        for i in range(30)
    ]

@pytest.fixture()
def traces_jsonl(small_traces, tmp_path) -> str:
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)

@pytest.fixture()
def at_risk_scores(small_traces) -> np.ndarray:
    return _compute_at_risk_scores(small_traces, active_weight=0.07, seed=RANDOM_SEED)

@pytest.fixture()
def decisions(small_traces) -> np.ndarray:
    return _compute_decisions_matrix(small_traces, active_weight=0.07, seed=RANDOM_SEED)

def test_apply_law_formula_correctness():
    lambda_min = 0.21
    expected = LAW_SLOPE * lambda_min + LAW_INTERCEPT
    result = apply_law(lambda_min)
    assert abs(result - max(0.0, expected)) < 1e-12

def test_apply_law_clips_negative_to_zero():
    result = apply_law(0.0)
    assert result == 0.0

def test_random_seed_is_not_experiment_number():
    assert RANDOM_SEED != 3521

def test_random_seed_is_deterministic():
    import hashlib
    material = b"exp3521_fr11_adaptive_online_beta_robust_default_v1"
    expected = int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**20)
    assert RANDOM_SEED == expected

def test_run_arm_returns_required_keys(small_traces, at_risk_scores, decisions):
    result = run_arm_with_progress(
        small_traces, at_risk_scores, decisions, n_iterations=10, arm_type="beta0",
        static_beta=0.0, config_name="test", arm_label="B_beta0"
    )
    required_keys = {
        "collapse_detected", "final_entropy", "final_mode_mass",
        "entropy_drop_ratio", "final_pass_rate", "final_true_accuracy", "final_gap",
    }
    for k in required_keys:
        assert k in result, f"Missing key: {k}"

def test_run_arm_adaptive_type(small_traces, at_risk_scores, decisions):
    result = run_arm_with_progress(
        small_traces, at_risk_scores, decisions, n_iterations=10, arm_type="adaptive",
        static_beta=0.0, config_name="test", arm_label="A_adaptive"
    )
    assert isinstance(result["collapse_detected"], bool)

def test_run_arm_static_type(small_traces, at_risk_scores, decisions):
    result = run_arm_with_progress(
        small_traces, at_risk_scores, decisions, n_iterations=10, arm_type="static",
        static_beta=0.1, config_name="test", arm_label="D_static"
    )
    assert isinstance(result["collapse_detected"], bool)

def test_run_arm_fixed_type(small_traces, at_risk_scores, decisions):
    result = run_arm_with_progress(
        small_traces, at_risk_scores, decisions, n_iterations=10, arm_type="fixed",
        static_beta=0.0, config_name="test", arm_label="C_fixed"
    )
    assert isinstance(result["collapse_detected"], bool)

def test_run_arm_unknown_type(small_traces, at_risk_scores, decisions):
    with pytest.raises(ValueError):
        run_arm_with_progress(
            small_traces, at_risk_scores, decisions, n_iterations=10, arm_type="unknown",
            static_beta=0.0, config_name="test", arm_label="U"
        )

def test_assert_sources_distinct_passes_for_at_risk(small_traces):
    for aw in [0.06, 0.22]:
        scores = _compute_at_risk_scores(small_traces, active_weight=aw, seed=RANDOM_SEED)
        is_correct = np.array([bool(t.get("is_correct", False)) for t in small_traces], dtype=float)
        verifier_pass = (scores > 0.5).astype(float)
        _assert_sources_distinct(verifier_pass, is_correct, aw)

REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "n_grounding_configs",
    "lambda_min_by_config",
    "collapse_detected_armA_adaptive_online",
    "collapse_detected_armB_beta0",
    "collapse_detected_armC_conservative",
    "collapse_detected_armD_static_offline_law",
    "adaptive_online_prevents_collapse",
    "conservative_default_prevents_collapse",
    "winning_arm_vs_least_regularized_accuracy_gap",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "recommended_phase5_rule",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]

def test_run_returns_all_required_fields(traces_jsonl, small_traces):
    mini_configs = [{"name": "mini_test", "active_weight": 0.06, "description": "smoke"}]
    result = run_adaptive_online_beta_robust_default(
        traces_path=traces_jsonl,
        n_iterations=5,
        seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in result, f"Missing required artifact field: {field}"

def test_honest_verdict_has_terminal_prefix(traces_jsonl):
    mini_configs = [{"name": "mini", "active_weight": 0.06, "description": "smoke"}]
    result = run_adaptive_online_beta_robust_default(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    verdict = result["honest_verdict"]
    terminal_prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
    assert any(verdict.startswith(p) for p in terminal_prefixes), (
        f"honest_verdict must start with a terminal prefix, got: {verdict!r}"
    )

def test_fresh_configs_has_at_least_four():
    assert len(FRESH_CONFIGS) >= 4, "FRESH_CONFIGS must have at least 4 entries"
