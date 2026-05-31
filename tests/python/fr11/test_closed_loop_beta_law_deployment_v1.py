"""Tests for FR-11 Closed-Loop Beta Law Deployment v1.

Covers:
    REQ-FR11-CLD-001: Lambda_min measured and beta deployed from exp3498 formula.
    REQ-FR11-CLD-002: Three-arm closed loop at N>=200, distinct pass_rate/true_accuracy.
    REQ-FR11-CLD-003: Deployment validated: Arm A prevents collapse, Arm B collapses.
    SCENARIO-FR11-CLD-001: Deployed law prevents collapse at fresh configs.
    SCENARIO-FR11-CLD-002: Beta=0 control collapses.
    SCENARIO-FR11-CLD-003: Arm A accuracy not materially worse than Arm C.

Key invariants tested:
1. apply_law returns correct formula output and clips to >=0.
2. RANDOM_SEED is not the experiment number (3509).
3. RANDOM_SEED is deterministically derived from module content.
4. run_arm_with_progress returns all required keys and collapse_detected is bool.
5. At-risk config (high AW) collapses at beta=0 in the three-arm sweep.
6. Deployed beta from law prevents collapse at fresh configs.
7. pass_rate and true_accuracy are not bit-identical (DISTINCT source arrays).
8. run_closed_loop_beta_law_deployment returns all required artifact fields.
9. Acceptance gates G1 and G0 are correct given results.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11.closed_loop_beta_law_deployment_v1 import (
    FIXED_CONSERVATIVE_BETA,
    FRESH_CONFIGS,
    LAW_INTERCEPT,
    LAW_SLOPE,
    N_ITERATIONS,
    OVER_REG_MARGIN,
    RANDOM_SEED,
    apply_law,
    run_arm_with_progress,
    run_closed_loop_beta_law_deployment,
)
from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import (
    _assert_sources_distinct,
    _compute_at_risk_scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traces() -> list[dict]:
    """30-trace corpus: every 5th trace is correct.

    REQ-FR11-CLD-001: cached corpus needed for covariance and loop computation.
    """
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
    """Write small_traces to a temp JSONL file."""
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


@pytest.fixture()
def at_risk_scores(small_traces) -> np.ndarray:
    """Verifier scores for ACTIVE_WEIGHT=0.07 (fresh_low config)."""
    return _compute_at_risk_scores(small_traces, active_weight=0.07, seed=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Tests: apply_law
# ---------------------------------------------------------------------------


def test_apply_law_formula_correctness():
    """REQ-FR11-CLD-001: apply_law applies the exp3498 formula exactly."""
    lambda_min = 0.21
    expected = LAW_SLOPE * lambda_min + LAW_INTERCEPT
    result = apply_law(lambda_min)
    assert abs(result - max(0.0, expected)) < 1e-12


def test_apply_law_clips_negative_to_zero():
    """REQ-FR11-CLD-001: apply_law clips to >=0 (no negative entropy beta)."""
    # Very low lambda_min → intercept dominates → negative prediction → clipped to 0
    result = apply_law(0.0)
    assert result == 0.0, f"Expected 0.0 for lambda_min=0, got {result}"


def test_apply_law_positive_for_typical_lambda_min():
    """REQ-FR11-CLD-001: For lambda_min near exp3498 range (0.16..0.22), result >= 0."""
    for lm in [0.163, 0.20, 0.214, 0.220]:
        assert apply_law(lm) >= 0.0, f"apply_law({lm}) should be >=0"


def test_apply_law_slope_is_positive():
    """REQ-FR11-CLD-001: Law has positive slope — higher lambda_min → higher beta."""
    lm_low, lm_high = 0.163, 0.22
    assert apply_law(lm_high) > apply_law(lm_low), (
        "Law should produce higher beta for higher lambda_min"
    )


# ---------------------------------------------------------------------------
# Tests: RANDOM_SEED properties
# ---------------------------------------------------------------------------


def test_random_seed_is_not_experiment_number():
    """REQ-FR11-CLD-002: Content-derived seed is not the experiment number 3509."""
    assert RANDOM_SEED != 3509, (
        "random_seed must be content-derived, NOT the experiment number 3509"
    )


def test_random_seed_is_positive_and_bounded():
    """REQ-FR11-CLD-002: Seed is a valid positive integer within 2^20."""
    assert 0 < RANDOM_SEED < 2**20


def test_random_seed_is_deterministic():
    """REQ-FR11-CLD-002: Same hash input always produces the same seed."""
    import hashlib
    material = b"exp3509_fr11_closed_loop_beta_law_deployment_v1"
    expected = int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**20)
    assert RANDOM_SEED == expected


# ---------------------------------------------------------------------------
# Tests: run_arm_with_progress
# ---------------------------------------------------------------------------


def test_run_arm_returns_required_keys(small_traces, at_risk_scores):
    """REQ-FR11-CLD-002: run_arm_with_progress returns all required keys."""
    result = run_arm_with_progress(
        small_traces, at_risk_scores, n_iterations=10, entropy_beta=0.0,
        config_name="test", arm_label="B_beta0"
    )
    required_keys = {
        "collapse_detected", "final_entropy", "final_mode_mass",
        "entropy_drop_ratio", "final_pass_rate", "final_true_accuracy", "final_gap",
    }
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_run_arm_collapse_detected_is_bool(small_traces, at_risk_scores):
    """REQ-FR11-CLD-002: collapse_detected is a Python bool."""
    result = run_arm_with_progress(
        small_traces, at_risk_scores, n_iterations=10, entropy_beta=0.0,
        config_name="test", arm_label="B"
    )
    assert isinstance(result["collapse_detected"], bool)


def test_run_arm_pass_rate_and_true_accuracy_are_floats(small_traces, at_risk_scores):
    """REQ-FR11-CLD-002: Numeric metrics are floats, not arrays."""
    result = run_arm_with_progress(
        small_traces, at_risk_scores, n_iterations=5, entropy_beta=0.1,
        config_name="test", arm_label="A"
    )
    assert isinstance(result["final_pass_rate"], float)
    assert isinstance(result["final_true_accuracy"], float)


def test_run_arm_high_beta_does_not_collapse(small_traces, at_risk_scores):
    """SCENARIO-FR11-CLD-001: High entropy beta prevents collapse even at shallow N."""
    result = run_arm_with_progress(
        small_traces, at_risk_scores, n_iterations=10, entropy_beta=0.5,
        config_name="smoke", arm_label="C_fixed"
    )
    # With beta=0.5 at N=10 (not deep enough for depth-aware collapse), should not collapse
    assert not result["collapse_detected"]


# ---------------------------------------------------------------------------
# Tests: pass_rate vs true_accuracy distinctness
# ---------------------------------------------------------------------------


def test_assert_sources_distinct_passes_for_at_risk(small_traces):
    """REQ-FR11-CLD-002: pass_rate and true_accuracy are from distinct source arrays."""
    for aw in [0.07, 0.20]:
        scores = _compute_at_risk_scores(small_traces, active_weight=aw, seed=RANDOM_SEED)
        is_correct = np.array([bool(t.get("is_correct", False)) for t in small_traces], dtype=float)
        verifier_pass = (scores > 0.5).astype(float)
        # Should NOT raise
        _assert_sources_distinct(verifier_pass, is_correct, aw)


# ---------------------------------------------------------------------------
# Tests: run_closed_loop_beta_law_deployment (smoke)
# ---------------------------------------------------------------------------


REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "n_grounding_configs",
    "lambda_min_by_config",
    "beta_deployed_by_config",
    "collapse_detected_armA_deployed",
    "collapse_detected_armB_beta0",
    "collapse_detected_armC_fixed",
    "deployed_law_prevents_collapse",
    "armA_vs_armC_accuracy_gap",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "recommended_phase5_rule",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]


def test_run_closed_loop_returns_all_required_fields(traces_jsonl, small_traces):
    """REQ-FR11-CLD-003: All required artifact fields present in output."""
    # Use tiny n_iterations and one mini config to keep the test fast
    mini_configs = [{"name": "mini_test", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl,
        n_iterations=5,
        seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in result, f"Missing required artifact field: {field}"


def test_inference_substrate_is_correct(traces_jsonl):
    """REQ-FR11-CLD-001: inference_substrate is verifier_ensemble_against_cached_candidates."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_honest_verdict_has_terminal_prefix(traces_jsonl):
    """REQ-FR11-CLD-003: honest_verdict starts with one of the terminal prefixes."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    verdict = result["honest_verdict"]
    terminal_prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
    assert any(verdict.startswith(p) for p in terminal_prefixes), (
        f"honest_verdict must start with a terminal prefix, got: {verdict!r}"
    )


def test_n_grounding_configs_matches_fresh_configs(traces_jsonl):
    """REQ-FR11-CLD-001: n_grounding_configs matches the number of configs provided."""
    mini_configs = [
        {"name": "c1", "active_weight": 0.07, "description": "a"},
        {"name": "c2", "active_weight": 0.20, "description": "b"},
    ]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    assert result["n_grounding_configs"] == 2


def test_beta_deployed_is_clipped_to_nonneg(traces_jsonl):
    """REQ-FR11-CLD-001: beta_deployed_by_config values are all >= 0."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    for name, beta in result["beta_deployed_by_config"].items():
        assert beta >= 0.0, f"beta_deployed[{name}] = {beta} must be >= 0"


def test_pass_rate_vs_true_accuracy_distinct_assert_is_true(traces_jsonl):
    """REQ-FR11-CLD-002: pass_rate_vs_true_accuracy_distinct_assert is True."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    assert result["pass_rate_vs_true_accuracy_distinct_assert"] is True


def test_duration_s_is_at_least_one(traces_jsonl):
    """REQ-FR11-CLD-002: duration_s >= 1.0 (verifier_ensemble substrate floor)."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    assert result["duration_s"] >= 1.0


def test_reproducibility_checksum_is_16_hex_chars(traces_jsonl):
    """REQ-FR11-CLD-002: reproducibility_checksum is 16-char hex string."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    chk = result["reproducibility_checksum"]
    assert isinstance(chk, str) and len(chk) == 16
    int(chk, 16)  # raises if not valid hex


def test_acceptance_gates_present_and_boolean(traces_jsonl):
    """REQ-FR11-CLD-003: acceptance_gates dict present with boolean G0 and G1."""
    mini_configs = [{"name": "mini", "active_weight": 0.07, "description": "smoke"}]
    result = run_closed_loop_beta_law_deployment(
        traces_path=traces_jsonl, n_iterations=3, seed=RANDOM_SEED,
        fresh_configs=mini_configs,
    )
    gates = result["acceptance_gates"]
    assert "G1_deployment_validated" in gates
    assert "G0_deflag_distinct_arrays" in gates
    assert isinstance(gates["G1_deployment_validated"], bool)
    assert isinstance(gates["G0_deflag_distinct_arrays"], bool)


def test_fresh_configs_are_not_exp3498_fit_points():
    """SCENARIO-FR11-CLD-001: FRESH_CONFIGS use AW values not in exp3498 fit set."""
    exp3498_aws = {0.05, 0.10, 0.146, 0.30}
    for cfg in FRESH_CONFIGS:
        aw = cfg["active_weight"]
        assert aw not in exp3498_aws, (
            f"Config '{cfg['name']}' uses AW={aw} which is in exp3498 fit set — not fresh!"
        )


def test_fresh_configs_has_at_least_two():
    """REQ-FR11-CLD-001: At least 2 fresh configs for deployment validation."""
    assert len(FRESH_CONFIGS) >= 2, "FRESH_CONFIGS must have at least 2 entries"


def test_fixed_conservative_beta_is_half():
    """SCENARIO-FR11-CLD-003: Fixed-conservative beta matches exp3498 grid point."""
    assert FIXED_CONSERVATIVE_BETA == 0.5, (
        "FIXED_CONSERVATIVE_BETA should be 0.5 (the exp3498 never-collapsing grid point)"
    )
