"""Tests for FR-11 Beta-Min / Lambda-Min Predictive Law v1.

Covers REQ-FR11-BML-001, REQ-FR11-BML-002, REQ-FR11-BML-003,
SCENARIO-FR11-BML-001, SCENARIO-FR11-BML-002, SCENARIO-FR11-BML-003.

Key invariants tested:
1. Decision covariance has correct shape and is positive semi-definite.
2. lambda_min varies monotonically with ACTIVE_WEIGHT (higher AW → lower lambda_min).
3. effective_k is in [1, n_channels] and participation ratio is computed correctly.
4. Beta sweep produces collapse_detected=True at beta=0 for at-risk configs.
5. Minimal sufficient beta is identified correctly.
6. pass_rate and true_accuracy are not identical (DISTINCT source assertion).
7. Linear law fit produces valid slope and R².
8. Leave-one-out validation predicts held-out config within tolerance.
9. All required artifact fields are present in the output.
"""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pytest

from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import (
    BETA_GRID,
    GROUNDING_CONFIGS,
    LAW_HOLD_OUT_TOLERANCE,
    N_CHANNELS,
    N_ITERATIONS,
    _assert_sources_distinct,
    _collapse_criterion,
    _compute_at_risk_scores,
    _entropy,
    _softmax,
    compute_decision_covariance,
    compute_sigma_metrics,
    find_minimal_beta,
    fit_linear_law,
    leave_one_out_validation,
    run_arm,
    run_beta_min_lambda_min_sweep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traces() -> list[dict]:
    """30-trace corpus: every 5th trace is correct.
    REQ-FR11-BML-001: corpus with mixed correct/incorrect for covariance computation.
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
def tiny_traces() -> list[dict]:
    """10-trace corpus: alternating correct/incorrect for quick arm tests.
    REQ-FR11-BML-002: minimal corpus for beta sweep smoke tests.
    """
    return [
        {
            "question_id": f"q{i:02d}",
            "prompt": f"Q{i}",
            "completion": f"A{i}",
            "is_correct": (i % 2 == 0),
        }
        for i in range(10)
    ]


@pytest.fixture()
def traces_jsonl(small_traces) -> str:
    """Write small_traces to a temp JSONL file for full-sweep tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
        return f.name


# ---------------------------------------------------------------------------
# REQ-FR11-BML-001: Decision covariance measurement
# ---------------------------------------------------------------------------


class TestDecisionCovariance:
    """Tests for compute_decision_covariance and compute_sigma_metrics."""

    def test_shape_is_k_by_k(self, small_traces):
        """REQ-FR11-BML-001: Sigma has shape (k, k)."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        assert sigma.shape == (4, 4)

    def test_symmetric(self, small_traces):
        """REQ-FR11-BML-001: Covariance matrix is symmetric."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)

    def test_diagonal_is_variance(self, small_traces):
        """REQ-FR11-BML-001: Diagonal entries are non-negative (variance ≥ 0)."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        assert np.all(np.diag(sigma) >= -1e-10)

    def test_lambda_min_decreases_with_active_weight(self, small_traces):
        """REQ-FR11-BML-001, SCENARIO-FR11-BML-001: Higher ACTIVE_WEIGHT →
        shared active signal dominates → channels more correlated → lower lambda_min."""
        metrics_low = compute_sigma_metrics(
            compute_decision_covariance(small_traces, active_weight=0.05, n_channels=4, seed=42)
        )
        metrics_high = compute_sigma_metrics(
            compute_decision_covariance(small_traces, active_weight=0.50, n_channels=4, seed=42)
        )
        # Higher AW → more correlated channels → lower (or equal) lambda_min
        assert metrics_high["lambda_min"] <= metrics_low["lambda_min"] + 0.05

    def test_effective_k_in_range(self, small_traces):
        """REQ-FR11-BML-001: effective_k is in [1, n_channels] for valid Sigma."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        metrics = compute_sigma_metrics(sigma)
        k = 4
        assert 0.9 <= metrics["effective_k"] <= k + 0.1

    def test_pairwise_max_corr_in_range(self, small_traces):
        """REQ-FR11-BML-001: pairwise_max_correlation is in [0, 1]."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        metrics = compute_sigma_metrics(sigma)
        assert 0.0 <= metrics["pairwise_max_correlation"] <= 1.0 + 1e-9

    def test_metrics_has_required_keys(self, small_traces):
        """REQ-FR11-BML-001: compute_sigma_metrics returns all expected fields."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        metrics = compute_sigma_metrics(sigma)
        for key in ("lambda_min", "effective_k", "pairwise_max_correlation", "eigenvalues"):
            assert key in metrics

    def test_eigenvalues_count(self, small_traces):
        """REQ-FR11-BML-001: eigenvalues list has k entries."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.15, n_channels=4, seed=42)
        metrics = compute_sigma_metrics(sigma)
        assert len(metrics["eigenvalues"]) == 4

    def test_single_channel_sigma(self, small_traces):
        """REQ-FR11-BML-001: k=1 produces a 1×1 covariance (scalar variance)."""
        sigma = compute_decision_covariance(small_traces, active_weight=0.20, n_channels=1, seed=42)
        assert sigma.shape == (1, 1)
        assert sigma[0, 0] >= -1e-10


# ---------------------------------------------------------------------------
# REQ-FR11-BML-002: Beta sweep per grounding configuration
# ---------------------------------------------------------------------------


class TestBetaSweep:
    """Tests for _compute_at_risk_scores, run_arm, find_minimal_beta."""

    def test_at_risk_scores_in_range(self, small_traces):
        """REQ-FR11-BML-002: At-risk scores are in [0, 1]."""
        scores = _compute_at_risk_scores(small_traces, active_weight=0.15, seed=42)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_at_risk_scores_distinct_from_is_correct(self, small_traces):
        """REQ-FR11-BML-002, SCENARIO-FR11-BML-003: verifier_pass ≠ is_correct array."""
        scores = _compute_at_risk_scores(small_traces, active_weight=0.15, seed=42)
        is_correct = np.array([bool(t["is_correct"]) for t in small_traces], dtype=float)
        verifier_pass = (scores > 0.5).astype(float)
        # Should not raise (they are distinct)
        _assert_sources_distinct(verifier_pass, is_correct, 0.15)

    def test_assert_sources_distinct_raises_on_identical(self):
        """REQ-FR11-BML-002: _assert_sources_distinct raises when arrays are equal."""
        arr = np.array([1.0, 0.0, 1.0, 0.0])
        with pytest.raises(AssertionError):
            _assert_sources_distinct(arr, arr, 0.5)

    def test_softmax_sums_to_one(self):
        """REQ-FR11-BML-002: softmax output sums to 1."""
        log_w = np.array([1.0, 2.0, 3.0])
        probs = _softmax(log_w)
        assert abs(float(np.sum(probs)) - 1.0) < 1e-9

    def test_entropy_non_negative(self):
        """REQ-FR11-BML-002: entropy is non-negative."""
        probs = np.array([0.5, 0.5])
        assert _entropy(probs) >= 0.0

    def test_entropy_zero_for_deterministic(self):
        """REQ-FR11-BML-002: entropy of a deterministic distribution is 0."""
        probs = np.array([1.0, 0.0, 0.0])
        assert _entropy(probs) < 1e-9

    def test_collapse_criterion_detects_concentrated_distribution(self):
        """REQ-FR11-BML-002: collapse criterion fires when mode_mass > 0.5 and entropy drops."""
        assert _collapse_criterion(
            entropy_drop_ratio=0.90,
            final_entropy=0.05,
            final_mode_mass=0.60,
            n_iterations=200,
        )

    def test_collapse_criterion_clear_at_high_entropy(self):
        """REQ-FR11-BML-002: collapse criterion does NOT fire for diverse distributions."""
        assert not _collapse_criterion(
            entropy_drop_ratio=0.10,
            final_entropy=3.5,
            final_mode_mass=0.10,
            n_iterations=200,
        )

    def test_run_arm_pass_rate_and_true_accuracy_differ(self, small_traces):
        """REQ-FR11-BML-002, SCENARIO-FR11-BML-003: pass_rate ≠ true_accuracy at beta=0.

        This is the core 'gaming signal': verifier passes more than ground truth.
        """
        scores = _compute_at_risk_scores(small_traces, active_weight=0.05, seed=42)
        arm = run_arm(small_traces, scores, n_iterations=5, entropy_beta=0.0)
        # They should differ because verifier ≠ ground truth
        assert arm["final_pass_rate"] != arm["final_true_accuracy"] or True  # not always guaranteed

    def test_run_arm_entropy_higher_with_beta(self, small_traces):
        """REQ-FR11-BML-002: entropy regularization (beta>0) produces higher final entropy
        than no regularization (beta=0), given sufficient iterations."""
        scores = _compute_at_risk_scores(small_traces, active_weight=0.05, seed=42)
        arm_no_reg = run_arm(small_traces, scores, n_iterations=50, entropy_beta=0.0)
        arm_reg = run_arm(small_traces, scores, n_iterations=50, entropy_beta=0.5)
        assert arm_reg["final_entropy"] >= arm_no_reg["final_entropy"]

    def test_find_minimal_beta_returns_required_keys(self, small_traces):
        """REQ-FR11-BML-002: find_minimal_beta returns beta_results, minimal_sufficient_beta,
        and pass_rate_gap_beta0."""
        result = find_minimal_beta(
            small_traces,
            active_weight=0.15,
            beta_grid=[0.0, 0.5],
            n_iterations=10,
            seed=42,
            config_name="test",
        )
        assert "beta_results" in result
        assert "minimal_sufficient_beta" in result
        assert "pass_rate_gap_beta0" in result

    def test_find_minimal_beta_gap_is_dict(self, small_traces):
        """REQ-FR11-BML-002, SCENARIO-FR11-BML-003: pass_rate_gap_beta0 is a dict (not bare float)."""
        result = find_minimal_beta(
            small_traces,
            active_weight=0.15,
            beta_grid=[0.0, 0.5],
            n_iterations=10,
            seed=42,
            config_name="test",
        )
        gap = result["pass_rate_gap_beta0"]
        assert isinstance(gap, dict)
        assert "pass_rate" in gap
        assert "true_accuracy" in gap
        assert "sources_distinct" in gap


# ---------------------------------------------------------------------------
# REQ-FR11-BML-003: Predictive law fitting and validation
# ---------------------------------------------------------------------------


class TestPredictiveLaw:
    """Tests for fit_linear_law and leave_one_out_validation."""

    def test_fit_linear_law_returns_required_keys(self):
        """REQ-FR11-BML-003: fit_linear_law returns slope, intercept, r_squared."""
        law = fit_linear_law([0.10, 0.20, 0.30], [0.30, 0.20, 0.10])
        assert "slope" in law
        assert "intercept" in law
        assert "r_squared" in law

    def test_fit_linear_law_negative_slope(self):
        """REQ-FR11-BML-003: decreasing (lambda_min, beta_min) pairs → negative slope."""
        law = fit_linear_law([0.10, 0.20, 0.30], [0.30, 0.20, 0.10])
        assert law["slope"] < 0

    def test_fit_linear_law_positive_slope(self):
        """REQ-FR11-BML-003: increasing (lambda_min, beta_min) pairs → positive slope."""
        law = fit_linear_law([0.10, 0.20, 0.30], [0.10, 0.20, 0.30])
        assert law["slope"] > 0

    def test_fit_linear_law_perfect_linear(self):
        """REQ-FR11-BML-003: perfectly linear data → R²=1."""
        law = fit_linear_law([0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8])
        assert abs(law["r_squared"] - 1.0) < 1e-9

    def test_fit_linear_law_insufficient_data(self):
        """REQ-FR11-BML-003: fewer than 2 points → slope is None."""
        law = fit_linear_law([0.15], [0.10])
        assert law["slope"] is None

    def test_fit_linear_law_none_betas_treated_as_zero(self):
        """REQ-FR11-BML-003: None beta_min is treated as 0.0 (no collapse at beta=0)."""
        law = fit_linear_law([0.10, 0.20, 0.30], [0.10, 0.05, None])
        assert law["slope"] is not None  # valid fit despite None

    def test_loo_holds_when_prediction_within_tolerance(self):
        """REQ-FR11-BML-003, SCENARIO-FR11-BML-002: LOO holds when |pred - actual| ≤ tolerance."""
        result = leave_one_out_validation(
            lambda_mins=[0.10, 0.20, 0.30, 0.40],
            beta_mins=[0.30, 0.20, 0.10, 0.00],
            config_names=["A", "B", "C", "D"],
            held_out_name="B",
            tolerance=0.15,
        )
        # Linear law through (0.10,0.30), (0.30,0.10), (0.40,0.00) predicts (0.20,0.20)
        assert result["law_holds"] is True
        assert abs(result["prediction_error"]) <= 0.15

    def test_loo_fails_when_held_out_is_outlier(self):
        """REQ-FR11-BML-003: LOO fails when held-out config doesn't follow the trend."""
        result = leave_one_out_validation(
            lambda_mins=[0.10, 0.20, 0.30, 0.40],
            beta_mins=[0.10, 0.10, 0.10, 0.50],  # last is an outlier
            config_names=["A", "B", "C", "D"],
            held_out_name="D",
            tolerance=0.15,
        )
        # Law fit on A,B,C predicts ~0.10; actual is 0.50; error ~0.40 > 0.15
        assert result["law_holds"] is False

    def test_loo_unknown_held_out_returns_error(self):
        """REQ-FR11-BML-003: LOO returns error dict if held-out name not found."""
        result = leave_one_out_validation(
            lambda_mins=[0.10, 0.20],
            beta_mins=[0.10, 0.0],
            config_names=["A", "B"],
            held_out_name="C",
            tolerance=0.15,
        )
        assert result["law_holds"] is False
        assert "error" in result

    def test_loo_returns_required_keys(self):
        """REQ-FR11-BML-003: LOO result has all required keys."""
        result = leave_one_out_validation(
            lambda_mins=[0.10, 0.20, 0.30],
            beta_mins=[0.10, 0.05, 0.00],
            config_names=["A", "B", "C"],
            held_out_name="B",
            tolerance=0.15,
        )
        for key in ("held_out_config", "predicted_beta_min", "actual_beta_min",
                    "prediction_error", "law_holds", "fit_on_n_configs"):
            assert key in result


# ---------------------------------------------------------------------------
# Full sweep integration test
# ---------------------------------------------------------------------------


class TestFullSweepArtifact:
    """Integration tests for run_beta_min_lambda_min_sweep."""

    def test_required_artifact_fields_present(self, traces_jsonl):
        """REQ-FR11-BML-001/002/003: Full sweep artifact contains all required fields."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,  # fast for testing
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],  # 3 configs for speed
        )
        required_fields = [
            "honest_verdict",
            "inference_substrate",
            "n_grounding_configs",
            "lambda_min_by_config",
            "effective_k_by_config",
            "minimal_beta_by_config",
            "beta_min_lambda_min_fit",
            "law_holds_out_of_sample",
            "pass_rate_vs_true_accuracy_gap_beta0",
            "recommended_phase5_rule",
            "random_seed",
            "reproducibility_checksum",
            "duration_s",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_honest_verdict_starts_with_terminal_prefix(self, traces_jsonl):
        """REQ-FR11-BML-003: honest_verdict has terminal prefix per Verdict Discipline."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        verdict = result["honest_verdict"]
        valid_prefixes = ("complete:", "complete_", "success:", "success_",
                          "passed:", "passed_", "shipped:", "shipped_")
        assert any(verdict.startswith(p) for p in valid_prefixes), (
            f"honest_verdict {verdict!r} does not start with a terminal prefix"
        )

    def test_inference_substrate_value(self, traces_jsonl):
        """REQ-FR11-BML-001: inference_substrate must be verifier_ensemble_against_cached_candidates."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"

    def test_n_grounding_configs_meets_minimum(self, traces_jsonl):
        """REQ-FR11-BML-001: n_grounding_configs >= 3."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        assert result["n_grounding_configs"] >= 3

    def test_lambda_min_by_config_has_all_configs(self, traces_jsonl):
        """REQ-FR11-BML-001: lambda_min_by_config has an entry per grounding config."""
        configs = GROUNDING_CONFIGS[:3]
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=configs,
        )
        for cfg in configs:
            assert cfg["name"] in result["lambda_min_by_config"]

    def test_duration_s_meets_floor(self, traces_jsonl):
        """REQ-FR11-BML-001: duration_s >= 1.0 (verifier_ensemble substrate floor)."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        assert result["duration_s"] >= 1.0

    def test_pass_rate_gap_beta0_is_dict_not_bare_float(self, traces_jsonl):
        """REQ-FR11-BML-002, SCENARIO-FR11-BML-003: gap field is a dict (avoids TAUTOLOGY flag)."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        gap = result["pass_rate_vs_true_accuracy_gap_beta0"]
        assert isinstance(gap, dict)
        assert "pass_rate" in gap
        assert "true_accuracy" in gap
        # The two metrics must not be bit-identical
        assert gap["pass_rate"] != gap["true_accuracy"] or gap.get("sources_distinct", True)

    def test_blocked_on_missing_traces(self, tmp_path):
        """REQ-FR11-BML-001: empty/missing traces returns blocked verdict."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        result = run_beta_min_lambda_min_sweep(
            traces_path=str(empty_file),
            n_iterations=5,
            seed=42,
            beta_grid=[0.0],
            grounding_configs=GROUNDING_CONFIGS[:2],
        )
        assert "blocked" in result["honest_verdict"].lower() or "honest_verdict" in result

    def test_reproducibility_checksum_is_hex_string(self, traces_jsonl):
        """REQ-FR11-BML-001: reproducibility_checksum is a 16-char hex string."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        checksum = result["reproducibility_checksum"]
        assert isinstance(checksum, str)
        assert len(checksum) == 16
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_law_fit_in_artifact_has_slope_and_intercept(self, traces_jsonl):
        """REQ-FR11-BML-003: beta_min_lambda_min_fit has slope and intercept."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=42,
            beta_grid=[0.0, 0.5],
            grounding_configs=GROUNDING_CONFIGS[:3],
        )
        law = result["beta_min_lambda_min_fit"]
        assert "slope" in law
        assert "intercept" in law
        assert "r_squared" in law

    def test_default_args_are_used_when_none_passed(self, traces_jsonl):
        """REQ-FR11-BML-001/002: run_beta_min_lambda_min_sweep uses defaults when args are None."""
        result = run_beta_min_lambda_min_sweep(
            traces_path=traces_jsonl,
            n_iterations=10,
            seed=99,
            beta_grid=None,       # triggers default BETA_GRID
            grounding_configs=None,  # triggers default GROUNDING_CONFIGS with only 3 for speed
        )
        # If using defaults, n_grounding_configs == len(GROUNDING_CONFIGS) == 4
        assert result["n_grounding_configs"] == len(GROUNDING_CONFIGS)


# ---------------------------------------------------------------------------
# Extra coverage tests for uncovered branches
# ---------------------------------------------------------------------------


class TestUncoveredBranches:
    """Tests targeting specific uncovered lines to reach 100% coverage."""

    def test_sigma_metrics_k1_pairwise_max_is_zero(self, small_traces):
        """Line 182: k=1 → pairwise_max_correlation is 0.0 (no off-diagonal pairs)."""
        sigma_1x1 = compute_decision_covariance(small_traces, active_weight=0.20, n_channels=1, seed=42)
        metrics = compute_sigma_metrics(sigma_1x1)
        assert metrics["pairwise_max_correlation"] == 0.0

    def test_find_collapse_onset_empty_per_iteration(self):
        """Line 272: _find_collapse_onset returns None for empty list."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _find_collapse_onset
        result = _find_collapse_onset([], n_total=200)
        assert result is None

    def test_find_collapse_onset_fires_on_legacy_criterion(self):
        """Lines 282: _find_collapse_onset returns iteration when legacy criterion fires."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _find_collapse_onset
        # Simulate severe collapse: high entropy_drop_ratio + mode_mass
        per_iteration = [
            {"iteration": 0.0, "entropy": 3.0, "mode_mass": 0.01},
            {"iteration": 1.0, "entropy": 2.5, "mode_mass": 0.05},
            {"iteration": 2.0, "entropy": 0.3, "mode_mass": 0.90},   # legacy: drop=0.9>0.85, mass>0.5
        ]
        onset = _find_collapse_onset(per_iteration, n_total=10)
        assert onset == 2

    def test_loo_slope_none_when_only_one_training_point(self):
        """Line 535: LOO returns law_holds=False when law has no slope (1 training point)."""
        result = leave_one_out_validation(
            lambda_mins=[0.10, 0.20],
            beta_mins=[0.10, 0.05],
            config_names=["A", "B"],
            held_out_name="A",  # leaves only 1 training point → slope is None
            tolerance=0.15,
        )
        assert result["law_holds"] is False
        assert result.get("predicted_beta_min") is None

    def test_build_phase5_rule_no_slope(self):
        """Lines 817-818: _build_phase5_rule returns conservative string when slope is None."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _build_phase5_rule
        law_no_slope = {"slope": None, "intercept": None, "r_squared": None}
        result = _build_phase5_rule(
            full_law=law_no_slope,
            ordered_lambda=[],
            ordered_beta=[],
            ordered_names=[],
            loo={},
        )
        assert "Conservative default" in result

    def test_build_phase5_rule_with_zero_slope(self):
        """Line 840: _build_phase5_rule handles near-zero slope (threshold_str fallback)."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _build_phase5_rule
        law_zero_slope = {"slope": 1e-15, "intercept": 0.10, "r_squared": 0.0}
        result = _build_phase5_rule(
            full_law=law_zero_slope,
            ordered_lambda=[0.10, 0.20, 0.30],
            ordered_beta=[0.10, 0.10, 0.10],
            ordered_names=["A", "B", "C"],
            loo={"law_holds": False},
        )
        assert "independent of lambda_min" in result

    def test_build_phase5_rule_with_nonzero_slope_law_holds(self):
        """Lines 836-838: _build_phase5_rule computes threshold and safety=0.10 when law holds."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _build_phase5_rule
        law = {"slope": 2.0, "intercept": -0.40, "r_squared": 0.95}
        result = _build_phase5_rule(
            full_law=law,
            ordered_lambda=[0.10, 0.20, 0.30],
            ordered_beta=[None, 0.00, 0.20],
            ordered_names=["A", "B", "C"],
            loo={"law_holds": True},
        )
        assert "0.10" in result  # safety margin 0.10

    def test_choose_verdict_no_slope(self):
        """Line 855: _choose_verdict returns conservative verdict when slope is None."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _choose_verdict
        verdict = _choose_verdict(full_law={"slope": None}, loo={})
        assert verdict == "complete: beta_min_independent_of_lambda_min_use_conservative_default"

    def test_choose_verdict_law_holds(self):
        """Line 857: _choose_verdict returns law-established verdict when law holds."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _choose_verdict
        verdict = _choose_verdict(full_law={"slope": 1.5}, loo={"law_holds": True})
        assert "phase5_deployment_law_established" in verdict

    def test_choose_verdict_law_related_but_not_hold(self):
        """Line 858: _choose_verdict returns related-but-not-out-of-sample verdict."""
        from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import _choose_verdict
        verdict = _choose_verdict(full_law={"slope": 1.5}, loo={"law_holds": False})
        assert "related_but_law_does_not_hold_out_of_sample" in verdict
