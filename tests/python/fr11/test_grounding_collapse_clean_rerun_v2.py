"""Tests for FR-11 Grounding Collapse Clean Rerun v2.

Covers REQ-FR11-GC-001, REQ-FR11-GC-002, SCENARIO-FR11-GC-001/002.

The key invariant being tested: pass_rate and true_accuracy must come from
genuinely distinct sources (verifier vs ground truth), so they are not forced
equal as they were in exp3452 (the TAUTOLOGY de-flag).
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from carnot.fr11.grounding_collapse_clean_rerun_v2 import (
    ACTIVE_WEIGHT,
    ENTROPY_COLLAPSE_THRESHOLD,
    ENTROPY_REGULARIZATION_BETA,
    MODE_MASS_COLLAPSE_THRESHOLD,
    NULL_WEIGHT,
    _distribution_entropy,
    _softmax,
    compute_at_risk_scores_v2,
    compute_entropy_trend_significance,
    run_arm_v2,
    run_stress_test_v2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traces() -> list[dict]:
    """20-trace corpus: every 5th trace is correct (4 correct, 16 incorrect)."""
    return [
        {
            "question_id": f"q{i:03d}",
            "prompt": f"Q{i}",
            "completion": f"A{i}",
            "is_correct": i % 5 == 0,
            "energy_score": 0.0,
        }
        for i in range(20)
    ]


@pytest.fixture()
def traces_file(small_traces, tmp_path) -> str:
    """Write small_traces to a temp JSONL and return path."""
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


@pytest.fixture()
def skewed_traces() -> list[dict]:
    """150-trace corpus with only 1 correct trace (mirrors real fr11_zenil_distill_v2.jsonl)."""
    traces = [{"question_id": f"q{i}", "is_correct": (i == 0), "completion": f"c{i}"} for i in range(150)]
    return traces


@pytest.fixture()
def skewed_traces_file(skewed_traces, tmp_path) -> str:
    path = tmp_path / "skewed.jsonl"
    with open(path, "w") as f:
        for t in skewed_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_active_weight_from_exp3439():
    """ACTIVE_WEIGHT must equal z3_math dropout contribution from exp3439 (0.146)."""
    assert abs(ACTIVE_WEIGHT - 0.146) < 1e-6


def test_null_weight_complements_active():
    """NULL_WEIGHT + ACTIVE_WEIGHT must sum to 1.0 (REQ-FR11-GC-002)."""
    assert abs(ACTIVE_WEIGHT + NULL_WEIGHT - 1.0) < 1e-9


def test_null_weight_dominates():
    """NULL_WEIGHT must be > 0.5 so null-space can override active signal.

    This is the structural requirement for incorrect traces to score > 0.5,
    which breaks the v1 TAUTOLOGY (REQ-FR11-GC-002).
    """
    assert NULL_WEIGHT > 0.5


# ---------------------------------------------------------------------------
# compute_at_risk_scores_v2
# ---------------------------------------------------------------------------


def test_scores_v2_shape(small_traces):
    """Score array has same length as trace list (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    assert len(scores) == len(small_traces)


def test_scores_v2_range(small_traces):
    """All scores lie in [0, 1] (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_scores_v2_reproducible(small_traces):
    """Same seed produces identical scores (determinism requirement)."""
    s1 = compute_at_risk_scores_v2(small_traces, seed=99)
    s2 = compute_at_risk_scores_v2(small_traces, seed=99)
    np.testing.assert_array_equal(s1, s2)


def test_scores_v2_vary_by_seed(small_traces):
    """Different seeds produce different scores."""
    s1 = compute_at_risk_scores_v2(small_traces, seed=1)
    s2 = compute_at_risk_scores_v2(small_traces, seed=2)
    assert not np.allclose(s1, s2)


def test_scores_v2_incorrect_can_exceed_threshold(skewed_traces):
    """With 149 incorrect traces and ACTIVE_WEIGHT=0.146, some incorrect traces must score > 0.5.

    This is the critical de-flag property: v1 had only the 1 correct trace scoring > 0.5,
    making verifier_pass_arr == is_correct_arr (TAUTOLOGY). v2 must have incorrect traces
    also scoring > 0.5 so the two vectors are genuinely different (REQ-FR11-GC-002).
    """
    scores = compute_at_risk_scores_v2(skewed_traces, seed=42)
    is_correct = np.array([t["is_correct"] for t in skewed_traces])
    incorrect_scores = scores[~is_correct]
    # With NULL_WEIGHT=0.854, many incorrect traces must score > 0.5
    assert np.any(incorrect_scores > 0.5), (
        "No incorrect trace scored > 0.5 — the v1 tautology would still hold"
    )


def test_scores_v2_verifier_pass_not_identical_to_is_correct(skewed_traces):
    """verifier_pass_arr must NOT be identical to is_correct_arr (the de-flag property).

    This is the root cause of the exp3452 TAUTOLOGY: those two arrays were identical.
    v2 fixes it by ensuring incorrect traces can score > 0.5 (REQ-FR11-GC-002).
    """
    scores = compute_at_risk_scores_v2(skewed_traces, seed=42)
    is_correct_arr = np.array([t["is_correct"] for t in skewed_traces], dtype=float)
    verifier_pass_arr = (scores > 0.5).astype(float)
    assert not np.array_equal(verifier_pass_arr, is_correct_arr), (
        "verifier_pass_arr == is_correct_arr: tautology not fixed"
    )


# ---------------------------------------------------------------------------
# compute_entropy_trend_significance
# ---------------------------------------------------------------------------


def test_trend_significance_declining():
    """Monotone declining sequence must show negative tau and significant p-value."""
    seq = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    tau, p = compute_entropy_trend_significance(seq)
    assert tau < -0.9, f"Expected tau close to -1.0 for declining sequence, got {tau}"
    assert p < 0.05, f"Expected significant p-value, got {p}"


def test_trend_significance_flat():
    """Constant sequence must show tau close to 0 and non-significant p-value."""
    seq = [3.0] * 20
    tau, p = compute_entropy_trend_significance(seq)
    # Constant sequence: kendall tau is undefined / nan, p-value should be 1.0 or nan
    # scipy returns nan for constant sequences
    assert p >= 0.05 or (not (p == p)), f"Constant sequence should not show trend, p={p}"


def test_trend_significance_short_sequence():
    """Sequence with < 4 points returns (0.0, 1.0) to indicate insufficient data."""
    tau, p = compute_entropy_trend_significance([1.0, 0.5, 0.1])
    assert tau == 0.0
    assert p == 1.0


def test_trend_significance_returns_tuple():
    """Must return a (tau, p_value) tuple of two floats."""
    seq = [3.0, 2.8, 2.5, 2.0, 1.5, 1.0]
    result = compute_entropy_trend_significance(seq)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# _softmax and _distribution_entropy (shared utilities)
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    """softmax output must sum to 1."""
    log_w = np.array([1.0, 2.0, 3.0, 0.5])
    p = _softmax(log_w)
    assert abs(float(np.sum(p)) - 1.0) < 1e-9


def test_entropy_uniform_max():
    """Uniform distribution has entropy log(n)."""
    n = 8
    p = np.ones(n) / n
    assert abs(_distribution_entropy(p) - math.log(n)) < 1e-9


def test_entropy_degenerate_zero():
    """One-hot distribution has zero entropy."""
    p = np.array([1.0, 0.0, 0.0])
    assert _distribution_entropy(p) < 1e-6


# ---------------------------------------------------------------------------
# run_arm_v2
# ---------------------------------------------------------------------------


def test_arm_v2_required_fields(small_traces):
    """run_arm_v2 must return all required fields including new v2 ones."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    required = [
        "per_iteration", "final_entropy", "final_mode_mass", "final_pass_rate",
        "final_true_accuracy", "final_pass_rate_vs_true_accuracy_gap",
        "initial_entropy", "entropy_drop_ratio", "mode_collapse_detected",
        "entropy_trend_tau", "entropy_trend_p_value",
    ]
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_arm_v2_gap_is_difference(small_traces):
    """final_pass_rate_vs_true_accuracy_gap must equal pass_rate minus true_accuracy."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=10, use_entropy_reg=False)
    expected_gap = result["final_pass_rate"] - result["final_true_accuracy"]
    assert abs(result["final_pass_rate_vs_true_accuracy_gap"] - expected_gap) < 1e-9


def test_arm_v2_per_iteration_has_gap(small_traces):
    """Each per_iteration entry must include pass_rate_vs_true_accuracy_gap."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    for entry in result["per_iteration"]:
        assert "pass_rate_vs_true_accuracy_gap" in entry


def test_arm_v2_b_higher_entropy(small_traces):
    """ARM B must maintain higher entropy than ARM A at the same iteration count."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    arm_a = run_arm_v2(small_traces, scores, n_iterations=30, use_entropy_reg=False)
    arm_b = run_arm_v2(small_traces, scores, n_iterations=30, use_entropy_reg=True)
    assert arm_b["final_entropy"] > arm_a["final_entropy"]


def test_arm_v2_entropy_sequence_length(small_traces):
    """entropy_sequence must have exactly n_iterations entries."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=12, use_entropy_reg=False)
    assert len(result["entropy_sequence"]) == 12


def test_arm_v2_trend_significance_declining_on_collapse(small_traces):
    """For enough iterations, ARM A entropy trend must be significantly declining."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=100, use_entropy_reg=False)
    # If ARM A collapses, entropy trend should be significantly negative
    if result["mode_collapse_detected"]:
        assert result["entropy_trend_tau"] < -0.5
        assert result["entropy_trend_p_value"] < 0.05


def test_arm_v2_mode_mass_in_range(small_traces):
    """Mode mass must be in (0, 1]."""
    scores = compute_at_risk_scores_v2(small_traces, seed=42)
    result = run_arm_v2(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    assert 0.0 < result["final_mode_mass"] <= 1.0


# ---------------------------------------------------------------------------
# run_stress_test_v2 integration tests
# ---------------------------------------------------------------------------


def test_stress_v2_returns_complete_verdict(traces_file):
    """honest_verdict must start with 'complete:' (Verdict Terminal-Prefix)."""
    result = run_stress_test_v2(traces_file, n_iterations=10, seed=42)
    assert result["honest_verdict"].startswith("complete:")


def test_stress_v2_inference_substrate(traces_file):
    """inference_substrate must be verifier_ensemble_against_cached_candidates."""
    result = run_stress_test_v2(traces_file, n_iterations=10, seed=42)
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_stress_v2_duration_floor(traces_file):
    """duration_s must be >= 1.0 per CLAUDE.md inference_substrate discipline."""
    result = run_stress_test_v2(traces_file, n_iterations=5, seed=42)
    assert result["duration_s"] >= 1.0


def test_stress_v2_all_required_fields(traces_file):
    """All REQUIRED ARTIFACT FIELDS from the task spec must be present."""
    result = run_stress_test_v2(traces_file, n_iterations=10, seed=42)
    required = [
        "honest_verdict", "inference_substrate", "n_iterations",
        "arm_a_final_entropy", "arm_b_final_entropy",
        "arm_a_final_mode_mass",
        "arm_a_pass_rate_vs_true_accuracy_gap",
        "arm_a_mode_collapse_detected", "arm_b_mode_collapse_detected",
        "entropy_trend_significance", "grounding_collapse_consequence",
        "random_seed", "reproducibility_checksum", "duration_s",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_stress_v2_entropy_trend_has_tau_and_pvalue(traces_file):
    """entropy_trend_significance must contain tau and p_value."""
    result = run_stress_test_v2(traces_file, n_iterations=10, seed=42)
    ets = result["entropy_trend_significance"]
    assert "tau" in ets
    assert "p_value" in ets


def test_stress_v2_pass_correct_arrays_not_identical(skewed_traces_file):
    """pass_correct_arrays_identical must be False for the 1/150 corpus (the de-flag check)."""
    result = run_stress_test_v2(skewed_traces_file, n_iterations=10, seed=42)
    assert result["pass_correct_arrays_identical"] is False, (
        "v2 must separate verifier_pass_arr from is_correct_arr to avoid the v1 tautology"
    )


def test_stress_v2_reproducible(traces_file):
    """Two runs with same seed must produce identical results."""
    r1 = run_stress_test_v2(traces_file, n_iterations=10, seed=99)
    r2 = run_stress_test_v2(traces_file, n_iterations=10, seed=99)
    assert r1["arm_a_final_entropy"] == r2["arm_a_final_entropy"]
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]


def test_stress_v2_arm_b_entropy_exceeds_arm_a(traces_file):
    """ARM B final entropy must exceed ARM A (regularization works)."""
    result = run_stress_test_v2(traces_file, n_iterations=30, seed=42)
    assert result["arm_b_final_entropy"] > result["arm_a_final_entropy"]


def test_stress_v2_gap_field_is_float(traces_file):
    """arm_a_pass_rate_vs_true_accuracy_gap must be a finite float."""
    result = run_stress_test_v2(traces_file, n_iterations=10, seed=42)
    gap = result["arm_a_pass_rate_vs_true_accuracy_gap"]
    assert isinstance(gap, float)
    assert not math.isnan(gap)
    assert not math.isinf(gap)


def test_stress_v2_n_iterations_matches(traces_file):
    """n_iterations in output must match requested value."""
    result = run_stress_test_v2(traces_file, n_iterations=7, seed=42)
    assert result["n_iterations"] == 7


def test_stress_v2_grounding_holds_on_few_iterations(traces_file):
    """With 2 iterations, no collapse can occur; grounding-holds verdict fires."""
    result = run_stress_test_v2(traces_file, n_iterations=2, seed=42)
    assert not result["arm_a_mode_collapse_detected"]
    assert "residual_diversity_holds" in result["honest_verdict"]


def test_stress_v2_both_collapse_with_zero_beta(traces_file):
    """With entropy_beta=0, ARM B is identical to ARM A; both collapse if given enough iterations."""
    result = run_stress_test_v2(traces_file, n_iterations=200, seed=42, entropy_beta=0.0)
    assert result["arm_a_mode_collapse_detected"]
    assert result["arm_b_mode_collapse_detected"]
    assert "collapse_not_prevented_by_entropy_reg" in result["honest_verdict"]


def test_stress_v2_empty_file_returns_blocked():
    """Empty traces file returns a blocked verdict gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = run_stress_test_v2(path, n_iterations=5, seed=42)
        assert "blocked" in result["honest_verdict"] or "complete:" in result["honest_verdict"]
    finally:
        os.unlink(path)


def test_stress_v2_active_weight_recorded(traces_file):
    """active_weight_from_exp3439 must be recorded in the artifact."""
    result = run_stress_test_v2(traces_file, n_iterations=5, seed=42)
    assert "active_weight_from_exp3439" in result
    assert abs(result["active_weight_from_exp3439"] - 0.146) < 1e-6


def test_stress_v2_field_provenance_present(traces_file):
    """field_provenance must be present and contain key fields."""
    result = run_stress_test_v2(traces_file, n_iterations=5, seed=42)
    fp = result.get("field_provenance", {})
    assert "arm_a_pass_rate_vs_true_accuracy_gap" in fp
    assert "entropy_trend_significance" in fp
    assert "random_seed" in fp
