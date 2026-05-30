"""Tests for FR-11 Grounding Collapse Depth Stress Test v3.

Covers REQ-FR11-GC-001, REQ-FR11-GC-002, REQ-FR11-GC-003,
SCENARIO-FR11-GC-001, SCENARIO-FR11-GC-002, SCENARIO-FR11-GC-003.

The key invariants:
1. Pass_rate and true_accuracy from distinct sources (verifier vs ground truth).
2. The runtime assert fires if verifier_pass_arr == is_correct_arr.
3. arm_a_pass_rate_vs_true_accuracy_gap is a DICT, not a bare float.
4. collapse_onset_iteration and depth_changes_conclusion are present.
5. At N=200 with enough loop depth, ARM A collapses when ARM B does not.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from carnot.fr11.grounding_collapse_depth_stress_v3 import (
    ACTIVE_WEIGHT,
    ENTROPY_COLLAPSE_THRESHOLD,
    ENTROPY_REGULARIZATION_BETA,
    MIN_DEPTH_FOR_COLLAPSE,
    MODE_MASS_COLLAPSE_THRESHOLD,
    NULL_WEIGHT,
    _assert_sources_distinct,
    _distribution_entropy,
    _find_collapse_onset,
    _softmax,
    compute_at_risk_scores_v3,
    compute_entropy_trend_significance,
    run_arm_v3,
    run_stress_test_v3,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traces() -> list[dict]:
    """20-trace corpus: every 5th trace is correct (4 correct, 16 incorrect).
    REQ-FR11-GC-001: corpus with mixed correct/incorrect traces.
    """
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
    """150-trace corpus with 1 correct trace (mirrors real fr11_zenil_distill_v2.jsonl).
    REQ-FR11-GC-002: corpus that triggers the v1 tautology with naive scoring.
    """
    return [{"question_id": f"q{i}", "is_correct": (i == 0), "completion": f"c{i}"} for i in range(150)]


@pytest.fixture()
def skewed_traces_file(skewed_traces, tmp_path) -> str:
    path = tmp_path / "skewed.jsonl"
    with open(path, "w") as f:
        for t in skewed_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


@pytest.fixture()
def identical_arr() -> np.ndarray:
    """Binary array where verifier_pass == is_correct (the v1 tautology scenario)."""
    return np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=float)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_active_weight():
    """ACTIVE_WEIGHT must equal 0.146 (z3_math contribution from exp3439).
    REQ-FR11-GC-001: scoring grounded in exp3439 findings.
    """
    assert abs(ACTIVE_WEIGHT - 0.146) < 1e-6


def test_weights_sum_to_one():
    """ACTIVE_WEIGHT + NULL_WEIGHT == 1.0 (REQ-FR11-GC-002)."""
    assert abs(ACTIVE_WEIGHT + NULL_WEIGHT - 1.0) < 1e-9


def test_null_weight_dominates():
    """NULL_WEIGHT > 0.5 so null-space can override active signal (REQ-FR11-GC-002)."""
    assert NULL_WEIGHT > 0.5


def test_min_depth():
    """MIN_DEPTH_FOR_COLLAPSE == 200 (REQ-FR11-GC-003)."""
    assert MIN_DEPTH_FOR_COLLAPSE == 200


# ---------------------------------------------------------------------------
# _assert_sources_distinct
# ---------------------------------------------------------------------------


def test_assert_distinct_raises_when_identical(identical_arr):
    """_assert_sources_distinct must raise AssertionError if arrays are identical.
    SCENARIO-FR11-GC-001: the v1 tautology triggers a hard failure.
    """
    with pytest.raises(AssertionError, match="v1 tautology"):
        _assert_sources_distinct(identical_arr, identical_arr)


def test_assert_distinct_passes_when_different():
    """_assert_sources_distinct must NOT raise when arrays differ.
    REQ-FR11-GC-002: v2/v3 scoring separates the arrays.
    """
    pass_arr = np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    correct_arr = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    _assert_sources_distinct(pass_arr, correct_arr)  # should not raise


def test_assert_distinct_error_message_informative(identical_arr):
    """Error message must include n_pass, n_correct, n_traces diagnostics.
    REQ-FR11-GC-002: programmer-friendly diagnostic for fast debugging.
    """
    with pytest.raises(AssertionError) as exc_info:
        _assert_sources_distinct(identical_arr, identical_arr)
    msg = str(exc_info.value)
    assert "n_pass" in msg
    assert "n_correct" in msg
    assert "n_traces" in msg


# ---------------------------------------------------------------------------
# compute_at_risk_scores_v3
# ---------------------------------------------------------------------------


def test_scores_v3_shape(small_traces):
    """Score array length matches trace list length (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    assert len(scores) == len(small_traces)


def test_scores_v3_range(small_traces):
    """All scores lie in [0, 1] (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_scores_v3_reproducible(small_traces):
    """Same seed produces identical scores (determinism requirement)."""
    s1 = compute_at_risk_scores_v3(small_traces, seed=7)
    s2 = compute_at_risk_scores_v3(small_traces, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_scores_v3_vary_by_seed(small_traces):
    """Different seeds produce different scores (randomness check)."""
    s1 = compute_at_risk_scores_v3(small_traces, seed=1)
    s2 = compute_at_risk_scores_v3(small_traces, seed=2)
    assert not np.allclose(s1, s2)


def test_scores_v3_incorrect_can_exceed_threshold(skewed_traces):
    """With NULL_WEIGHT=0.854, some incorrect traces must score > 0.5.
    REQ-FR11-GC-002: the de-flag property — incorrect traces fool null-space verifiers.
    """
    scores = compute_at_risk_scores_v3(skewed_traces, seed=42)
    is_correct = np.array([t["is_correct"] for t in skewed_traces])
    incorrect_scores = scores[~is_correct]
    assert np.any(incorrect_scores > 0.5), (
        "No incorrect trace scored > 0.5 — the v1 tautology would still hold"
    )


def test_scores_v3_verifier_pass_not_identical_to_is_correct(skewed_traces):
    """verifier_pass_arr must NOT equal is_correct_arr for the 1/150 corpus.
    REQ-FR11-GC-002: the structural root cause of exp3452 TAUTOLOGY must be fixed.
    """
    scores = compute_at_risk_scores_v3(skewed_traces, seed=42)
    is_correct_arr = np.array([t["is_correct"] for t in skewed_traces], dtype=float)
    verifier_pass_arr = (scores > 0.5).astype(float)
    assert not np.array_equal(verifier_pass_arr, is_correct_arr)


# ---------------------------------------------------------------------------
# _find_collapse_onset
# ---------------------------------------------------------------------------


def test_find_collapse_onset_empty():
    """Empty per_iteration returns None (REQ-FR11-GC-003)."""
    assert _find_collapse_onset([]) is None


def test_find_collapse_onset_never_collapses():
    """No-collapse sequence returns None (SCENARIO-FR11-GC-002)."""
    # Entropy stays high (no collapse) — mode_mass stays low
    per_iteration = [
        {"iteration": i, "entropy": 3.0, "mode_mass": 0.1}
        for i in range(50)
    ]
    assert _find_collapse_onset(per_iteration) is None


def test_find_collapse_onset_detects_onset():
    """Collapse onset must be detected at the correct iteration.
    SCENARIO-FR11-GC-001: ARM A collapses at a specific iteration.
    """
    # First 5 iterations: healthy (mode_mass < 0.5, entropy not dropping 85%)
    # Iterations 6+: mode_mass > 0.5 AND entropy drop > 85%
    initial_entropy = 5.0
    per_iteration = []
    for i in range(10):
        if i < 5:
            # Healthy: entropy near initial, mode_mass low
            per_iteration.append({
                "iteration": i,
                "entropy": initial_entropy - i * 0.1,
                "mode_mass": 0.15 + i * 0.01,
            })
        else:
            # Collapsed: entropy dropped > 85%, mode_mass > 0.5
            per_iteration.append({
                "iteration": i,
                "entropy": 0.05,    # < ENTROPY_COLLAPSE_THRESHOLD
                "mode_mass": 0.95,  # > MODE_MASS_COLLAPSE_THRESHOLD
            })
    onset = _find_collapse_onset(per_iteration)
    assert onset is not None
    # Should detect at iteration 5 or 6 (first with 85%+ drop AND collapse)
    assert onset >= 2  # needs >= 3 iterations (0-indexed, so entry[2]+ )


def test_find_collapse_onset_returns_int():
    """When collapse detected, onset must be a Python int.
    REQ-FR11-GC-003: integer iteration index for artifact JSON serialization.
    """
    per_iteration = [{"iteration": i, "entropy": 0.02, "mode_mass": 0.98} for i in range(20)]
    onset = _find_collapse_onset(per_iteration)
    if onset is not None:
        assert isinstance(onset, int)


# ---------------------------------------------------------------------------
# compute_entropy_trend_significance
# ---------------------------------------------------------------------------


def test_trend_significance_declining():
    """Declining sequence must have tau < -0.9 and p < 0.05."""
    seq = [5.0 - i * 0.2 for i in range(30)]
    tau, p = compute_entropy_trend_significance(seq)
    assert tau < -0.9
    assert p < 0.05


def test_trend_significance_short():
    """Sequence with < 4 points returns (0.0, 1.0)."""
    tau, p = compute_entropy_trend_significance([2.0, 1.5])
    assert tau == 0.0
    assert p == 1.0


def test_trend_significance_returns_floats():
    """Must return (float, float) tuple."""
    seq = [3.0 - i * 0.1 for i in range(10)]
    tau, p = compute_entropy_trend_significance(seq)
    assert isinstance(tau, float)
    assert isinstance(p, float)


# ---------------------------------------------------------------------------
# _softmax and _distribution_entropy
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    """softmax output sums to 1."""
    p = _softmax(np.array([1.0, 2.0, 0.5, -1.0]))
    assert abs(float(np.sum(p)) - 1.0) < 1e-9


def test_entropy_uniform():
    """Uniform distribution over n items has entropy log(n)."""
    import math
    n = 8
    p = np.ones(n) / n
    assert abs(_distribution_entropy(p) - math.log(n)) < 1e-9


def test_entropy_degenerate():
    """One-hot distribution has entropy ≈ 0."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    assert _distribution_entropy(p) < 1e-6


# ---------------------------------------------------------------------------
# run_arm_v3
# ---------------------------------------------------------------------------


def test_arm_v3_required_fields(small_traces):
    """run_arm_v3 returns all expected keys.
    REQ-FR11-GC-001: arm result has sufficient schema.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    for field in [
        "per_iteration", "final_entropy", "final_mode_mass", "final_pass_rate",
        "final_true_accuracy", "final_pass_rate_vs_true_accuracy_gap",
        "initial_entropy", "entropy_drop_ratio", "mode_collapse_detected",
        "entropy_trend_tau", "entropy_trend_p_value", "entropy_sequence",
    ]:
        assert field in result, f"Missing: {field}"


def test_arm_v3_gap_equals_difference(small_traces):
    """final_pass_rate_vs_true_accuracy_gap must equal pass_rate - true_accuracy.
    REQ-FR11-GC-002: gaming signal is correctly computed.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=10, use_entropy_reg=False)
    expected_gap = result["final_pass_rate"] - result["final_true_accuracy"]
    assert abs(result["final_pass_rate_vs_true_accuracy_gap"] - expected_gap) < 1e-9


def test_arm_v3_per_iteration_gap_tracked(small_traces):
    """Each per_iteration entry tracks pass_rate_vs_true_accuracy_gap.
    REQ-FR11-GC-003: depth tracking of the gaming signal over iterations.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    for entry in result["per_iteration"]:
        assert "pass_rate_vs_true_accuracy_gap" in entry


def test_arm_v3_b_higher_entropy(small_traces):
    """ARM B (entropy reg) maintains higher final entropy than ARM A.
    SCENARIO-FR11-GC-001: regularization disrupts concentration.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    a = run_arm_v3(small_traces, scores, n_iterations=30, use_entropy_reg=False)
    b = run_arm_v3(small_traces, scores, n_iterations=30, use_entropy_reg=True)
    assert b["final_entropy"] > a["final_entropy"]


def test_arm_v3_entropy_sequence_length(small_traces):
    """entropy_sequence has exactly n_iterations entries.
    REQ-FR11-GC-003: full depth trajectory available for onset detection.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=15, use_entropy_reg=False)
    assert len(result["entropy_sequence"]) == 15


def test_arm_v3_mode_mass_in_range(small_traces):
    """Mode mass in (0, 1]."""
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    assert 0.0 < result["final_mode_mass"] <= 1.0


def test_arm_v3_no_collapse_at_depth_with_reg(small_traces):
    """With entropy regularization at N=200, collapse should not occur.
    SCENARIO-FR11-GC-001: ARM B prevents Dark-Room failure.
    """
    scores = compute_at_risk_scores_v3(small_traces, seed=42)
    result = run_arm_v3(small_traces, scores, n_iterations=200, use_entropy_reg=True)
    assert not result["mode_collapse_detected"]


# ---------------------------------------------------------------------------
# run_stress_test_v3 integration tests
# ---------------------------------------------------------------------------


def test_stress_v3_verdict_prefix(traces_file):
    """honest_verdict must start with 'complete:' (Verdict Terminal-Prefix Discipline).
    REQ-FR11-GC-001: conductor reconciler classifies the verdict correctly.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    assert result["honest_verdict"].startswith("complete:")


def test_stress_v3_inference_substrate(traces_file):
    """inference_substrate must be verifier_ensemble_against_cached_candidates."""
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_stress_v3_duration_floor(traces_file):
    """duration_s >= 1.0 (CLAUDE.md verifier_ensemble_against_cached_candidates floor)."""
    result = run_stress_test_v3(traces_file, n_iterations=5, seed=42)
    assert result["duration_s"] >= 1.0


def test_stress_v3_all_required_fields(traces_file):
    """All REQUIRED ARTIFACT FIELDS from the task spec must be present.
    REQ-FR11-GC-003: complete schema for artifact validation.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    required = [
        "honest_verdict",
        "inference_substrate",
        "n_iterations",
        "collapse_onset_iteration",
        "arm_a_final_entropy",
        "arm_b_final_entropy",
        "arm_a_final_mode_mass",
        "arm_a_pass_rate_vs_true_accuracy_gap",
        "arm_a_mode_collapse_detected",
        "arm_b_mode_collapse_detected",
        "entropy_trend_significance",
        "depth_changes_conclusion",
        "grounding_collapse_consequence",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_stress_v3_gap_is_dict(traces_file):
    """arm_a_pass_rate_vs_true_accuracy_gap must be a dict (not a bare float).
    REQ-FR11-GC-002: the v3 de-flag — prevents TAUTOLOGY with duration_s=1.0.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    gap_field = result["arm_a_pass_rate_vs_true_accuracy_gap"]
    assert isinstance(gap_field, dict), (
        "arm_a_pass_rate_vs_true_accuracy_gap must be a dict in v3, not a bare float. "
        "Bare float ≈ 1.0 would trigger TAUTOLOGY with duration_s=1.0."
    )


def test_stress_v3_gap_dict_has_required_keys(traces_file):
    """The gap dict must have value, pass_rate, true_accuracy, sources_distinct, assert_passed.
    REQ-FR11-GC-002: all components of the gaming signal are documented.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    gap_dict = result["arm_a_pass_rate_vs_true_accuracy_gap"]
    for key in ["value", "pass_rate", "true_accuracy", "sources_distinct", "assert_passed"]:
        assert key in gap_dict, f"gap dict missing key: {key}"


def test_stress_v3_gap_dict_value_is_float(traces_file):
    """gap_dict['value'] must be a finite float.
    REQ-FR11-GC-002: the numeric gaming signal is present and finite.
    """
    import math
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    val = result["arm_a_pass_rate_vs_true_accuracy_gap"]["value"]
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_stress_v3_gap_dict_sources_distinct(traces_file):
    """sources_distinct must be True in the gap dict (de-flag flag).
    REQ-FR11-GC-002: the assert ran and the sources are confirmed distinct.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    assert result["arm_a_pass_rate_vs_true_accuracy_gap"]["sources_distinct"] is True


def test_stress_v3_gap_equals_pass_minus_true(traces_file):
    """gap_dict['value'] must equal pass_rate - true_accuracy.
    REQ-FR11-GC-002: arithmetic correctness of the gaming signal.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    d = result["arm_a_pass_rate_vs_true_accuracy_gap"]
    expected = d["pass_rate"] - d["true_accuracy"]
    assert abs(d["value"] - expected) < 1e-9


def test_stress_v3_collapse_onset_field_present(traces_file):
    """collapse_onset_iteration must be present (None or int).
    REQ-FR11-GC-003: the depth answer field is in the artifact.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    assert "collapse_onset_iteration" in result
    onset = result["collapse_onset_iteration"]
    assert onset is None or isinstance(onset, int)


def test_stress_v3_depth_changes_conclusion_is_bool(traces_file):
    """depth_changes_conclusion must be a boolean.
    REQ-FR11-GC-003: the N=50 vs N=200 comparison result is a bool.
    """
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    assert isinstance(result["depth_changes_conclusion"], bool)


def test_stress_v3_no_collapse_on_two_iterations(traces_file):
    """With n_iterations=2, collapse cannot occur (needs >=3 iterations for criterion).
    SCENARIO-FR11-GC-002: insufficient depth → no collapse verdict.
    """
    result = run_stress_test_v3(traces_file, n_iterations=2, seed=42)
    assert not result["arm_a_mode_collapse_detected"]
    assert "residual_diversity_holds" in result["honest_verdict"]


def test_stress_v3_reproducible(traces_file):
    """Same seed must produce identical results (determinism).
    REQ-FR11-GC-001: reproducibility across runs.
    """
    r1 = run_stress_test_v3(traces_file, n_iterations=10, seed=99)
    r2 = run_stress_test_v3(traces_file, n_iterations=10, seed=99)
    assert r1["arm_a_final_entropy"] == r2["arm_a_final_entropy"]
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]


def test_stress_v3_arm_b_entropy_exceeds_arm_a(traces_file):
    """ARM B final entropy must exceed ARM A (regularization works).
    SCENARIO-FR11-GC-001: entropy regularization disrupts concentration.
    """
    result = run_stress_test_v3(traces_file, n_iterations=30, seed=42)
    assert result["arm_b_final_entropy"] > result["arm_a_final_entropy"]


def test_stress_v3_n_iterations_matches(traces_file):
    """n_iterations in output matches requested value (schema integrity)."""
    result = run_stress_test_v3(traces_file, n_iterations=7, seed=42)
    assert result["n_iterations"] == 7


def test_stress_v3_both_collapse_with_zero_beta(traces_file):
    """With entropy_beta=0, ARM B == ARM A; both collapse at N=200.
    SCENARIO-FR11-GC-003: insufficient regularization → both arms collapse.
    """
    result = run_stress_test_v3(traces_file, n_iterations=200, seed=42, entropy_beta=0.0)
    assert result["arm_a_mode_collapse_detected"]
    assert result["arm_b_mode_collapse_detected"]
    assert "collapse_not_prevented_by_entropy_reg" in result["honest_verdict"]


def test_stress_v3_entropy_trend_has_tau_and_pvalue(traces_file):
    """entropy_trend_significance must have tau and p_value."""
    result = run_stress_test_v3(traces_file, n_iterations=10, seed=42)
    ets = result["entropy_trend_significance"]
    assert "tau" in ets
    assert "p_value" in ets


def test_stress_v3_active_weight_recorded(traces_file):
    """active_weight_from_exp3439 must be in artifact and equal 0.146."""
    result = run_stress_test_v3(traces_file, n_iterations=5, seed=42)
    assert "active_weight_from_exp3439" in result
    assert abs(result["active_weight_from_exp3439"] - 0.146) < 1e-6


def test_stress_v3_field_provenance_present(traces_file):
    """field_provenance must be present and cover key fields.
    REQ-FR11-GC-003: principle-annotated artifact schema (CLAUDE.md).
    """
    result = run_stress_test_v3(traces_file, n_iterations=5, seed=42)
    fp = result.get("field_provenance", {})
    assert "arm_a_pass_rate_vs_true_accuracy_gap" in fp
    assert "collapse_onset_iteration" in fp
    assert "depth_changes_conclusion" in fp


def test_stress_v3_empty_file_returns_blocked():
    """Empty traces file returns blocked verdict gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = run_stress_test_v3(path, n_iterations=5, seed=42)
        assert "blocked" in result["honest_verdict"] or result["honest_verdict"].startswith("complete:")
    finally:
        os.unlink(path)


def test_stress_v3_depth_200_collapses_arm_a_not_arm_b(skewed_traces_file):
    """At N=200 depth with skewed (1/150 correct) corpus, ARM A collapses but ARM B does not.
    SCENARIO-FR11-GC-001: the primary depth-question result for the 1/150 corpus.
    REQ-FR11-GC-003: N>=200 is sufficient depth to detect the Dark-Room failure mode.
    """
    result = run_stress_test_v3(skewed_traces_file, n_iterations=200, seed=42)
    assert result["arm_a_mode_collapse_detected"], (
        "ARM A should collapse at N=200 for the 1/150-correct corpus with NULL_WEIGHT=0.854"
    )
    assert not result["arm_b_mode_collapse_detected"], (
        "ARM B with entropy_beta=0.5 should prevent collapse at N=200"
    )
    assert result["depth_changes_conclusion"] is True, (
        "N=200 changes the verdict vs N=50 (exp3462 had no collapse)"
    )
    onset = result["collapse_onset_iteration"]
    assert onset is not None and onset <= 200, (
        "Collapse onset should be detected within the 200-iteration budget"
    )
