"""Tests for FR-11 Minimal Beta + Grounding-Dependence Sweep v4.

Covers REQ-FR11-MB-001, REQ-FR11-MB-002, REQ-FR11-MB-003,
SCENARIO-FR11-MB-001, SCENARIO-FR11-MB-002, SCENARIO-FR11-MB-003.

Key invariants:
1. Scores remain distinct from is_correct at each grounding strength.
2. Minimal sufficient beta is identified or explicitly null.
3. Grounding dependence boolean is reported.
4. All required artifact fields are present.
5. Beta=0 control collapses at N=200 (replicates exp3474).
6. Beta grid sweep results are serializable (no numpy types in artifact).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from carnot.fr11.minimal_beta_grounding_dependence_v4 import (
    BETA_GRID,
    ENTROPY_COLLAPSE_THRESHOLD,
    GROUNDING_STRENGTHS,
    MIN_DEPTH_FOR_COLLAPSE,
    MODE_MASS_COLLAPSE_THRESHOLD,
    _assert_sources_distinct_v4,
    _check_grounding_dependence,
    _collapse_criterion,
    _distribution_entropy,
    _find_collapse_onset_v4,
    _kendall_tau,
    _softmax,
    compute_at_risk_scores_v4,
    run_arm_v4,
    run_minimal_beta_sweep,
    sweep_beta_for_grounding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traces() -> list[dict]:
    """20-trace corpus: every 5th trace is correct.
    REQ-FR11-MB-001: corpus with mixed correct/incorrect traces.
    """
    return [
        {
            "question_id": f"q{i:03d}",
            "prompt": f"Q{i}",
            "completion": f"A{i}",
            "is_correct": i % 5 == 0,
        }
        for i in range(20)
    ]


@pytest.fixture()
def skewed_traces() -> list[dict]:
    """150-trace corpus with 1 correct trace (mirrors fr11_zenil_distill_v2.jsonl).
    REQ-FR11-MB-003: corpus where beta=0 causes collapse at N=200.
    """
    return [
        {"question_id": f"q{i}", "is_correct": (i == 0), "completion": f"c{i}"}
        for i in range(150)
    ]


@pytest.fixture()
def traces_file(small_traces, tmp_path) -> str:
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


@pytest.fixture()
def skewed_traces_file(skewed_traces, tmp_path) -> str:
    path = tmp_path / "skewed.jsonl"
    with open(path, "w") as f:
        for t in skewed_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_beta_grid_contains_zero():
    """BETA_GRID must contain 0.0 (the control condition).
    REQ-FR11-MB-003: beta=0 is the control.
    """
    assert 0.0 in BETA_GRID


def test_beta_grid_contains_half():
    """BETA_GRID must contain 0.5 (the exp3474 cure value).
    REQ-FR11-MB-001: 0.5 is the known-sufficient beta.
    """
    assert 0.5 in BETA_GRID


def test_grounding_strengths_contains_at_risk():
    """GROUNDING_STRENGTHS must contain 0.146 (at-risk from exp3439).
    REQ-FR11-MB-002: at-risk grounding is one test condition.
    """
    assert any(abs(g - 0.146) < 1e-6 for g in GROUNDING_STRENGTHS)


def test_grounding_strengths_has_two_values():
    """GROUNDING_STRENGTHS must have at least 2 values for comparison.
    REQ-FR11-MB-002: grounding-dependence requires at least 2 grounding conditions.
    """
    assert len(GROUNDING_STRENGTHS) >= 2


def test_min_depth_is_200():
    """MIN_DEPTH_FOR_COLLAPSE must be 200 (inherited from exp3474 v3).
    REQ-FR11-MB-003: N>=200 is sufficient depth to detect collapse.
    """
    assert MIN_DEPTH_FOR_COLLAPSE == 200


# ---------------------------------------------------------------------------
# compute_at_risk_scores_v4
# ---------------------------------------------------------------------------


def test_scores_v4_shape(small_traces):
    """Score array length matches trace list.
    REQ-FR11-MB-001: parametric scoring works for any active_weight.
    """
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    assert len(scores) == len(small_traces)


def test_scores_v4_range(small_traces):
    """All scores in [0, 1] (REQ-FR11-MB-001)."""
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.3, seed=42)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_scores_v4_reproducible(small_traces):
    """Same seed produces identical scores (determinism)."""
    s1 = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=99)
    s2 = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=99)
    np.testing.assert_array_equal(s1, s2)


def test_scores_v4_differ_by_active_weight(small_traces):
    """Different active_weight values produce different score distributions.
    REQ-FR11-MB-002: parametric active_weight actually changes scores.
    """
    s1 = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    s2 = compute_at_risk_scores_v4(small_traces, active_weight=0.30, seed=42)
    assert not np.allclose(s1, s2)


def test_scores_v4_null_weight_makes_incorrect_exceed_threshold(skewed_traces):
    """With at-risk grounding, some incorrect traces must score > 0.5.
    REQ-FR11-MB-001: de-flag property preserved with parametric active_weight.
    """
    scores = compute_at_risk_scores_v4(skewed_traces, active_weight=0.146, seed=42)
    is_correct = np.array([t["is_correct"] for t in skewed_traces])
    incorrect_scores = scores[~is_correct]
    assert np.any(incorrect_scores > 0.5)


# ---------------------------------------------------------------------------
# _assert_sources_distinct_v4
# ---------------------------------------------------------------------------


def test_assert_distinct_v4_raises_on_identical():
    """Must raise AssertionError when arrays are identical.
    SCENARIO-FR11-MB-001: active_weight-aware assertion catches regressions.
    """
    arr = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
    with pytest.raises(AssertionError):
        _assert_sources_distinct_v4(arr, arr, active_weight=0.146)


def test_assert_distinct_v4_passes_when_different():
    """Must not raise when arrays differ."""
    pass_arr = np.array([1.0, 1.0, 0.0], dtype=float)
    correct_arr = np.array([0.0, 0.0, 1.0], dtype=float)
    _assert_sources_distinct_v4(pass_arr, correct_arr, active_weight=0.146)


def test_assert_distinct_v4_error_includes_active_weight():
    """Error message must include the active_weight value for diagnostics.
    REQ-FR11-MB-002: grounding context in error messages.
    """
    arr = np.array([1.0, 0.0], dtype=float)
    with pytest.raises(AssertionError, match="active_weight"):
        _assert_sources_distinct_v4(arr, arr, active_weight=0.30)


# ---------------------------------------------------------------------------
# _collapse_criterion
# ---------------------------------------------------------------------------


def test_collapse_criterion_depth_aware():
    """Depth-aware criterion fires at N>=200 with mode_mass>0.5 and drop>75%.
    REQ-FR11-MB-003: same collapse logic as exp3474 v3.
    """
    assert _collapse_criterion(
        entropy_drop_ratio=0.80,
        final_entropy=1.0,
        final_mode_mass=0.6,
        n_iterations=200,
    )


def test_collapse_criterion_no_collapse_low_drop():
    """No collapse with entropy drop < 75% even at N=200.
    SCENARIO-FR11-MB-002: healthy grounding holds off collapse.
    """
    assert not _collapse_criterion(
        entropy_drop_ratio=0.50,
        final_entropy=2.0,
        final_mode_mass=0.3,
        n_iterations=200,
    )


def test_collapse_criterion_legacy_fires():
    """Legacy criterion fires at N>=3 with drop>85% and mode_mass>0.5."""
    assert _collapse_criterion(
        entropy_drop_ratio=0.90,
        final_entropy=1.5,
        final_mode_mass=0.8,
        n_iterations=10,
    )


# ---------------------------------------------------------------------------
# _find_collapse_onset_v4
# ---------------------------------------------------------------------------


def test_find_onset_empty():
    """Empty sequence returns None (REQ-FR11-MB-001)."""
    assert _find_collapse_onset_v4([], n_total=200) is None


def test_find_onset_no_collapse():
    """High-entropy sequence never triggers collapse onset.
    SCENARIO-FR11-MB-002: healthy arm has no onset.
    """
    per_iteration = [{"iteration": float(i), "entropy": 4.0, "mode_mass": 0.05}
                     for i in range(50)]
    assert _find_collapse_onset_v4(per_iteration, n_total=200) is None


def test_find_onset_detects_collapse():
    """Collapse onset detected when criteria fire.
    SCENARIO-FR11-MB-001: onset is a non-negative integer.

    The sequence starts at high entropy (4.0) then drops sharply to 0.02 with
    high mode_mass=0.98.  At that point the legacy criterion fires:
    drop_ratio=(4.0-0.02)/4.0=0.995 > 0.85 AND entropy < 0.1 (ENTROPY_COLLAPSE_THRESHOLD).
    """
    # First 5 iterations: healthy (high entropy, low mode_mass)
    per_iteration = [{"iteration": float(i), "entropy": 4.0, "mode_mass": 0.05}
                     for i in range(5)]
    # Iterations 5+: collapsed (entropy << initial, mode_mass >> threshold)
    per_iteration += [{"iteration": float(i), "entropy": 0.02, "mode_mass": 0.98}
                      for i in range(5, 20)]
    onset = _find_collapse_onset_v4(per_iteration, n_total=200)
    assert onset is not None
    assert isinstance(onset, int)
    assert onset >= 0


# ---------------------------------------------------------------------------
# _kendall_tau
# ---------------------------------------------------------------------------


def test_kendall_tau_declining():
    """Declining sequence has tau < -0.9 and p < 0.05."""
    seq = [5.0 - i * 0.1 for i in range(30)]
    tau, p = _kendall_tau(seq)
    assert tau < -0.9
    assert p < 0.05


def test_kendall_tau_short_sequence():
    """< 4 points returns (0.0, 1.0)."""
    tau, p = _kendall_tau([2.0, 1.0])
    assert tau == 0.0
    assert p == 1.0


# ---------------------------------------------------------------------------
# _softmax / _distribution_entropy
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    p = _softmax(np.array([0.5, -0.5, 1.0]))
    assert abs(float(np.sum(p)) - 1.0) < 1e-9


def test_entropy_uniform():
    n = 8
    p = np.ones(n) / n
    assert abs(_distribution_entropy(p) - math.log(n)) < 1e-9


def test_entropy_degenerate():
    p = np.array([1.0, 0.0, 0.0])
    assert _distribution_entropy(p) < 1e-6


# ---------------------------------------------------------------------------
# run_arm_v4
# ---------------------------------------------------------------------------


def test_arm_v4_required_fields(small_traces):
    """run_arm_v4 returns all expected keys.
    REQ-FR11-MB-001: arm result schema is complete.
    """
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    result = run_arm_v4(small_traces, scores, n_iterations=5, use_entropy_reg=False)
    for field in [
        "per_iteration", "entropy_sequence", "final_entropy", "final_mode_mass",
        "final_pass_rate", "final_true_accuracy", "final_gap",
        "initial_entropy", "entropy_drop_ratio", "mode_collapse_detected",
        "collapse_onset", "entropy_trend_tau", "entropy_trend_p_value",
    ]:
        assert field in result, f"Missing: {field}"


def test_arm_v4_gap_equals_difference(small_traces):
    """final_gap must equal final_pass_rate - final_true_accuracy.
    REQ-FR11-MB-001: gaming signal is correctly computed.
    """
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    result = run_arm_v4(small_traces, scores, n_iterations=10, use_entropy_reg=False)
    expected = result["final_pass_rate"] - result["final_true_accuracy"]
    assert abs(result["final_gap"] - expected) < 1e-9


def test_arm_v4_reg_preserves_entropy(small_traces):
    """ARM with entropy reg has higher final entropy than without.
    SCENARIO-FR11-MB-001: regularization disrupts mode concentration.
    """
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    arm_a = run_arm_v4(small_traces, scores, n_iterations=30, use_entropy_reg=False)
    arm_b = run_arm_v4(small_traces, scores, n_iterations=30, use_entropy_reg=True, entropy_beta=0.5)
    assert arm_b["final_entropy"] > arm_a["final_entropy"]


def test_arm_v4_entropy_sequence_length(small_traces):
    """entropy_sequence has exactly n_iterations entries."""
    scores = compute_at_risk_scores_v4(small_traces, active_weight=0.146, seed=42)
    result = run_arm_v4(small_traces, scores, n_iterations=12, use_entropy_reg=False)
    assert len(result["entropy_sequence"]) == 12


# ---------------------------------------------------------------------------
# sweep_beta_for_grounding
# ---------------------------------------------------------------------------


def test_sweep_required_keys(traces_file):
    """sweep_beta_for_grounding returns expected top-level keys.
    REQ-FR11-MB-001: sweep result has complete schema.
    """
    traces = []
    with open(traces_file) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    result = sweep_beta_for_grounding(
        traces=traces,
        active_weight=0.146,
        beta_grid=[0.0, 0.5],
        n_iterations=5,
        seed=42,
    )
    for key in ["active_weight", "null_weight", "arm_a_result", "beta_results", "minimal_sufficient_beta"]:
        assert key in result, f"Missing key: {key}"


def test_sweep_beta_results_keys(traces_file):
    """beta_results has entries for each beta in grid.
    REQ-FR11-MB-001: all betas are covered.
    """
    traces = []
    with open(traces_file) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    result = sweep_beta_for_grounding(
        traces=traces,
        active_weight=0.146,
        beta_grid=[0.0, 0.25, 0.5],
        n_iterations=5,
        seed=42,
    )
    # Expect keys "0.000", "0.250", "0.500"
    assert "0.000" in result["beta_results"]
    assert "0.250" in result["beta_results"]
    assert "0.500" in result["beta_results"]


def test_sweep_minimal_beta_none_when_all_collapse(small_traces):
    """minimal_sufficient_beta is None when all betas cause collapse.
    SCENARIO-FR11-MB-002: correctly reports no-cure scenario.
    """
    # Use zero beta only — all arms are the control, all collapse at N=200
    result = sweep_beta_for_grounding(
        traces=small_traces,
        active_weight=0.146,
        beta_grid=[0.0],
        n_iterations=200,
        seed=42,
    )
    # beta=0 always is the control (same as ARM A): if ARM A collapses, minimal is None
    if result["arm_a_result"]["collapse_detected"]:
        assert result["minimal_sufficient_beta"] is None


def test_sweep_minimal_beta_at_half(skewed_traces):
    """With skewed corpus at at-risk grounding, beta=0.5 prevents collapse.
    SCENARIO-FR11-MB-001: minimal beta <= 0.5 for at-risk grounding.
    REQ-FR11-MB-001: acceptance gate G1 met.
    """
    result = sweep_beta_for_grounding(
        traces=skewed_traces,
        active_weight=0.146,
        beta_grid=[0.0, 0.1, 0.25, 0.5],
        n_iterations=200,
        seed=42,
    )
    # ARM A must collapse (replicates exp3474)
    assert result["arm_a_result"]["collapse_detected"], "ARM A must collapse at N=200 for skewed corpus"
    # A minimal beta <= 0.5 must exist
    assert result["minimal_sufficient_beta"] is not None
    assert result["minimal_sufficient_beta"] <= 0.5


# ---------------------------------------------------------------------------
# _check_grounding_dependence
# ---------------------------------------------------------------------------


def test_grounding_dependence_differs_minimal_beta():
    """True when minimal betas differ across grounding strengths.
    REQ-FR11-MB-002: correctly detects grounding-dependence.
    """
    per_grounding = {
        "0.146": {"minimal_sufficient_beta": 0.5, "beta_results": {}},
        "0.300": {"minimal_sufficient_beta": 0.1, "beta_results": {}},
    }
    result = _check_grounding_dependence(per_grounding, [0.146, 0.3], [0.0, 0.1, 0.5])
    assert result is True


def test_grounding_dependence_false_when_same():
    """False when minimal betas are identical across grounding strengths.
    REQ-FR11-MB-002: correctly identifies no-dependence.
    """
    per_grounding = {
        "0.146": {"minimal_sufficient_beta": 0.25, "beta_results": {"0.000": {"collapse_onset": 100}}},
        "0.300": {"minimal_sufficient_beta": 0.25, "beta_results": {"0.000": {"collapse_onset": 110}}},
    }
    result = _check_grounding_dependence(per_grounding, [0.146, 0.3], [0.0, 0.25])
    # Onset difference 10 < 20 threshold and same minimal beta → False
    assert result is False


def test_grounding_dependence_onset_spread():
    """True when collapse onset at beta=0 differs by >= 20 iterations.
    REQ-FR11-MB-002: onset spread is a secondary grounding-dependence signal.
    """
    per_grounding = {
        "0.146": {
            "minimal_sufficient_beta": 0.25,
            "beta_results": {"0.000": {"collapse_onset": 50}},
        },
        "0.300": {
            "minimal_sufficient_beta": 0.25,
            "beta_results": {"0.000": {"collapse_onset": 180}},
        },
    }
    result = _check_grounding_dependence(per_grounding, [0.146, 0.3], [0.0, 0.25])
    assert result is True


# ---------------------------------------------------------------------------
# run_minimal_beta_sweep — integration tests
# ---------------------------------------------------------------------------


def test_sweep_verdict_prefix(traces_file):
    """honest_verdict must start with 'complete:' (Verdict Terminal-Prefix Discipline).
    REQ-FR11-MB-001: conductor reconciler classifies the verdict correctly.
    """
    result = run_minimal_beta_sweep(
        traces_path=traces_file,
        n_iterations=10,
        seed=42,
        beta_grid=[0.0, 0.5],
        grounding_strengths=[0.146],
    )
    assert result["honest_verdict"].startswith("complete:")


def test_sweep_inference_substrate(traces_file):
    """inference_substrate must be verifier_ensemble_against_cached_candidates."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_sweep_duration_floor(traces_file):
    """duration_s >= 1.0 (CLAUDE.md verifier_ensemble floor)."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert result["duration_s"] >= 1.0


def test_sweep_all_required_artifact_fields(traces_file):
    """All REQUIRED ARTIFACT FIELDS from the task spec must be present.
    REQ-FR11-MB-001, REQ-FR11-MB-002, REQ-FR11-MB-003.
    """
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146, 0.3])
    required = [
        "honest_verdict",
        "inference_substrate",
        "n_iterations",
        "beta_grid",
        "minimal_sufficient_beta",
        "collapse_onset_by_beta",
        "pass_rate_vs_true_accuracy_gap_beta0",
        "grounding_strengths_tested",
        "minimal_beta_depends_on_grounding",
        "entropy_trend_significance_beta0",
        "recommended_phase5_default",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_sweep_pass_rate_gap_is_dict(traces_file):
    """pass_rate_vs_true_accuracy_gap_beta0 must be a dict (not a bare float).
    REQ-FR11-MB-001: de-flag from exp3474 v3 preserved.
    """
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert isinstance(result["pass_rate_vs_true_accuracy_gap_beta0"], dict)


def test_sweep_gap_dict_keys(traces_file):
    """Gap dict must have value, pass_rate, true_accuracy, sources_distinct, assert_passed."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    d = result["pass_rate_vs_true_accuracy_gap_beta0"]
    for key in ["value", "pass_rate", "true_accuracy", "sources_distinct", "assert_passed"]:
        assert key in d, f"Missing key: {key}"


def test_sweep_gap_value_finite(traces_file):
    """gap_dict['value'] must be a finite float."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    val = result["pass_rate_vs_true_accuracy_gap_beta0"]["value"]
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_sweep_gap_sources_distinct(traces_file):
    """sources_distinct must be True."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert result["pass_rate_vs_true_accuracy_gap_beta0"]["sources_distinct"] is True


def test_sweep_entropy_trend_has_tau_pvalue(traces_file):
    """entropy_trend_significance_beta0 must have tau and p_value."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    ets = result["entropy_trend_significance_beta0"]
    assert "tau" in ets
    assert "p_value" in ets


def test_sweep_minimal_beta_depends_is_bool(traces_file):
    """minimal_beta_depends_on_grounding must be a bool.
    REQ-FR11-MB-002: grounding-dependence result is a boolean.
    """
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146, 0.3])
    assert isinstance(result["minimal_beta_depends_on_grounding"], bool)


def test_sweep_grounding_strengths_recorded(traces_file):
    """grounding_strengths_tested must match what was requested."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146, 0.3])
    assert sorted(result["grounding_strengths_tested"]) == sorted([0.146, 0.3])


def test_sweep_beta_grid_recorded(traces_file):
    """beta_grid in artifact must match the requested grid."""
    grid = [0.0, 0.1, 0.5]
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=grid, grounding_strengths=[0.146])
    assert result["beta_grid"] == sorted(grid)


def test_sweep_reproducible(traces_file):
    """Same seed produces identical results (determinism)."""
    r1 = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=77,
                                 beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    r2 = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=77,
                                 beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]
    assert r1["minimal_sufficient_beta"] == r2["minimal_sufficient_beta"]


def test_sweep_acceptance_gates_present(traces_file):
    """acceptance_gates field must have G1 and G2."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=10, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    gates = result["acceptance_gates"]
    assert "G1_minimal_beta_found" in gates
    assert "G2_beta0_still_collapses" in gates


def test_sweep_field_provenance_present(traces_file):
    """field_provenance must cover all required artifact fields.
    REQ-FR11-MB-001: principle-annotated schema per CLAUDE.md.
    """
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    fp = result.get("field_provenance", {})
    for key in ["honest_verdict", "minimal_sufficient_beta", "collapse_onset_by_beta",
                "pass_rate_vs_true_accuracy_gap_beta0", "minimal_beta_depends_on_grounding",
                "recommended_phase5_default", "duration_s"]:
        assert key in fp, f"field_provenance missing: {key}"


def test_sweep_empty_file_returns_blocked(tmp_path):
    """Empty traces file returns blocked verdict gracefully."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    result = run_minimal_beta_sweep(str(empty), n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert result["honest_verdict"].startswith("complete:")
    assert "blocked" in result["honest_verdict"]


def test_sweep_beta0_collapses_at_n200_skewed_corpus(skewed_traces_file):
    """At N=200 with skewed corpus, beta=0 ARM A collapses — replicates exp3474.
    REQ-FR11-MB-003: G2 acceptance gate holds.
    SCENARIO-FR11-MB-001: the sweep measures a real effect, not a stale corpus.
    """
    result = run_minimal_beta_sweep(
        traces_path=skewed_traces_file,
        n_iterations=200,
        seed=42,
        beta_grid=[0.0, 0.5],
        grounding_strengths=[0.146],
    )
    assert result["acceptance_gates"]["G2_beta0_still_collapses"], (
        "G2: beta=0 must still collapse at N=200 (replicates exp3474)"
    )
    onset = result["collapse_onset_by_beta"].get("0.000")
    assert onset is not None, "collapse_onset_by_beta[0] must not be null (G2)"


def test_sweep_minimal_beta_at_most_half_skewed_corpus(skewed_traces_file):
    """Minimal sufficient beta must be <= 0.5 for skewed corpus (G1 gate).
    REQ-FR11-MB-001: G1 acceptance gate met.
    SCENARIO-FR11-MB-001: at-risk grounding requires some beta but no more than 0.5.
    """
    result = run_minimal_beta_sweep(
        traces_path=skewed_traces_file,
        n_iterations=200,
        seed=42,
        beta_grid=[0.0, 0.1, 0.25, 0.5],
        grounding_strengths=[0.146],
    )
    assert result["acceptance_gates"]["G1_minimal_beta_found"], (
        "G1: minimal_sufficient_beta must not be null for the skewed corpus"
    )
    assert result["minimal_sufficient_beta"] is not None
    assert result["minimal_sufficient_beta"] <= 0.5


def test_sweep_recommended_phase5_default_is_string(traces_file):
    """recommended_phase5_default must be a non-empty string."""
    result = run_minimal_beta_sweep(traces_path=traces_file, n_iterations=5, seed=42,
                                    beta_grid=[0.0, 0.5], grounding_strengths=[0.146])
    assert isinstance(result["recommended_phase5_default"], str)
    assert len(result["recommended_phase5_default"]) > 10
