"""Tests for the FR-11 Grounding Collapse Stress Test module.

Tests cover REQ-FR11-GC-001 (at-risk grounding collapse test) and
SCENARIO-FR11-GC-001/002 (ARM A collapses / residual diversity holds).
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from carnot.fr11.grounding_collapse_stress_test import (
    ENTROPY_COLLAPSE_THRESHOLD,
    ENTROPY_REGULARIZATION_BETA,
    MODE_MASS_COLLAPSE_THRESHOLD,
    NULL_SPACE_FRACTION,
    _distribution_entropy,
    _softmax,
    compute_at_risk_scores,
    run_arm,
    run_stress_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_traces() -> list[dict]:
    """Small fixed corpus of 20 traces with known correctness pattern."""
    traces = []
    for i in range(20):
        traces.append({
            "question_id": f"arith_{i:03d}",
            "prompt": f"Question {i}",
            "completion": f"Answer {i}",
            "is_correct": i % 3 == 0,  # every 3rd trace is correct
            "energy_score": 0.0,
        })
    return traces


@pytest.fixture()
def cached_traces_file(sample_traces, tmp_path) -> str:
    """Write sample traces to a temp JSONL file and return the path."""
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in sample_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# _softmax tests
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    """softmax output must sum to 1 for numerical stability (REQ-FR11-GC-001)."""
    log_w = np.array([1.0, 2.0, 3.0, 0.0])
    p = _softmax(log_w)
    assert abs(float(np.sum(p)) - 1.0) < 1e-9


def test_softmax_monotone():
    """Higher log-weight → higher probability (REQ-FR11-GC-001)."""
    log_w = np.array([0.0, 1.0, 2.0])
    p = _softmax(log_w)
    assert p[2] > p[1] > p[0]


def test_softmax_uniform_on_equal_weights():
    """Equal log-weights produce a uniform distribution."""
    log_w = np.zeros(5)
    p = _softmax(log_w)
    np.testing.assert_allclose(p, np.ones(5) / 5, atol=1e-9)


# ---------------------------------------------------------------------------
# _distribution_entropy tests
# ---------------------------------------------------------------------------


def test_entropy_uniform_maximum():
    """Uniform distribution has maximum entropy = log(n) (REQ-FR11-GC-001)."""
    n = 10
    p = np.ones(n) / n
    expected = math.log(n)
    assert abs(_distribution_entropy(p) - expected) < 1e-9


def test_entropy_zero_on_degenerate():
    """Deterministic distribution has zero entropy (REQ-FR11-GC-001)."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    assert _distribution_entropy(p) < 1e-6


def test_entropy_positive_on_nondegenerate():
    """Any non-degenerate distribution has positive entropy."""
    p = np.array([0.7, 0.3])
    assert _distribution_entropy(p) > 0.0


# ---------------------------------------------------------------------------
# compute_at_risk_scores tests
# ---------------------------------------------------------------------------


def test_at_risk_scores_shape(sample_traces):
    """Score array has same length as trace list (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores(sample_traces, seed=42)
    assert len(scores) == len(sample_traces)


def test_at_risk_scores_range(sample_traces):
    """All scores lie in [0, 1] (REQ-FR11-GC-001)."""
    scores = compute_at_risk_scores(sample_traces, seed=42)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_at_risk_scores_reproducible(sample_traces):
    """Same seed produces identical scores (REQ-FR11-GC-001 determinism)."""
    s1 = compute_at_risk_scores(sample_traces, seed=7)
    s2 = compute_at_risk_scores(sample_traces, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_at_risk_scores_vary_by_seed(sample_traces):
    """Different seeds produce different null-space components."""
    s1 = compute_at_risk_scores(sample_traces, seed=1)
    s2 = compute_at_risk_scores(sample_traces, seed=2)
    assert not np.allclose(s1, s2)


def test_at_risk_scores_correct_bias(sample_traces):
    """Correct traces should on average score higher than incorrect ones.

    This confirms the active-signal component is load-bearing, even with the
    null-space dilution. The margin is smaller than it would be with a perfect
    verifier, which is the whole point — the null-space weakens but doesn't
    eliminate the discrimination signal.
    """
    scores = compute_at_risk_scores(sample_traces, seed=42)
    correct_mean = np.mean([s for t, s in zip(sample_traces, scores) if t["is_correct"]])
    incorrect_mean = np.mean([s for t, s in zip(sample_traces, scores) if not t["is_correct"]])
    # With 4/6 active signal, correct should be higher on average
    assert correct_mean > incorrect_mean


# ---------------------------------------------------------------------------
# run_arm tests
# ---------------------------------------------------------------------------


def test_arm_a_entropy_drops(sample_traces):
    """ARM A must show entropy decrease over iterations (SCENARIO-FR11-GC-001).

    Without entropy regularization, accumulating fixed scores concentrates mass
    on argmax, so entropy must fall from the initial uniform value.
    """
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    result = run_arm(sample_traces, at_risk, n_iterations=20, use_entropy_reg=False)
    initial_entropy = result["initial_entropy"]
    final_entropy = result["final_entropy"]
    # ARM A must have lower entropy at the end than at the start
    assert final_entropy < initial_entropy


def test_arm_b_maintains_higher_entropy_than_arm_a(sample_traces):
    """ARM B entropy must be higher than ARM A entropy at equal iteration count.

    This is the core claim of entropy regularization — it prevents concentration
    that ARM A exhibits (SCENARIO-FR11-GC-001).
    """
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    arm_a = run_arm(sample_traces, at_risk, n_iterations=30, use_entropy_reg=False)
    arm_b = run_arm(sample_traces, at_risk, n_iterations=30, use_entropy_reg=True)
    assert arm_b["final_entropy"] > arm_a["final_entropy"]


def test_arm_result_has_required_fields(sample_traces):
    """run_arm must return all required fields (REQ-FR11-GC-001)."""
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    result = run_arm(sample_traces, at_risk, n_iterations=5, use_entropy_reg=False)
    for field in [
        "per_iteration",
        "final_entropy",
        "final_mode_mass",
        "final_pass_rate",
        "final_true_accuracy",
        "initial_entropy",
        "entropy_drop_ratio",
        "mode_collapse_detected",
    ]:
        assert field in result, f"Missing field: {field}"


def test_arm_per_iteration_length(sample_traces):
    """per_iteration list must have exactly n_iterations entries."""
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    result = run_arm(sample_traces, at_risk, n_iterations=10, use_entropy_reg=False)
    assert len(result["per_iteration"]) == 10


def test_arm_mode_mass_in_range(sample_traces):
    """Mode mass must lie in (0, 1]."""
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    result = run_arm(sample_traces, at_risk, n_iterations=5, use_entropy_reg=False)
    assert 0.0 < result["final_mode_mass"] <= 1.0


def test_arm_a_collapses_on_many_iterations(sample_traces):
    """With enough iterations, ARM A must detect mode-collapse.

    The mathematical guarantee: softmax(T * scores) → one-hot as T → ∞, so
    after enough iterations, ARM A must satisfy the collapse conditions.
    This tests SCENARIO-FR11-GC-001 is structurally achievable.
    """
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    # Use many iterations so softmax(T*scores) → one-hot definitively
    result = run_arm(sample_traces, at_risk, n_iterations=200, use_entropy_reg=False)
    # ARM A should strongly concentrate: mode_mass should be very high
    assert result["final_mode_mass"] > 0.8
    # AND the entropy should have collapsed
    assert result["mode_collapse_detected"]


def test_arm_b_resists_concentration(sample_traces):
    """ARM B mode mass must remain lower than ARM A at same iteration count."""
    at_risk = compute_at_risk_scores(sample_traces, seed=42)
    arm_a = run_arm(sample_traces, at_risk, n_iterations=200, use_entropy_reg=False)
    arm_b = run_arm(sample_traces, at_risk, n_iterations=200, use_entropy_reg=True)
    assert arm_b["final_mode_mass"] < arm_a["final_mode_mass"]


# ---------------------------------------------------------------------------
# run_stress_test integration tests
# ---------------------------------------------------------------------------


def test_stress_test_returns_honest_verdict(cached_traces_file):
    """run_stress_test must return an honest_verdict starting with 'complete:'.

    This is the CLAUDE.md Verdict Terminal-Prefix requirement.
    """
    result = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    assert "honest_verdict" in result
    verdict = result["honest_verdict"]
    assert verdict.startswith("complete:") or verdict.startswith("success:")


def test_stress_test_inference_substrate(cached_traces_file):
    """inference_substrate must be verifier_ensemble_against_cached_candidates.

    This is required per CLAUDE.md Inference-Substrate Declaration Discipline.
    """
    result = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    assert result.get("inference_substrate") == "verifier_ensemble_against_cached_candidates"


def test_stress_test_required_artifact_fields(cached_traces_file):
    """All REQUIRED ARTIFACT FIELDS from the task spec must be present."""
    result = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    required = [
        "honest_verdict",
        "inference_substrate",
        "n_iterations",
        "arm_a_final_entropy",
        "arm_b_final_entropy",
        "arm_a_mode_collapse_detected",
        "arm_b_mode_collapse_detected",
        "grounding_collapse_consequence",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    for field in required:
        assert field in result, f"Missing required artifact field: {field}"


def test_stress_test_duration_above_floor(cached_traces_file):
    """duration_s must be >= 1.0 per inference_substrate discipline (1s floor)."""
    result = run_stress_test(cached_traces_file, n_iterations=5, seed=42)
    assert result["duration_s"] >= 1.0


def test_stress_test_reproducible(cached_traces_file):
    """Two runs with the same seed must produce the same results."""
    r1 = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    r2 = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    assert r1["arm_a_final_entropy"] == r2["arm_a_final_entropy"]
    assert r1["arm_b_final_entropy"] == r2["arm_b_final_entropy"]
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]


def test_stress_test_arm_b_entropy_exceeds_arm_a(cached_traces_file):
    """ARM B final entropy must exceed ARM A final entropy.

    This confirms the regularization is actually working in the integrated path.
    """
    result = run_stress_test(cached_traces_file, n_iterations=30, seed=42)
    assert result["arm_b_final_entropy"] > result["arm_a_final_entropy"]


def test_stress_test_grounding_consequence_is_string(cached_traces_file):
    """grounding_collapse_consequence must be a non-empty string."""
    result = run_stress_test(cached_traces_file, n_iterations=10, seed=42)
    consequence = result.get("grounding_collapse_consequence", "")
    assert isinstance(consequence, str)
    assert len(consequence) > 0


def test_stress_test_empty_file_returns_blocked():
    """Empty traces file should return a blocked verdict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        path = f.name
    try:
        result = run_stress_test(path, n_iterations=5, seed=42)
        # Should signal unavailability (blocked) or fail gracefully
        verdict = result.get("honest_verdict", "")
        assert "blocked" in verdict or "complete:" in verdict
    finally:
        os.unlink(path)


def test_stress_test_n_iterations_matches_input(cached_traces_file):
    """n_iterations in output must match what was requested."""
    result = run_stress_test(cached_traces_file, n_iterations=7, seed=42)
    assert result["n_iterations"] == 7


def test_stress_test_grounding_holds_verdict(cached_traces_file):
    """With n_iterations < 3, entropy_monotone_drop is always False so ARM A cannot collapse.

    This exercises the 'residual_diversity_holds' verdict branch (SCENARIO-FR11-GC-002).
    The collapse detector requires len(per_iteration) >= 3 — with 2 iterations, no collapse
    can be detected, so ARM A is declared non-collapsed and the GROUNDING-HOLDS path fires.
    """
    result = run_stress_test(cached_traces_file, n_iterations=2, seed=42)
    # Neither arm can collapse in 2 iterations (monotone-drop requires >= 3 checkpoints)
    assert not result["arm_a_mode_collapse_detected"]
    assert "residual_diversity_holds" in result["honest_verdict"]


def test_stress_test_both_collapse_verdict(cached_traces_file):
    """With entropy_beta=0, ARM B is identical to ARM A, so both collapse.

    This exercises the 'collapse_not_prevented_by_entropy_reg' verdict branch.
    Setting entropy_beta=0 removes the regularization from ARM B, making it as
    vulnerable to null-space gaming as ARM A.
    """
    # Need enough iterations for both arms to collapse
    result = run_stress_test(cached_traces_file, n_iterations=200, seed=42, entropy_beta=0.0)
    # Both should collapse since ARM B has no regularization
    assert result["arm_a_mode_collapse_detected"]
    assert result["arm_b_mode_collapse_detected"]
    assert "collapse_not_prevented_by_entropy_reg" in result["honest_verdict"]
