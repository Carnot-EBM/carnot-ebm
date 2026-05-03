"""Tests for Exp 1209 GRPO-VPS step-level process supervision.

Spec: REQ-LEARN-1209, SCENARIO-LEARN-1211, SCENARIO-LEARN-1212,
      SCENARIO-LEARN-1213, SCENARIO-LEARN-1214
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training.grpo_vps import aggregate_step_rewards, segment_reward
from carnot.verify.z3_math_verifier import Z3MathVerifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DELIVERABLE = REPO_ROOT / "results" / "experiment_1209_grpo_vps_step_level_supervision.json"


# ---------------------------------------------------------------------------
# REQ-LEARN-1209-1: CausalReasoningVerifier.verify_step
# ---------------------------------------------------------------------------


def test_causal_verify_step_consistent_returns_low():
    """SCENARIO-LEARN-1211: consistent prior + current step gives < 0.5."""
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier

    prior = "There are 12 apples in 3 bags."
    current = "Each bag has 4 apples."
    score = CausalReasoningVerifier().verify_step(current, prior)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score < 0.5


def test_causal_verify_step_none_prior_returns_zero_or_low():
    """REQ-LEARN-1209-1: no prior step means no causal check possible."""
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier

    score = CausalReasoningVerifier().verify_step("We start with 10 items.", None)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_causal_verify_step_returns_float_in_range():
    """REQ-LEARN-1209-1: return value is always in [0.0, 1.0]."""
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier

    crv = CausalReasoningVerifier()
    for step, prior in [
        ("Step with no numbers at all.", None),
        ("47 + 28 = 75.", "We have 47 items."),
        ("So the total is 100.", "So the total is 50."),
    ]:
        result = crv.verify_step(step, prior)
        assert 0.0 <= result <= 1.0, f"out of range for step={step!r}: {result}"


def test_causal_verify_step_no_arg_constructor():
    """REQ-LEARN-1209-1: CausalReasoningVerifier() (no-arg) must be usable."""
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier

    crv = CausalReasoningVerifier()
    score = crv.verify_step("The answer is 42.", None)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# REQ-LEARN-1209-2: Z3MathVerifier.verify_step
# ---------------------------------------------------------------------------


def test_z3_verify_step_wrong_arithmetic_returns_high():
    """SCENARIO-LEARN-1212: '3 + 4 = 8' is wrong, score > 0.5."""
    score = Z3MathVerifier().verify_step("3 + 4 = 8")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_z3_verify_step_correct_arithmetic_returns_zero():
    """REQ-LEARN-1209-2: correct arithmetic gives 0.0 (no violation)."""
    score = Z3MathVerifier().verify_step("3 + 4 = 7")
    assert score == 0.0


def test_z3_verify_step_no_arithmetic_returns_zero():
    """REQ-LEARN-1209-2: no arithmetic claims → 0.0 (not 0.5 sentinel)."""
    score = Z3MathVerifier().verify_step("The sky is blue and birds fly.")
    assert score == 0.0


def test_z3_verify_step_empty_string_returns_zero():
    """REQ-LEARN-1209-2: empty input → 0.0."""
    assert Z3MathVerifier().verify_step("") == 0.0


def test_z3_verify_step_returns_float_in_range():
    """REQ-LEARN-1209-2: return value is always in [0.0, 1.0]."""
    z3v = Z3MathVerifier()
    for step in ["10 + 5 = 15", "10 + 5 = 20", "no math here", ""]:
        result = z3v.verify_step(step)
        assert 0.0 <= result <= 1.0, f"out of range for step={step!r}: {result}"


# ---------------------------------------------------------------------------
# REQ-LEARN-1209-3: segment_reward
# ---------------------------------------------------------------------------


def test_segment_reward_symmetric_average():
    """SCENARIO-LEARN-1213: reward = 0.5*(1-causal) + 0.5*(1-z3)."""
    # Use a step with no violations for known scores.
    reward = segment_reward("3 + 4 = 7", step_index=0, prior_step=None)
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0
    # Correct arithmetic → z3_score = 0.0; no prior → causal_score = 0.0.
    # Expected reward = 0.5*(1-0) + 0.5*(1-0) = 1.0.
    assert reward == pytest.approx(1.0, abs=0.01)


def test_segment_reward_arithmetic_violation_lowers_reward():
    """REQ-LEARN-1209-3: wrong arithmetic reduces reward below 1.0."""
    reward_bad = segment_reward("3 + 4 = 8", step_index=0, prior_step=None)
    reward_ok = segment_reward("3 + 4 = 7", step_index=0, prior_step=None)
    assert reward_bad < reward_ok


def test_segment_reward_clamped_to_unit_interval():
    """REQ-LEARN-1209-3: output is always in [0.0, 1.0]."""
    for text in ["1 + 1 = 2", "1 + 1 = 3", "no math"]:
        r = segment_reward(text, 0, None)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# REQ-LEARN-1209-4: aggregate_step_rewards
# ---------------------------------------------------------------------------


def test_aggregate_step_rewards_geometric_decay():
    """SCENARIO-LEARN-1214: [1, 1, 1] with gamma=0.9 → 1 + 0.9 + 0.81 = 2.71."""
    result = aggregate_step_rewards([1.0, 1.0, 1.0], gamma=0.9)
    assert result == pytest.approx(2.71, abs=1e-6)


def test_aggregate_step_rewards_empty_list():
    """REQ-LEARN-1209-4: empty list returns 0.0."""
    assert aggregate_step_rewards([]) == 0.0


def test_aggregate_step_rewards_single_step():
    """REQ-LEARN-1209-4: single step reward is returned as-is."""
    assert aggregate_step_rewards([0.75]) == pytest.approx(0.75)


def test_aggregate_step_rewards_default_gamma():
    """REQ-LEARN-1209-4: default gamma is 0.9."""
    result = aggregate_step_rewards([1.0, 1.0])
    assert result == pytest.approx(1.0 + 0.9, abs=1e-6)


def test_aggregate_step_rewards_gamma_one_equal_sum():
    """REQ-LEARN-1209-4: gamma=1.0 gives the plain sum."""
    assert aggregate_step_rewards([0.5, 0.3, 0.2], gamma=1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Deliverable schema
# ---------------------------------------------------------------------------


def test_deliverable_exists():
    """REQ-LEARN-1209-5: artifact file must exist after the experiment runs."""
    assert DELIVERABLE.exists(), f"Artifact not found: {DELIVERABLE}"


def test_deliverable_required_fields():
    """REQ-LEARN-1209-5: artifact must contain all required schema fields."""
    data = json.loads(DELIVERABLE.read_text())
    if data.get("status") == "in_progress":
        pytest.skip("Experiment not yet run")
    required = [
        "n_questions_evaluated",
        "causal_verifier_violations_pct",
        "z3_verifier_violations_pct",
        "step_reward_correctness_correlation",
        "outcome_baseline_accuracy",
        "grpo_vps_accuracy",
        "grpo_vps_delta_pp",
        "grpo_vps_step_delta_measured",
        "model_used",
        "honest_verdict",
    ]
    for field in required:
        assert field in data, f"Missing required field: {field}"


def test_deliverable_honest_verdict_valid():
    """REQ-LEARN-1209-6: honest_verdict must be one of the defined values."""
    data = json.loads(DELIVERABLE.read_text())
    valid = {
        "step_supervision_improves_over_outcome",
        "step_supervision_no_delta",
        "step_supervision_degrades",
        "insufficient_step_signal",
        "in_progress",
    }
    assert data["honest_verdict"] in valid, f"Unexpected verdict: {data['honest_verdict']}"


def test_deliverable_n_questions_evaluated():
    """REQ-LEARN-1209-5: n_questions_evaluated must be 50 after run."""
    data = json.loads(DELIVERABLE.read_text())
    if data.get("status") == "in_progress":
        pytest.skip("Experiment not yet run")
    assert data["n_questions_evaluated"] == 50


def test_deliverable_step_delta_measured():
    """REQ-LEARN-1209-5: grpo_vps_step_delta_measured must be True after run."""
    data = json.loads(DELIVERABLE.read_text())
    if data.get("status") == "in_progress":
        pytest.skip("Experiment not yet run")
    assert data["grpo_vps_step_delta_measured"] is True
