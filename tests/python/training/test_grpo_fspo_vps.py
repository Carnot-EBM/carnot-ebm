"""Tests for GRPO-v6 FSPO per-token factuality weighting module.

Spec: REQ-LEARN-1221, SCENARIO-LEARN-1226, SCENARIO-LEARN-1227, SCENARIO-LEARN-1228
"""

from __future__ import annotations

import math

import pytest

from carnot.training.grpo_fspo_vps import (
    REQUIRED_GRPO_V6_TOKEN_REG_ARTIFACT_FIELDS,
    build_grpo_v6_token_reg_artifact_fields,
    compute_grpo_v6_token_metrics,
    compute_fspo_vps_advantage,
    compute_token_regulated_fspo_vps_advantage,
    derive_fspo_honest_verdict,
    derive_grpo_v6_token_reg_honest_verdict,
    select_best_completion,
)


# ---------------------------------------------------------------------------
# compute_fspo_vps_advantage
# ---------------------------------------------------------------------------


def test_fspo_advantage_single_step_broadcasts_to_all_tokens() -> None:
    """SCENARIO-LEARN-1226: token advantages inherit step factuality score.

    A single step with reward=0.8, factuality=0.5, tokens=3 should produce
    three identical advantage values.  With only one step the normalised reward
    equals the raw reward (std=0 path returns raw rewards), so the advantage
    per token is 0.8 * 0.5 = 0.4.

    Spec: REQ-LEARN-1221-1, SCENARIO-LEARN-1226
    """
    advantages = compute_fspo_vps_advantage(
        step_rewards=[0.8],
        factuality_scores=[0.5],
        tokens_per_step=[3],
    )
    assert len(advantages) == 3
    for adv in advantages:
        assert abs(adv - 0.8 * 0.5) < 1e-9, f"expected 0.4, got {adv}"


def test_fspo_advantage_multiple_steps_length_matches_token_sum() -> None:
    """Total token count equals sum of tokens_per_step.

    Spec: REQ-LEARN-1221-1
    """
    advantages = compute_fspo_vps_advantage(
        step_rewards=[1.0, 0.0],
        factuality_scores=[1.0, 0.5],
        tokens_per_step=[2, 5],
    )
    assert len(advantages) == 7


def test_fspo_advantage_normalises_when_variance_nonzero() -> None:
    """When step rewards differ, z-score normalisation is applied.

    Spec: REQ-LEARN-1221-1
    """
    # Two steps with rewards [1.0, 0.0]. mean=0.5, std=0.5.
    # normalised[0] = (1.0-0.5)/0.5 = 1.0
    # normalised[1] = (0.0-0.5)/0.5 = -1.0
    # factuality=[1.0, 1.0] so advantages: [1.0*1.0, -1.0*1.0]
    advantages = compute_fspo_vps_advantage(
        step_rewards=[1.0, 0.0],
        factuality_scores=[1.0, 1.0],
        tokens_per_step=[1, 1],
    )
    assert abs(advantages[0] - 1.0) < 1e-9
    assert abs(advantages[1] - (-1.0)) < 1e-9


def test_fspo_advantage_factuality_zero_zeros_out_token() -> None:
    """A step with factuality_score=0.0 produces zero advantage tokens.

    Spec: REQ-LEARN-1221-1
    """
    advantages = compute_fspo_vps_advantage(
        step_rewards=[1.0, 0.5],
        factuality_scores=[1.0, 0.0],
        tokens_per_step=[2, 3],
    )
    # Tokens in step 1 (factuality=0) should all be 0.0.
    for adv in advantages[2:]:
        assert abs(adv) < 1e-9


def test_fspo_advantage_empty_returns_empty() -> None:
    """Empty input produces empty output without error.

    Spec: REQ-LEARN-1221-1
    """
    assert compute_fspo_vps_advantage([], [], []) == []


def test_fspo_advantage_length_mismatch_raises() -> None:
    """Mismatched list lengths raise ValueError.

    Spec: REQ-LEARN-1221-1
    """
    with pytest.raises(ValueError, match="same length"):
        compute_fspo_vps_advantage([1.0, 0.5], [1.0], [2, 3])


# ---------------------------------------------------------------------------
# select_best_completion
# ---------------------------------------------------------------------------


def test_select_best_completion_picks_highest_sum() -> None:
    """SCENARIO-LEARN-1227: completion with highest total advantage wins.

    Spec: REQ-LEARN-1221-2, SCENARIO-LEARN-1227
    """
    completions = ["A", "B"]
    advantages = [[0.6, 0.6], [0.2, 0.2]]  # sums: 1.2 vs 0.4
    result = select_best_completion(completions, advantages)
    assert result == "A"


def test_select_best_completion_tie_returns_first() -> None:
    """Ties broken by returning the first maximally-advantaged completion.

    Spec: REQ-LEARN-1221-2
    """
    completions = ["X", "Y", "Z"]
    advantages = [[0.5], [0.5], [0.3]]
    result = select_best_completion(completions, advantages)
    assert result == "X"


def test_select_best_completion_single_candidate() -> None:
    """Single completion always wins regardless of advantage.

    Spec: REQ-LEARN-1221-2
    """
    result = select_best_completion(["only"], [[-0.5, -0.5]])
    assert result == "only"


def test_select_best_completion_empty_raises() -> None:
    """Empty completions list raises ValueError.

    Spec: REQ-LEARN-1221-2
    """
    with pytest.raises(ValueError, match="empty"):
        select_best_completion([], [])


def test_select_best_completion_length_mismatch_raises() -> None:
    """Mismatched lengths raise ValueError.

    Spec: REQ-LEARN-1221-2
    """
    with pytest.raises(ValueError, match="same length"):
        select_best_completion(["A", "B"], [[0.5]])


def test_select_best_completion_last_wins() -> None:
    """When the last completion has the highest score it is selected.

    Exercises the score > best_score branch inside the loop.

    Spec: REQ-LEARN-1221-2
    """
    completions = ["low", "medium", "high"]
    advantages = [[0.1], [0.5], [0.9]]
    result = select_best_completion(completions, advantages)
    assert result == "high"


# ---------------------------------------------------------------------------
# derive_fspo_honest_verdict
# ---------------------------------------------------------------------------


def test_derive_fspo_honest_verdict_positive_delta() -> None:
    """SCENARIO-LEARN-1228: positive delta maps to fspo_improves_over_vps.

    Spec: REQ-LEARN-1221-3, SCENARIO-LEARN-1228
    """
    assert derive_fspo_honest_verdict(3.0) == "fspo_improves_over_vps"


def test_derive_fspo_honest_verdict_negative_delta() -> None:
    """SCENARIO-LEARN-1228: negative delta maps to fspo_degrades_vps.

    Spec: REQ-LEARN-1221-3, SCENARIO-LEARN-1228
    """
    assert derive_fspo_honest_verdict(-2.0) == "fspo_degrades_vps"


def test_derive_fspo_honest_verdict_zero_delta() -> None:
    """Zero delta maps to fspo_matches_vps.

    Spec: REQ-LEARN-1221-3
    """
    assert derive_fspo_honest_verdict(0.0) == "fspo_matches_vps"


def test_derive_fspo_honest_verdict_small_positive() -> None:
    """Even a tiny positive delta maps to fspo_improves_over_vps.

    Spec: REQ-LEARN-1221-3
    """
    assert derive_fspo_honest_verdict(0.0001) == "fspo_improves_over_vps"


# ---------------------------------------------------------------------------
# GRPO-v6 token regulation
# ---------------------------------------------------------------------------


def test_token_regulated_advantage_upweights_high_logprob_token() -> None:
    """SCENARIO-LEARN-1235: high-confidence tokens receive more signal.

    Spec: REQ-LEARN-1235-1, REQ-LEARN-1235-2, SCENARIO-LEARN-1235
    """
    advantages = compute_token_regulated_fspo_vps_advantage(
        step_causal_scores=[0.5],
        step_z3_scores=[0.25],
        tokens_per_step=[2],
        token_logprobs=[-0.1, -4.0],
    )

    raw_advantage = (0.5 + 0.25) * 0.5
    expected_high_weight = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-4.0))

    assert len(advantages) == 2
    assert advantages[0] > advantages[1]
    assert math.isclose(advantages[0], raw_advantage * expected_high_weight)
    assert math.isclose(sum(advantages), raw_advantage)


def test_token_regulated_advantage_zeroes_negative_factuality() -> None:
    """REQ-LEARN-1235-1: max(0, causal_score) gates factuality weight."""
    advantages = compute_token_regulated_fspo_vps_advantage(
        step_causal_scores=[-0.25],
        step_z3_scores=[1.0],
        tokens_per_step=[3],
        token_logprobs=[-0.5, -0.5, -0.5],
    )

    assert advantages == [0.0, 0.0, 0.0]


def test_token_regulated_advantage_length_mismatch_raises() -> None:
    """REQ-LEARN-1235-1: per-step and per-token inputs must align."""
    with pytest.raises(ValueError, match="token_logprobs length"):
        compute_token_regulated_fspo_vps_advantage(
            step_causal_scores=[0.5],
            step_z3_scores=[0.5],
            tokens_per_step=[2],
            token_logprobs=[-1.0],
        )


def test_token_regulated_advantage_step_length_mismatch_raises() -> None:
    """REQ-LEARN-1235-1: all per-step arrays must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        compute_token_regulated_fspo_vps_advantage(
            step_causal_scores=[0.5, 0.25],
            step_z3_scores=[0.5],
            tokens_per_step=[1, 1],
            token_logprobs=[-1.0, -1.0],
        )


def test_token_regulated_advantage_empty_inputs_return_empty() -> None:
    """REQ-LEARN-1235-1: empty completions produce no token advantages."""
    assert (
        compute_token_regulated_fspo_vps_advantage(
            step_causal_scores=[],
            step_z3_scores=[],
            tokens_per_step=[],
            token_logprobs=[],
        )
        == []
    )


def test_token_regulated_advantage_zero_token_step_returns_empty() -> None:
    """REQ-LEARN-1235-2: a zero-token step has no regulated token signal."""
    assert (
        compute_token_regulated_fspo_vps_advantage(
            step_causal_scores=[0.5],
            step_z3_scores=[0.5],
            tokens_per_step=[0],
            token_logprobs=[],
        )
        == []
    )


def test_token_regulated_advantage_negative_token_count_raises() -> None:
    """REQ-LEARN-1235-1: token counts cannot be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        compute_token_regulated_fspo_vps_advantage(
            step_causal_scores=[0.5, 0.5],
            step_z3_scores=[0.5, 0.5],
            tokens_per_step=[-1, 1],
            token_logprobs=[],
        )


def test_grpo_v6_token_metrics_report_logprob_mean_and_fspo_coverage() -> None:
    """REQ-LEARN-1235-3: metrics cover logprob quality and FSPO coverage."""
    metrics = compute_grpo_v6_token_metrics(
        step_causal_scores=[0.5, -0.1, 0.0],
        token_logprobs=[-1.0, None, -3.0, -5.0],
    )

    assert math.isclose(metrics["mean_token_logprob"], -3.0)
    assert math.isclose(metrics["fspo_coverage_fraction"], 1.0 / 3.0)


def test_grpo_v6_token_verdict_maps_wall_budget_and_delta() -> None:
    """SCENARIO-LEARN-1236: coverage gate precedes delta verdicts.

    Spec: REQ-LEARN-1235-4, SCENARIO-LEARN-1236
    """
    assert (
        derive_grpo_v6_token_reg_honest_verdict(
            grpo_v6_improvement_pp=0.1,
            n_questions_evaluated=9,
            prereq_ok=True,
        )
        == "wall_budget_still_insufficient"
    )
    assert (
        derive_grpo_v6_token_reg_honest_verdict(
            grpo_v6_improvement_pp=0.1,
            n_questions_evaluated=13,
            prereq_ok=True,
        )
        == "grpo_v6_beats_vps"
    )
    assert (
        derive_grpo_v6_token_reg_honest_verdict(
            grpo_v6_improvement_pp=0.0,
            n_questions_evaluated=13,
            prereq_ok=True,
        )
        == "grpo_v6_no_improvement"
    )
    assert (
        derive_grpo_v6_token_reg_honest_verdict(
            grpo_v6_improvement_pp=-0.1,
            n_questions_evaluated=13,
            prereq_ok=True,
        )
        == "grpo_v6_regression"
    )
    assert (
        derive_grpo_v6_token_reg_honest_verdict(
            grpo_v6_improvement_pp=2.0,
            n_questions_evaluated=13,
            prereq_ok=False,
        )
        == "blocked"
    )


def test_build_grpo_v6_token_reg_artifact_fields_required_schema() -> None:
    """REQ-LEARN-1235-5: artifact helper emits all required fields."""
    artifact = build_grpo_v6_token_reg_artifact_fields(
        wall_budget_s=1200,
        n_questions_evaluated=13,
        vps_baseline_accuracy=0.95,
        fspo_vps_accuracy=0.96,
        mean_token_logprob=-2.5,
        fspo_coverage_fraction=0.75,
        dualgpu_confirmed=True,
        prereq_ok=True,
    )

    for field in REQUIRED_GRPO_V6_TOKEN_REG_ARTIFACT_FIELDS:
        assert field in artifact
    assert math.isclose(artifact["grpo_v6_improvement_pp"], 1.0)
    assert artifact["beats_vps_floor"] is True
    assert artifact["grpo_v6_improvement_measured"] is True
    assert artifact["honest_verdict"] == "grpo_v6_beats_vps"
