"""Unit tests for :mod:`carnot.training.grpo_vps_training` (Exp 1220).

Spec: REQ-LEARN-1220, SCENARIO-LEARN-1222, SCENARIO-LEARN-1223,
      SCENARIO-LEARN-1224, SCENARIO-LEARN-1225.
"""

from __future__ import annotations

from typing import Any

import pytest

from carnot.training.grpo_vps_training import (
    ALLOWED_GRPO_VPS_VERDICTS,
    REQUIRED_GRPO_VPS_TRAINING_ARTIFACT_FIELDS,
    V4_BASELINE_IMPROVEMENT_PP,
    build_grpo_vps_training_artifact_fields,
    compute_vps_aggregate_reward,
    derive_grpo_vps_honest_verdict,
    mix_phase_b_reward,
    soft_confidence_weight,
)


# ---------------------------------------------------------------------------
# REQ-LEARN-1220-1 / SCENARIO-LEARN-1222: VPS aggregate reward discount
# ---------------------------------------------------------------------------


def test_compute_vps_aggregate_reward_uses_geometric_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1220-1: aggregate equals sum(decay^k * step_reward[k])."""
    fake_steps = [1.0, 0.5, 0.2]
    monkeypatch.setattr(
        "carnot.training.grpo_vps.compute_step_rewards_for_response",
        lambda _response: list(fake_steps),
    )
    result = compute_vps_aggregate_reward("ignored", decay=0.9)
    assert result == pytest.approx(1.0 + 0.9 * 0.5 + 0.81 * 0.2, rel=1e-9)


def test_compute_vps_aggregate_reward_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1220-1: empty segmenter output returns 0.0."""
    monkeypatch.setattr(
        "carnot.training.grpo_vps.compute_step_rewards_for_response",
        lambda _response: [],
    )
    assert compute_vps_aggregate_reward("", decay=0.9) == 0.0


def test_compute_vps_aggregate_reward_decay_one_keeps_all_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1220-1: decay=1.0 sums all step rewards equally."""
    monkeypatch.setattr(
        "carnot.training.grpo_vps.compute_step_rewards_for_response",
        lambda _response: [0.5, 0.5, 0.5],
    )
    assert compute_vps_aggregate_reward("x", decay=1.0) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# REQ-LEARN-1220-2 / SCENARIO-LEARN-1223: soft-confidence weighting
# ---------------------------------------------------------------------------


def test_soft_confidence_weight_elementwise_product() -> None:
    """REQ-LEARN-1220-2: returns elementwise product, no zeroing inside band."""
    rewards = [1.0, 0.0, 1.0]
    confidences = [0.9, 0.5, 0.2]
    assert soft_confidence_weight(rewards, confidences) == [0.9, 0.0, 0.2]


def test_soft_confidence_weight_length_mismatch_raises() -> None:
    """REQ-LEARN-1220-2: length mismatch raises ValueError, not silent."""
    with pytest.raises(ValueError, match="length mismatch"):
        soft_confidence_weight([1.0, 0.5], [0.9])


def test_soft_confidence_weight_returns_floats() -> None:
    """REQ-LEARN-1220-2: integer inputs are coerced to float (JSON-clean)."""
    out = soft_confidence_weight([1, 2], [3, 4])
    assert all(isinstance(x, float) for x in out)
    assert out == [3.0, 8.0]


# ---------------------------------------------------------------------------
# REQ-LEARN-1220-3 / SCENARIO-LEARN-1224: phase-B mix
# ---------------------------------------------------------------------------


def test_mix_phase_b_reward_default_weights() -> None:
    """REQ-LEARN-1220-3: 0.5/0.3/0.2 default mix."""
    result = mix_phase_b_reward(0.8, 0.4, 1.0)
    assert result == pytest.approx(0.5 * 0.8 + 0.3 * 0.4 + 0.2 * 1.0)


def test_mix_phase_b_reward_unbalanced_weights_raise() -> None:
    """REQ-LEARN-1220-3: weights must sum to 1.0 within tolerance."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        mix_phase_b_reward(
            0.5, 0.5, 0.5, w_vps=0.5, w_reflect=0.3, w_correctness=0.3
        )


def test_mix_phase_b_reward_custom_weights_balanced() -> None:
    """REQ-LEARN-1220-3: any convex weights are accepted."""
    result = mix_phase_b_reward(
        1.0, 1.0, 1.0, w_vps=0.7, w_reflect=0.2, w_correctness=0.1
    )
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# REQ-LEARN-1220-4 / SCENARIO-LEARN-1225: verdict mapping
# ---------------------------------------------------------------------------


def test_derive_verdict_beats_v4() -> None:
    """REQ-LEARN-1220-4: improvement > 10.0 -> beats_v4."""
    assert (
        derive_grpo_vps_honest_verdict(
            12.0, training_completed=True, prereq_ok=True
        )
        == "vps_training_beats_v4"
    )


def test_derive_verdict_matches_v4() -> None:
    """REQ-LEARN-1220-4: improvement in [0, 10] -> matches_v4."""
    assert (
        derive_grpo_vps_honest_verdict(
            5.0, training_completed=True, prereq_ok=True
        )
        == "vps_training_matches_v4"
    )
    # Boundary: exactly 10.0 still maps to matches (strict greater-than gate).
    assert (
        derive_grpo_vps_honest_verdict(
            10.0, training_completed=True, prereq_ok=True
        )
        == "vps_training_matches_v4"
    )
    assert (
        derive_grpo_vps_honest_verdict(
            0.0, training_completed=True, prereq_ok=True
        )
        == "vps_training_matches_v4"
    )


def test_derive_verdict_below_v4() -> None:
    """REQ-LEARN-1220-4: improvement < 0 -> below_v4."""
    assert (
        derive_grpo_vps_honest_verdict(
            -3.0, training_completed=True, prereq_ok=True
        )
        == "vps_training_below_v4"
    )


def test_derive_verdict_wall_hit() -> None:
    """REQ-LEARN-1220-4: training_completed=False with prereq_ok -> wall_hit."""
    assert (
        derive_grpo_vps_honest_verdict(
            0.0, training_completed=False, prereq_ok=True
        )
        == "training_wall_hit"
    )


def test_derive_verdict_blocked_no_gpu() -> None:
    """REQ-LEARN-1220-4: prereq_ok=False -> blocked_no_gpu (overrides all)."""
    for completed, imp in [(True, 50.0), (False, 0.0), (True, -100.0)]:
        assert (
            derive_grpo_vps_honest_verdict(
                imp, training_completed=completed, prereq_ok=False
            )
            == "blocked_no_gpu"
        )


def test_all_verdicts_are_in_allowed_set() -> None:
    """REQ-LEARN-1220-4: every emitted verdict is in the allow-list."""
    cases = [
        (12.0, True, True),
        (5.0, True, True),
        (-3.0, True, True),
        (0.0, False, True),
        (50.0, True, False),
    ]
    for imp, completed, prereq in cases:
        v = derive_grpo_vps_honest_verdict(
            imp, training_completed=completed, prereq_ok=prereq
        )
        assert v in ALLOWED_GRPO_VPS_VERDICTS


# ---------------------------------------------------------------------------
# REQ-LEARN-1220-5 / -6: artifact field block
# ---------------------------------------------------------------------------


def _live_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "llama_cpp_gpu_offload": True,
        "cuda_device_count": 2,
        "model_used": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "exp1219_fix_applied": "soft_weight + n_train>=32 + balanced_holdout",
        "training_completed": True,
        "n_training_questions": 200,
        "n_eval_questions": 200,
        "grpo_vps_fraction_correct_before": 0.50,
        "grpo_vps_fraction_correct_after": 0.65,
    }
    base.update(overrides)
    return base


def test_build_artifact_includes_every_required_field() -> None:
    """REQ-LEARN-1220-5: every required field is present after build."""
    fields = build_grpo_vps_training_artifact_fields(**_live_kwargs())
    for name in REQUIRED_GRPO_VPS_TRAINING_ARTIFACT_FIELDS:
        assert name in fields, f"missing required field: {name}"


def test_build_artifact_improvement_pp_arithmetic() -> None:
    """REQ-LEARN-1220-5: improvement_pp = 100 * (after - before)."""
    fields = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(
            grpo_vps_fraction_correct_before=0.40,
            grpo_vps_fraction_correct_after=0.55,
        )
    )
    assert fields["grpo_vps_improvement_pp"] == pytest.approx(15.0)


def test_build_artifact_beats_v4_strict_gt() -> None:
    """REQ-LEARN-1220-6: beats_v4_floor uses strict greater-than 10.0."""
    above = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(
            grpo_vps_fraction_correct_before=0.40,
            grpo_vps_fraction_correct_after=0.55,
        )
    )
    assert above["beats_v4_floor"] is True
    assert above["honest_verdict"] == "vps_training_beats_v4"

    exactly_at = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(
            grpo_vps_fraction_correct_before=0.40,
            grpo_vps_fraction_correct_after=0.50,
        )
    )
    assert exactly_at["beats_v4_floor"] is False
    assert exactly_at["honest_verdict"] == "vps_training_matches_v4"


def test_build_artifact_v4_baseline_is_locked() -> None:
    """REQ-LEARN-1220-6: v4 floor is hard-coded at 10.0."""
    fields = build_grpo_vps_training_artifact_fields(**_live_kwargs())
    assert fields["v4_baseline_improvement_pp"] == V4_BASELINE_IMPROVEMENT_PP == 10.0


def test_build_artifact_blocked_when_prereq_fails() -> None:
    """REQ-LEARN-1220-4: missing GPU collapses verdict to blocked_no_gpu."""
    fields = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(
            llama_cpp_gpu_offload=False,
            cuda_device_count=0,
            training_completed=False,
            grpo_vps_fraction_correct_before=0.0,
            grpo_vps_fraction_correct_after=0.0,
        )
    )
    assert fields["honest_verdict"] == "blocked_no_gpu"


def test_build_artifact_wall_hit_branch() -> None:
    """REQ-LEARN-1220-4: prereq_ok + not completed -> training_wall_hit."""
    fields = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(
            training_completed=False,
            grpo_vps_fraction_correct_before=0.0,
            grpo_vps_fraction_correct_after=0.0,
        )
    )
    assert fields["honest_verdict"] == "training_wall_hit"


def test_build_artifact_grpo_vps_training_completed_mirrors() -> None:
    """REQ-LEARN-1220-5: grpo_vps_training_completed mirrors training_completed."""
    fields = build_grpo_vps_training_artifact_fields(**_live_kwargs())
    assert fields["grpo_vps_training_completed"] == fields["training_completed"]


def test_build_artifact_single_cuda_device_blocks() -> None:
    """REQ-LEARN-1220-4: cuda_device_count<2 collapses to blocked_no_gpu."""
    fields = build_grpo_vps_training_artifact_fields(
        **_live_kwargs(cuda_device_count=1, training_completed=False)
    )
    assert fields["honest_verdict"] == "blocked_no_gpu"
