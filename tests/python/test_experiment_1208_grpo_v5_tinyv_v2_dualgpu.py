"""Tests for Exp 1208 GRPO v5 TinyV confidence abstention + DualGPU.

Spec: REQ-LEARN-1208, SCENARIO-LEARN-1208, SCENARIO-LEARN-1209,
      SCENARIO-LEARN-1210.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.training import grpo_v5_2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DELIVERABLE = REPO_ROOT / "results" / "experiment_1208_grpo_v5_tinyv_v2_dualgpu.json"


# REQ-LEARN-1208-2 — abstention band
def test_tinyv_confidence_abstain_inside_band_inclusive_low():
    """SCENARIO-LEARN-1208: 0.30 boundary triggers abstention."""
    assert grpo_v5_2.tinyv_confidence_abstain(0.30) is True


def test_tinyv_confidence_abstain_inside_band_inclusive_high():
    """SCENARIO-LEARN-1208: 0.70 boundary triggers abstention."""
    assert grpo_v5_2.tinyv_confidence_abstain(0.70) is True


def test_tinyv_confidence_abstain_below_band_returns_false():
    """REQ-LEARN-1208-2: high-confidence rejection passes through."""
    assert grpo_v5_2.tinyv_confidence_abstain(0.10) is False


def test_tinyv_confidence_abstain_above_band_returns_false():
    """REQ-LEARN-1208-2: high-confidence acceptance passes through."""
    assert grpo_v5_2.tinyv_confidence_abstain(0.90) is False


def test_tinyv_confidence_abstain_nan_treated_as_abstain():
    """REQ-LEARN-1208-2: NaN means no decision; abstain is the safe default."""
    assert grpo_v5_2.tinyv_confidence_abstain(float("nan")) is True


def test_tinyv_confidence_abstain_custom_band():
    """REQ-LEARN-1208-2: callers can widen or narrow the band."""
    assert grpo_v5_2.tinyv_confidence_abstain(0.45, low=0.4, high=0.6) is True
    assert grpo_v5_2.tinyv_confidence_abstain(0.30, low=0.4, high=0.6) is False


# REQ-LEARN-1208-3 — apply_tinyv_abstention
def test_apply_tinyv_abstention_zeroes_uncertain_rewards():
    """SCENARIO-LEARN-1208: rewards inside the band become 0.0."""
    confidences = [0.10, 0.45, 0.55, 0.80, 0.30, 0.70]
    rewards = [1.0, 0.5, 0.7, 0.9, 0.6, 0.8]
    filtered, count = grpo_v5_2.apply_tinyv_abstention(confidences, rewards)
    assert filtered == [1.0, 0.0, 0.0, 0.9, 0.0, 0.0]
    assert count == 4


def test_apply_tinyv_abstention_empty_inputs():
    """REQ-LEARN-1208-3: empty input returns ([], 0) without error."""
    filtered, count = grpo_v5_2.apply_tinyv_abstention([], [])
    assert filtered == []
    assert count == 0


def test_apply_tinyv_abstention_length_mismatch_raises():
    """REQ-LEARN-1208-3: length mismatch fails loudly."""
    with pytest.raises(ValueError, match="length mismatch"):
        grpo_v5_2.apply_tinyv_abstention([0.5, 0.6], [0.1])


def test_apply_tinyv_abstention_no_abstain_passthrough():
    """REQ-LEARN-1208-3: rewards outside the band pass through unchanged."""
    filtered, count = grpo_v5_2.apply_tinyv_abstention([0.05, 0.95], [0.3, 0.4])
    assert filtered == [0.3, 0.4]
    assert count == 0


# REQ-LEARN-1208-7 — verdict mapping
def test_verdict_blocked_no_gpu_offload():
    """REQ-LEARN-1208-1: missing GPU offload blocks before any other check."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=False,
        cuda_device_count=2,
        training_completed=True,
        improvement_over_baseline_pp=20.0,
    )
    assert verdict == "blocked_no_gpu_offload"


def test_verdict_blocked_no_dualgpu():
    """REQ-LEARN-1208-1: cuda_device_count<2 blocks even when GPU offload works."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=True,
        cuda_device_count=1,
        training_completed=True,
        improvement_over_baseline_pp=20.0,
    )
    assert verdict == "blocked_no_dualgpu"


def test_verdict_training_wall_hit_when_incomplete():
    """REQ-LEARN-1208-7: prereqs OK but training_completed=False maps to wall hit."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        training_completed=False,
        improvement_over_baseline_pp=0.0,
    )
    assert verdict == "training_wall_hit"


def test_verdict_improvement_above_v4():
    """SCENARIO-LEARN-1209: positive delta beyond tolerance maps above v4."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        training_completed=True,
        improvement_over_baseline_pp=5.0,
    )
    assert verdict == "improvement_above_v4"


def test_verdict_improvement_below_v4():
    """REQ-LEARN-1208-7: negative delta beyond tolerance maps below v4."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        training_completed=True,
        improvement_over_baseline_pp=-2.0,
    )
    assert verdict == "improvement_below_v4"


def test_verdict_improvement_equal_v4_within_tolerance():
    """REQ-LEARN-1208-7: |delta| <= 0.5pp counts as equal to floor."""
    verdict = grpo_v5_2.derive_grpo_v5_2_honest_verdict(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        training_completed=True,
        improvement_over_baseline_pp=0.3,
    )
    assert verdict == "improvement_equal_v4"


# REQ-LEARN-1208-5 — Spurious Reward threshold + REQ-LEARN-1208-6 fields
def test_build_artifact_fields_above_threshold():
    """SCENARIO-LEARN-1209: 5pp improvement beats the 3pp Spurious threshold."""
    fields = grpo_v5_2.build_grpo_v5_2_artifact_fields(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        dualgpu_confirmed=True,
        model_used="unsloth/Qwen3.6-35B-A3B-GGUF",
        training_completed=True,
        tinyv_abstention_count=4,
        tinyv_abstention_rate=0.4,
        v5_fraction_correct_before=0.40,
        v5_fraction_correct_after=0.55,
        dualgpu_gpu0_utilization_pct=72.0,
        dualgpu_gpu1_utilization_pct=68.0,
    )
    assert math.isclose(fields["improvement_over_baseline_pp"], 5.0, abs_tol=1e-9)
    assert fields["beats_spurious_reward_threshold"] is True
    assert fields["honest_verdict"] == "improvement_above_v4"


def test_build_artifact_fields_below_threshold_but_above_v4():
    """REQ-LEARN-1208-5: a 1pp lift over v4 does NOT beat 3pp Spurious threshold."""
    fields = grpo_v5_2.build_grpo_v5_2_artifact_fields(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        dualgpu_confirmed=True,
        model_used="unsloth/Qwen3.6-35B-A3B-GGUF",
        training_completed=True,
        tinyv_abstention_count=2,
        tinyv_abstention_rate=0.2,
        v5_fraction_correct_before=0.40,
        v5_fraction_correct_after=0.51,
        dualgpu_gpu0_utilization_pct=70.0,
        dualgpu_gpu1_utilization_pct=70.0,
    )
    assert math.isclose(fields["improvement_over_baseline_pp"], 1.0, abs_tol=1e-9)
    assert fields["beats_spurious_reward_threshold"] is False
    assert fields["honest_verdict"] == "improvement_above_v4"


def test_build_artifact_fields_blocked_no_offload_zeroes_metrics():
    """REQ-LEARN-1208-1: blocked artifacts still surface every required field."""
    fields = grpo_v5_2.build_grpo_v5_2_artifact_fields(
        llama_cpp_gpu_offload=False,
        cuda_device_count=2,
        dualgpu_confirmed=False,
        model_used="unsloth/Qwen3.6-35B-A3B-GGUF",
        training_completed=False,
        tinyv_abstention_count=0,
        tinyv_abstention_rate=0.0,
        v5_fraction_correct_before=0.0,
        v5_fraction_correct_after=0.0,
        dualgpu_gpu0_utilization_pct=0.0,
        dualgpu_gpu1_utilization_pct=0.0,
    )
    for key in grpo_v5_2.REQUIRED_GRPO_V5_2_ARTIFACT_FIELDS:
        assert key in fields, f"missing required field: {key}"
    assert fields["honest_verdict"] == "blocked_no_gpu_offload"
    assert fields["beats_spurious_reward_threshold"] is False


def test_build_artifact_fields_required_keys_present():
    """REQ-LEARN-1208-6: artifact builder emits every required field."""
    fields = grpo_v5_2.build_grpo_v5_2_artifact_fields(
        llama_cpp_gpu_offload=True,
        cuda_device_count=2,
        dualgpu_confirmed=True,
        model_used="unsloth/Qwen3.6-35B-A3B-GGUF",
        training_completed=True,
        tinyv_abstention_count=1,
        tinyv_abstention_rate=0.1,
        v5_fraction_correct_before=0.4,
        v5_fraction_correct_after=0.5,
        dualgpu_gpu0_utilization_pct=80.0,
        dualgpu_gpu1_utilization_pct=82.0,
    )
    missing = set(grpo_v5_2.REQUIRED_GRPO_V5_2_ARTIFACT_FIELDS) - set(fields.keys())
    assert not missing, f"REQ-LEARN-1208-6 fields missing: {missing}"


# REQ-LEARN-1208-4 — v4 floor constant
def test_v4_baseline_constant_is_ten_pp():
    """REQ-LEARN-1208-4: the v4 floor encodes Exp 1159's measured +10pp."""
    assert grpo_v5_2.V4_BASELINE_IMPROVEMENT_PP == 10.0


def test_spurious_reward_threshold_is_three_pp():
    """REQ-LEARN-1208-5: the Spurious-Reward threshold is 3pp per arXiv 2506.10947."""
    assert grpo_v5_2.SPURIOUS_REWARD_THRESHOLD_PP == 3.0


# Probe helpers — exercised so the shipped module has full coverage on its
# public surface, not just the spec helpers above.
def test_detect_cuda_device_count_returns_non_negative_int():
    """Live or stubbed, the count must be a non-negative integer."""
    n = grpo_v5_2.detect_cuda_device_count()
    assert isinstance(n, int)
    assert n >= 0


def test_llama_cpp_supports_gpu_offload_returns_bool():
    """Live or stubbed, the probe must return a Python bool."""
    assert isinstance(grpo_v5_2.llama_cpp_supports_gpu_offload(), bool)


# Integration: deliverable artifact must be present and well-formed once
# the experiment script has run. This guards REQ-LEARN-1208-6.
def test_deliverable_artifact_has_all_required_fields():
    """REQ-LEARN-1208-6: results JSON contains every required field."""
    assert DELIVERABLE.exists(), "experiment script must write the deliverable"
    artifact = json.loads(DELIVERABLE.read_text())
    if artifact.get("status") == "in_progress":
        pytest.skip("artifact is the bootstrap skeleton, awaiting experiment run")
    missing = set(grpo_v5_2.REQUIRED_GRPO_V5_2_ARTIFACT_FIELDS) - set(artifact.keys())
    assert not missing, f"deliverable missing required fields: {missing}"
    assert artifact["honest_verdict"] in grpo_v5_2.ALLOWED_HONEST_VERDICTS
