"""Tests for Exp 1184 GRPO v5 continuous TinyV v2 reward + DualGPU split.

Spec: REQ-LEARN-1184, SCENARIO-LEARN-1184, SCENARIO-LEARN-1185,
      SCENARIO-LEARN-1186.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from carnot.training import grpo_v5

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1184_grpo_v5_tinyv_v2_dualGPU.py"
DELIVERABLE = REPO_ROOT / "results" / "experiment_1184_grpo_v5_tinyv_v2_dualGPU.json"


def _load_script_module():
    """Load scripts/experiment_1184... as a fresh module each call.

    Test isolation: tests monkey-patch ``detect_cuda_device_count`` and
    ``llama_cpp_supports_gpu_offload`` on the loaded module, so each
    test loads its own copy to avoid cross-test pollution.
    """
    spec = importlib.util.spec_from_file_location("exp1184", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1184"] = module
    spec.loader.exec_module(module)
    return module


def test_continuous_tinyv_v2_reward_mixes_energy_and_reflection_with_default_weights():
    """SCENARIO-LEARN-1184: r_total = 0.6 * r_energy + 0.4 * r_reflect."""
    reward = grpo_v5.continuous_tinyv_v2_reward(0.8, 0.5)
    assert math.isclose(reward, 0.68, abs_tol=1e-12)


def test_continuous_tinyv_v2_reward_with_custom_weights():
    """REQ-LEARN-1184-2: weights are configurable, must sum to 1.0."""
    weights = grpo_v5.TinyVV2Weights(energy_weight=0.7, reflection_weight=0.3)
    reward = grpo_v5.continuous_tinyv_v2_reward(1.0, 0.0, weights=weights)
    assert math.isclose(reward, 0.7, abs_tol=1e-12)


def test_tinyv_v2_weights_reject_negative_values():
    """REQ-LEARN-1184-2: weights must be non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        grpo_v5.TinyVV2Weights(energy_weight=-0.1, reflection_weight=1.1)


def test_tinyv_v2_weights_reject_non_unit_sum():
    """REQ-LEARN-1184-2: weights must sum to 1.0 within 1e-9."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        grpo_v5.TinyVV2Weights(energy_weight=0.5, reflection_weight=0.4)


def test_continuous_tinyv_v2_reward_group_aligns_lengths():
    """REQ-LEARN-1184-3: aligned reward groups apply per-completion mix."""
    rewards = grpo_v5.continuous_tinyv_v2_reward_group(
        [0.8, 0.4, 0.0],
        [0.5, 1.0, -1.0],
    )
    assert rewards == [
        round(0.6 * 0.8 + 0.4 * 0.5, 12),
        round(0.6 * 0.4 + 0.4 * 1.0, 12),
        round(0.6 * 0.0 + 0.4 * -1.0, 12),
    ]


def test_continuous_tinyv_v2_reward_group_length_mismatch_raises():
    """REQ-LEARN-1184-3: misaligned reward groups must fail loudly."""
    with pytest.raises(ValueError, match="length mismatch"):
        grpo_v5.continuous_tinyv_v2_reward_group([0.5, 0.6], [0.1])


def test_derive_honest_verdict_prereq_blocks_first():
    """SCENARIO-LEARN-1185: missing GPU prereq dominates other state."""
    verdict = grpo_v5.derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=False,
        training_completed=True,
        grpo_v5_delta_pp=0.99,
    )
    assert verdict == "gpu_offload_prerequisite_not_met"


def test_derive_honest_verdict_above_v4_when_delta_positive():
    """SCENARIO-LEARN-1186: positive delta over tolerance => above_v4."""
    verdict = grpo_v5.derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=True,
        training_completed=True,
        grpo_v5_delta_pp=0.06,
    )
    assert verdict == "grpo_v5_above_v4"


def test_derive_honest_verdict_no_delta_within_tolerance():
    """REQ-LEARN-1184-6: small absolute delta resolves to no_delta."""
    verdict = grpo_v5.derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=True,
        training_completed=True,
        grpo_v5_delta_pp=0.003,
    )
    assert verdict == "grpo_v5_no_delta"


def test_derive_honest_verdict_regression_when_delta_negative():
    """REQ-LEARN-1184-6: negative delta beyond tolerance => regression."""
    verdict = grpo_v5.derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=True,
        training_completed=True,
        grpo_v5_delta_pp=-0.04,
    )
    assert verdict == "grpo_v5_regression_vs_v4"


def test_derive_honest_verdict_training_wall_hit_when_incomplete():
    """REQ-LEARN-1184-6: prereq met but training incomplete => wall_hit."""
    verdict = grpo_v5.derive_grpo_v5_honest_verdict(
        gpu_offload_prerequisite_met=True,
        training_completed=False,
        grpo_v5_delta_pp=0.0,
    )
    assert verdict == "training_wall_hit"


def test_build_artifact_fields_includes_required_keys_and_correct_delta():
    """REQ-LEARN-1184-5: all required fields present with computed delta."""
    fields = grpo_v5.build_grpo_v5_artifact_fields(
        gpu_offload_prerequisite_met=True,
        training_completed=True,
        dualgpu_confirmed=True,
        training_tokens_per_sec=72.5,
        grpo_v5_pass_rate=0.32,
        tinyv_v2_mean_reward=0.71,
        n_eval_questions=47,
    )
    for key in grpo_v5.REQUIRED_GRPO_V5_ARTIFACT_FIELDS:
        assert key in fields
    assert fields["grpo_v4_baseline_pass_rate"] == 0.26
    assert math.isclose(fields["grpo_v5_delta_pp"], 0.06, abs_tol=1e-9)
    assert fields["honest_verdict"] == "grpo_v5_above_v4"
    assert fields["n_eval_questions"] == 47


def test_build_artifact_fields_blocked_reports_prereq_not_met():
    """SCENARIO-LEARN-1185: blocked artifact fields use the prereq verdict."""
    fields = grpo_v5.build_grpo_v5_artifact_fields(
        gpu_offload_prerequisite_met=False,
        training_completed=False,
        dualgpu_confirmed=False,
        training_tokens_per_sec=0.0,
        grpo_v5_pass_rate=0.0,
        tinyv_v2_mean_reward=0.0,
        n_eval_questions=0,
    )
    assert fields["honest_verdict"] == "gpu_offload_prerequisite_not_met"
    assert fields["gpu_offload_prerequisite_met"] is False
    assert fields["dualgpu_confirmed"] is False
    assert fields["training_completed"] is False


def test_gpu_offload_prerequisite_met_requires_both_checks():
    """REQ-LEARN-1184-1: both CUDA count and llama.cpp offload are needed."""
    assert grpo_v5.gpu_offload_prerequisite_met(
        cuda_device_count=2,
        llama_cpp_gpu_offload=True,
    )
    assert not grpo_v5.gpu_offload_prerequisite_met(
        cuda_device_count=2,
        llama_cpp_gpu_offload=False,
    )
    assert not grpo_v5.gpu_offload_prerequisite_met(
        cuda_device_count=1,
        llama_cpp_gpu_offload=True,
    )
    assert not grpo_v5.gpu_offload_prerequisite_met(
        cuda_device_count=0,
        llama_cpp_gpu_offload=False,
    )


def test_llama_cpp_supports_gpu_offload_returns_bool_without_crashing():
    """Probe is wrapped in try/except so missing torch/llama_cpp won't crash."""
    result = grpo_v5.llama_cpp_supports_gpu_offload()
    assert isinstance(result, bool)


def test_detect_cuda_device_count_returns_non_negative_int():
    """detect_cuda_device_count must always return a usable int sentinel."""
    count = grpo_v5.detect_cuda_device_count()
    assert isinstance(count, int)
    assert count >= 0


def test_script_blocked_when_llama_cpp_lacks_gpu_offload(monkeypatch):
    """SCENARIO-LEARN-1185: CPU-only llama.cpp blocks training honestly."""
    exp1184 = _load_script_module()
    monkeypatch.setattr(exp1184, "detect_cuda_device_count", lambda: 2)
    monkeypatch.setattr(exp1184, "llama_cpp_supports_gpu_offload", lambda: False)

    artifact = exp1184._run_experiment()

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "gpu_offload_prerequisite_not_met"
    assert artifact["gpu_offload_prerequisite_met"] is False
    assert artifact["dualgpu_confirmed"] is False
    assert artifact["training_completed"] is False
    assert artifact["llama_cpp_gpu_offload"] is False
    assert artifact["cuda_device_count"] == 2
    for key in grpo_v5.REQUIRED_GRPO_V5_ARTIFACT_FIELDS:
        assert key in artifact


def test_script_blocked_when_only_one_cuda_device_visible(monkeypatch):
    """REQ-LEARN-1184-7: <2 GPUs cannot host the DualGPU split."""
    exp1184 = _load_script_module()
    monkeypatch.setattr(exp1184, "detect_cuda_device_count", lambda: 1)
    monkeypatch.setattr(exp1184, "llama_cpp_supports_gpu_offload", lambda: True)

    artifact = exp1184._run_experiment()

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "gpu_offload_prerequisite_not_met"
    assert artifact["dualgpu_confirmed"] is False
    assert "only 1 CUDA device" in artifact["blocked_reason"]
    assert artifact["llama_cpp_gpu_offload"] is True


def test_script_blocked_when_both_checks_fail(monkeypatch):
    """REQ-LEARN-1184-1: combined failure surfaces a combined reason."""
    exp1184 = _load_script_module()
    monkeypatch.setattr(exp1184, "detect_cuda_device_count", lambda: 0)
    monkeypatch.setattr(exp1184, "llama_cpp_supports_gpu_offload", lambda: False)

    artifact = exp1184._run_experiment()

    assert artifact["status"] == "blocked"
    assert "lacks GPU offload" in artifact["blocked_reason"]
    assert "fewer than two CUDA" in artifact["blocked_reason"]


def test_script_live_path_reached_when_prereq_passes(monkeypatch):
    """REQ-LEARN-1184-7: prereq-true path uses dualgpu_confirmed=True."""
    exp1184 = _load_script_module()
    monkeypatch.setattr(exp1184, "detect_cuda_device_count", lambda: 2)
    monkeypatch.setattr(exp1184, "llama_cpp_supports_gpu_offload", lambda: True)

    artifact = exp1184._run_experiment()

    assert artifact["gpu_offload_prerequisite_met"] is True
    assert artifact["dualgpu_confirmed"] is True
    assert artifact["cuda_device_count"] == 2
    assert artifact["llama_cpp_gpu_offload"] is True
    assert artifact["tensor_split"] == [0.5, 0.5]
    assert artifact["main_gpu"] == 0
    assert artifact["n_gpu_layers"] == -1
    for key in grpo_v5.REQUIRED_GRPO_V5_ARTIFACT_FIELDS:
        assert key in artifact


def test_script_main_writes_deliverable_with_required_schema(monkeypatch, tmp_path):
    """REQ-LEARN-1184-5: main() writes a JSON artifact with all required fields."""
    exp1184 = _load_script_module()
    monkeypatch.setattr(exp1184, "detect_cuda_device_count", lambda: 0)
    monkeypatch.setattr(exp1184, "llama_cpp_supports_gpu_offload", lambda: False)
    target = tmp_path / "exp1184.json"
    monkeypatch.setattr(exp1184, "DELIVERABLE", target)

    rc = exp1184.main()
    assert rc == 0
    payload = json.loads(target.read_text())
    for key in grpo_v5.REQUIRED_GRPO_V5_ARTIFACT_FIELDS:
        assert key in payload
    assert payload["honest_verdict"] in grpo_v5.ALLOWED_HONEST_VERDICTS
    assert payload["experiment"] == 1184


def test_real_deliverable_has_required_schema():
    """The on-disk artifact must satisfy REQ-LEARN-1184-5.

    The conductor invariant is that ``main()`` writes the deliverable
    before exiting, and ``main()`` raises if any required field is
    missing — so the file must always exist at this point in the
    suite. We re-verify the schema here as a belt-and-suspenders check
    that the on-disk file matches what the test loaded modules promise.
    """
    assert DELIVERABLE.exists(), f"deliverable missing: {DELIVERABLE}"
    payload = json.loads(DELIVERABLE.read_text())
    for key in grpo_v5.REQUIRED_GRPO_V5_ARTIFACT_FIELDS:
        assert key in payload, f"missing required field: {key}"
    assert payload["honest_verdict"] in grpo_v5.ALLOWED_HONEST_VERDICTS
    assert payload["experiment"] == 1184
