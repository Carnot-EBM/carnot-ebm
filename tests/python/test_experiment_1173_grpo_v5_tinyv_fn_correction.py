"""Tests for Exp 1173 GRPO v5 TinyV false-negative correction.

Spec: REQ-LEARN-1173, SCENARIO-LEARN-1173, SCENARIO-LEARN-1174,
      SCENARIO-LEARN-1175, SCENARIO-LEARN-1176.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from carnot.training import grpo_reflection_reward as grr
from carnot.training import grpo_structural_warmup as gsw


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1173_grpo_v5_tinyv_fn_correction.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp1173", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1173"] = module
    spec.loader.exec_module(module)
    return module


def test_tinyv_abstention_zeroes_uncertain_reward_and_keeps_raw_reward():
    """SCENARIO-LEARN-1173: uncertain ThinkPRM confidence emits zero reward."""
    result = grr.combine_rewards_with_tinyv_abstention(
        0.5,
        0.8,
        fn_abstain_thresh_low=0.3,
        fn_abstain_thresh_high=0.7,
    )

    assert result.abstained is True
    assert result.thinkprm_confidence == 0.5
    assert math.isclose(result.raw_reward, 0.74)
    assert result.emitted_reward == 0.0


def test_tinyv_abstention_thresholds_are_inclusive_and_group_rate_is_counted():
    """REQ-LEARN-1173-3/4: thresholds are inclusive and rate counts samples."""
    group = grr.combine_reward_groups_with_tinyv_abstention(
        [0.29, 0.3, 0.7, 0.71],
        [1.0, 1.0, -1.0, 0.0],
        fn_abstain_thresh_low=0.3,
        fn_abstain_thresh_high=0.7,
    )

    assert group.abstained == [False, True, True, False]
    assert group.rewards == [0.59, 0.0, 0.0, 0.71]
    assert group.raw_rewards == [0.59, 0.6, 0.4, 0.71]
    assert group.fn_abstention_rate == 0.5


def test_tinyv_group_length_mismatch_raises():
    """REQ-LEARN-1173-4: TinyV group diagnostics preserve completion alignment."""
    with pytest.raises(ValueError, match="length mismatch"):
        grr.combine_reward_groups_with_tinyv_abstention([0.5, 0.6], [0.1])


def test_tinyv_threshold_validation_rejects_invalid_intervals():
    """REQ-LEARN-1173-3: TinyV thresholds must be ordered probabilities."""
    with pytest.raises(ValueError, match="0.0 <= low <= high <= 1.0"):
        grr.tinyv_confidence_abstains(0.5, low=0.8, high=0.2)
    with pytest.raises(ValueError, match="0.0 <= low <= high <= 1.0"):
        grr.tinyv_confidence_abstains(0.5, low=-0.1, high=0.7)
    with pytest.raises(ValueError, match="0.0 <= low <= high <= 1.0"):
        grr.tinyv_confidence_abstains(0.5, low=0.3, high=1.2)


def test_v5_phase_rewards_keep_warmup_reflection_only_then_abstain_in_full(monkeypatch):
    """SCENARIO-LEARN-1174: warm-up ignores TinyV; full phase applies it."""
    exp1173 = _load_script_module()
    warmup, full = gsw.build_structural_warmup_phase_configs()

    class FakeEvaluator:
        def score(self, question: str, response: str) -> grr.ReflectionRewardResult:
            return grr.ReflectionRewardResult(
                energy_before=2.0,
                energy_after=1.0,
                reward=0.5,
                repaired_response=response,
                repair_attempted=True,
                clipped=False,
            )

    def fake_score(_response: str, _question: str) -> float:
        return 0.5

    monkeypatch.setattr(exp1173.exp1129, "thinkprm_v2_score", fake_score)

    warmup_record = exp1173.score_completion_for_phase("c", "q", warmup, FakeEvaluator())
    full_record = exp1173.score_completion_for_phase("c", "q", full, FakeEvaluator())

    assert warmup_record["total_reward"] == 0.5
    assert warmup_record["raw_total_reward"] == 0.5
    assert warmup_record["thinkprm_score"] == 0.0
    assert warmup_record["tinyv_abstained"] is False

    assert full_record["total_reward"] == 0.0
    assert full_record["raw_total_reward"] == 0.65
    assert full_record["thinkprm_score"] == 0.5
    assert full_record["tinyv_abstained"] is True


def test_v5_artifact_fields_and_verdict_mapping():
    """REQ-LEARN-1173-5/6: v5 artifact fields and verdicts are canonical."""
    fields = grr.build_tinyv_artifact_fields(
        cuda_device_count=2,
        dualgpu_confirmed=True,
        training_completed=True,
        training_wall_budget_hit=False,
        advantage_stdev_warmup=0.2,
        advantage_stdev_full=0.25,
        n_eval_questions=50,
        baseline_fraction_correct=0.2,
        trained_fraction_correct=0.34,
        improvement_over_baseline=0.14,
        fn_abstention_rate=0.125,
        fn_threshold_tuned=0.3,
        fn_abstain_thresh_high=0.7,
    )

    for key in grr.REQUIRED_TINYV_ARTIFACT_FIELDS:
        assert key in fields
    assert fields["v4_baseline"] == 0.10
    assert fields["fn_abstention_rate"] == 0.125
    assert fields["fn_threshold_tuned"] == 0.3
    assert fields["fn_abstain_thresh_high"] == 0.7
    assert fields["training_completed"] is True
    assert fields["dualgpu_confirmed"] is True
    assert fields["grpo_v5_honest_result"] is True
    assert fields["honest_verdict"] == "tinyv_improves_over_v4"

    assert grr.derive_tinyv_honest_verdict(True, 0.10) == "tinyv_tied_with_v4"
    assert grr.derive_tinyv_honest_verdict(True, 0.08) == "tinyv_degrades_v4"
    assert grr.derive_tinyv_honest_verdict(False, 0.20) == "training_wall_hit"


def test_script_blocked_artifact_uses_required_v5_schema():
    """SCENARIO-LEARN-1175: blocker artifact still reports v5 TinyV schema."""
    exp1173 = _load_script_module()

    artifact = exp1173._build_blocked_artifact(
        started_at="2026-05-02T00:00:00Z",
        cuda_device_count=1,
        sota_path=None,
        thinkprm_v2_auroc=0.9946,
        blocked_reason="torch.cuda.device_count() < 2 in active runtime",
    )

    assert artifact["experiment"] == 1173
    assert artifact["status"] == "blocked"
    assert artifact["dualgpu_confirmed"] is False
    assert artifact["training_completed"] is False
    assert artifact["grpo_v5_honest_result"] is False
    assert artifact["honest_verdict"] == "training_wall_hit"
    assert artifact["fn_abstention_rate"] == 0.0
    assert artifact["fn_threshold_tuned"] == 0.3
    assert artifact["train_slice"] == "[1000, 1200)"
    assert artifact["eval_slice"] == "[1200, 1250)"
    for key in grr.REQUIRED_TINYV_ARTIFACT_FIELDS:
        assert key in artifact


def test_script_blocks_when_llama_runtime_cannot_offload_to_gpu(monkeypatch):
    """SCENARIO-LEARN-1176: CPU-only llama.cpp cannot be dual-GPU confirmed."""
    exp1173 = _load_script_module()

    monkeypatch.setattr(exp1173, "detect_cuda_device_count", lambda: 2)
    monkeypatch.setattr(exp1173, "llama_cpp_supports_gpu_offload", lambda: False)
    monkeypatch.setattr(exp1173.exp1129, "load_thinkprm_v2_auroc", lambda: 0.9946)
    monkeypatch.setattr(exp1173.exp1129, "resolve_sota_path", lambda: "/models/qwen.gguf")

    artifact = exp1173._run_experiment()

    assert artifact["status"] == "blocked"
    assert artifact["dualgpu_confirmed"] is False
    assert artifact["training_completed"] is False
    assert artifact["cuda_device_count"] == 2
    assert artifact["blocked_reason"] == "llama.cpp runtime lacks GPU offload support"
