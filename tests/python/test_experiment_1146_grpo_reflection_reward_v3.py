"""Tests for Exp 1146 repair-grounded GRPO reflection reward.

Spec: REQ-LEARN-1146, SCENARIO-LEARN-1146, SCENARIO-LEARN-1147.

These tests intentionally cover only the new pure-Python reward and artifact
schema logic. The live Qwen3.6-35B-A3B-GGUF path requires two CUDA GPUs and is
validated by the experiment script's blocker artifact when that hardware is not
present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from carnot.training import grpo_reflection_reward as grr


def test_reflection_reward_normalizes_energy_drop_and_clips():
    """REQ-LEARN-1146: r_reflect is normalized energy delta clipped to [-1, 1]."""
    assert math.isclose(grr.normalized_reflection_reward(4.0, 1.0), 0.75)
    assert grr.normalized_reflection_reward(10.0, 25.0) == -1.0
    assert grr.normalized_reflection_reward(10.0, -5.0) == 1.0
    assert grr.normalized_reflection_reward(0.0, 1.0) == 0.0
    assert grr.normalized_reflection_reward(-1.0, -2.0) == 0.0


def test_total_reward_uses_reflection_weight():
    """SCENARIO-LEARN-1146: r_total = r_thinkprm + 0.3 * r_reflect."""
    assert math.isclose(grr.combine_rewards(0.40, 0.75), 0.625)
    assert grr.combine_reward_groups([0.5, 0.2], [0.5, -1.0]) == [0.65, -0.1]


def test_total_reward_group_length_mismatch_raises():
    """REQ-LEARN-1146-3: reward groups must preserve completion alignment."""
    with pytest.raises(ValueError, match="length mismatch"):
        grr.combine_reward_groups([0.1, 0.2], [0.3])


@dataclass
class _Verification:
    verified: bool
    energy: float
    violations: list[str]


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def verify(self, question: str, response: str, domain: str | None = None) -> _Verification:
        self.calls.append((question, response, domain))
        if response == "broken":
            return _Verification(False, 4.0, ["bad arithmetic"])
        if response == "fixed":
            return _Verification(True, 1.0, [])
        return _Verification(True, 2.0, [])

    @staticmethod
    def _format_violations(violations: list[str]) -> str:
        return "; ".join(violations)


def test_reflection_evaluator_runs_one_repair_step():
    """SCENARIO-LEARN-1146: one verifier-guided repair step produces r_reflect."""
    prompts: list[str] = []
    pipeline = _FakePipeline()

    def repair_generate_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "fixed"

    evaluator = grr.ReflectionRewardEvaluator(
        pipeline=pipeline,
        repair_generate_fn=repair_generate_fn,
        domain="math",
    )
    result = evaluator.score("what is 2 + 2?", "broken")

    assert result.repair_attempted is True
    assert result.repaired_response == "fixed"
    assert result.energy_before == 4.0
    assert result.energy_after == 1.0
    assert result.reward == 0.75
    assert result.clipped is False
    assert len(prompts) == 1
    assert "bad arithmetic" in prompts[0]
    assert pipeline.calls == [
        ("what is 2 + 2?", "broken", "math"),
        ("what is 2 + 2?", "fixed", "math"),
    ]


def test_reflection_evaluator_skips_generation_when_verified():
    """REQ-LEARN-1146: verified completions keep reward 0 without extra repair."""
    pipeline = _FakePipeline()

    def repair_generate_fn(_prompt: str) -> str:
        raise AssertionError("repair generator should not run for verified responses")

    evaluator = grr.ReflectionRewardEvaluator(pipeline, repair_generate_fn)
    result = evaluator.score("q", "already correct")

    assert result.repair_attempted is False
    assert result.energy_before == 2.0
    assert result.energy_after == 2.0
    assert result.reward == 0.0
    assert result.repaired_response == "already correct"


def test_reflection_evaluator_zero_before_energy_reward_is_zero():
    """REQ-LEARN-1146: E_before <= 0 returns r_reflect=0.0."""

    class ZeroEnergyPipeline(_FakePipeline):
        def verify(self, question: str, response: str, domain: str | None = None) -> _Verification:
            self.calls.append((question, response, domain))
            if response == "broken":
                return _Verification(False, 0.0, ["metadata-only violation"])
            return _Verification(True, 0.0, [])

    evaluator = grr.ReflectionRewardEvaluator(
        pipeline=ZeroEnergyPipeline(),
        repair_generate_fn=lambda _prompt: "fixed",
    )
    result = evaluator.score("q", "broken")

    assert result.repair_attempted is True
    assert result.energy_before == 0.0
    assert result.energy_after == 0.0
    assert result.reward == 0.0


def test_reflection_honest_verdict_mapping():
    """REQ-LEARN-1146-4: Exp 1146 emits only the canonical verdict labels."""
    assert grr.derive_reflection_honest_verdict(False, 1.0) == "blocked_no_dualgpu"
    assert grr.derive_reflection_honest_verdict(True, 0.09) == "reflection_positive_above_0851"
    assert grr.derive_reflection_honest_verdict(True, 0.02) == "positive_below_exp1129"
    assert grr.derive_reflection_honest_verdict(True, 0.0) == "neutral"
    assert grr.derive_reflection_honest_verdict(True, -0.01) == "negative_regression"


def test_required_artifact_fields_for_blocked_dualgpu():
    """SCENARIO-LEARN-1147: blocked artifact still carries reflection schema."""
    fields = grr.build_reflection_artifact_fields(
        cuda_device_count=0,
        dualgpu_used=False,
        n_training_questions=0,
        training_seconds=0.0,
        training_wall_budget_hit=False,
        advantage_stdev=0.0,
        n_eval_questions=0,
        baseline_fraction_correct=0.0,
        trained_fraction_correct=0.0,
        improvement_over_baseline=0.0,
    )

    assert fields["dualgpu_used"] is False
    assert fields["cuda_device_count"] == 0
    assert fields["honest_verdict"] == "blocked_no_dualgpu"
    assert fields["reflection_weight"] == 0.3
    assert fields["reflection_reward_integrated"] is True
    assert fields["n_repair_steps_per_completion"] == 1
    assert fields["alpha_t_at_training"] == 0.52
    assert fields["fr11_self_learning_signal_used"] is True
    assert fields["grpo_reflection_honest_result"] is True
