"""Tests for Exp 1393 GRPO v8 NGRPO zero-reward fix.

Spec: REQ-LEARN-1393, SCENARIO-LEARN-1393, SCENARIO-LEARN-1394.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from carnot.reporting import grpo_v8_ngrpo_zero_reward_fix as exp


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA = "unsloth/gemma-4-31B-it-GGUF"


def _case(case_id: str, *, label: int = 1) -> exp.FoVerJuryCase:
    return exp.FoVerJuryCase(
        case_id=case_id,
        question="What should the verifier do?",
        response="A short FoVer reasoning step.",
        label=label,
        source="test",
    )


def _cached_specs() -> list[dict[str, object]]:
    return [
        {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": 0, "model_path": "/m/qwen.gguf"},
        {"name": "Gemma4-31B-it", "hf_id": GEMMA, "gpu": 1, "model_path": "/m/gemma.gguf"},
    ]


def test_req_learn_1393_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1393-1: the run starts with an in-progress JSON."""

    out_path = tmp_path / exp.OUTPUT_FILE

    artifact = exp.write_in_progress_artifact(out_path, project_root=tmp_path)

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["grpo_version"] == "v8"
    assert artifact["wall_budget_s"] == 3600
    assert artifact["ngrpo_advantage_calibration_applied"] is False


def test_req_learn_1393_model_specs_use_dual_gpu_tensor_split() -> None:
    """REQ-LEARN-1393-5: specs come from cached_sota_pair on both GPUs."""

    resolution = exp.resolve_model_specs(cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs())

    assert resolution["cached_sota_available"] is True
    assert [spec["gpu"] for spec in resolution["MODEL_SPECS"]] == [0, 1]
    assert all(spec["tensor_split"] == [0.5, 0.5] for spec in resolution["MODEL_SPECS"])
    assert {model["hf_id"] for model in resolution["models_used"]} == {QWEN, GEMMA}
    assert all(model["headline_eligible"] for model in resolution["models_used"])


def test_scenario_learn_1393_ngrpo_injects_virtual_max_reward_sample() -> None:
    """SCENARIO-LEARN-1393: all-zero UNKNOWN groups receive negative advantages."""

    reward = exp.jury_ngrpo_reward_for_case(
        _case("unknown", label=1),
        ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
    )

    assert reward.verifier_result == "UNKNOWN"
    assert reward.raw_rewards == [0.0, 0.0, 0.0, 0.0]
    assert reward.ngrpo_advantage_calibration_applied is True
    assert reward.virtual_max_reward_sample_injected is True
    assert reward.virtual_reward == 1.0
    assert reward.advantages == [-0.2, -0.2, -0.2, -0.2]
    assert reward.virtual_advantage == 0.8
    assert round(sum(reward.advantages) + reward.virtual_advantage, 12) == 0.0


def test_scenario_learn_1394_all_unknown_no_delta_retires(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1394: unchanged all-UNKNOWN outcome retires the rerun."""

    def fake_rollout_generator(
        _case_obj: exp.FoVerJuryCase,
        n_rollouts: int,
        _model_spec: Mapping[str, Any],
        _runtime_settings: Mapping[str, Any],
    ) -> list[str]:
        assert n_rollouts == 4
        return ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"]

    cases = [_case(f"case_{idx}", label=idx % 2) for idx in range(8)]

    artifact = exp.run(
        out_path=tmp_path / exp.OUTPUT_FILE,
        project_root=tmp_path,
        cases=cases,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
        rollout_generator=fake_rollout_generator,
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["ngrpo_advantage_calibration_applied"] is True
    assert artifact["virtual_max_reward_sample_injected"] is True
    assert artifact["training_steps_completed"] == 4
    assert artifact["formal_reward_pass_rate"] == 0.0
    assert artifact["unknown_rollout_rate"] == 1.0
    assert artifact["grpo_v8_improvement_pp"] == 0.0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["terminal_blocker"] == "ngrpo_still_zero_reward_fover_incompatible_with_jury_rl"


def test_req_learn_1393_positive_delta_gates_headline(tmp_path: Path) -> None:
    """REQ-LEARN-1393-5: positive improvement allows headline results."""

    def fake_rollout_generator(
        case: exp.FoVerJuryCase,
        n_rollouts: int,
        _model_spec: Mapping[str, Any],
        _runtime_settings: Mapping[str, Any],
    ) -> list[str]:
        assert n_rollouts == 4
        if case.case_id.startswith("train"):
            return ["REPAIR_HINT", "REPAIR_HINT", "REPAIR_HINT", "SAT"]
        return ["SAT", "REPAIR_HINT", "SAT", "REPAIR_HINT"]

    cases = [
        _case("train_0", label=1),
        _case("train_1", label=1),
        _case("heldout_0", label=1),
        _case("heldout_1", label=1),
    ]

    artifact = exp.run(
        out_path=tmp_path / exp.OUTPUT_FILE,
        project_root=tmp_path,
        cases=cases,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
        rollout_generator=fake_rollout_generator,
        train_case_count=2,
        heldout_case_count=2,
    )

    exp.validate_artifact(artifact)
    assert artifact["formal_reward_pass_rate"] == 1.0
    assert artifact["grpo_v8_improvement_pp"] == 100.0
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "grpo_v8_ngrpo_positive_improvement_100_0pp"
