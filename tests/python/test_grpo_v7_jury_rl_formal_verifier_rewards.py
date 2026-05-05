"""Tests for Exp 1383 GRPO v7 JURY-RL formal verifier rewards.

Spec: REQ-LEARN-1383, SCENARIO-LEARN-1383, SCENARIO-LEARN-1384.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from carnot.reporting import grpo_v7_jury_rl_formal_verifier_rewards as exp


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


def test_req_learn_1383_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1383-1: the run starts with an auditable in-progress JSON."""

    out_path = tmp_path / exp.OUTPUT_FILE

    artifact = exp.write_in_progress_artifact(out_path, project_root=tmp_path)

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["grpo_version"] == "v7"
    assert artifact["wall_budget_s"] == 2400
    assert artifact["honest_verdict"] == "in_progress"


def test_req_learn_1383_model_specs_use_dual_gpu_tensor_split() -> None:
    """REQ-LEARN-1383-4/5: headline specs come from cached_sota_pair on two GPUs."""

    resolution = exp.resolve_model_specs(cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs())

    assert resolution["cached_sota_available"] is True
    assert [spec["gpu"] for spec in resolution["MODEL_SPECS"]] == [0, 1]
    assert all(spec["tensor_split"] == [0.5, 0.5] for spec in resolution["MODEL_SPECS"])
    assert {model["hf_id"] for model in resolution["models_used"]} == {QWEN, GEMMA}
    assert all(model["headline_eligible"] for model in resolution["models_used"])


def test_req_learn_1383_reward_math_for_verified_unknown_and_rejected() -> None:
    """REQ-LEARN-1383-2/3: rewards follow proof outcome and ResZero rules."""

    verified = exp.jury_reward_for_case(
        _case("verified", label=1),
        ["REPAIR_HINT", "SAT", "REPAIR_HINT", "UNKNOWN"],
    )
    assert verified.candidate_answer == "REPAIR_HINT"
    assert verified.verifier_result == "VERIFIED"
    assert verified.rewards == [1.0, -1.0, 1.0, -1.0]
    assert verified.reszero_applied is False

    unknown = exp.jury_reward_for_case(
        _case("unknown", label=1),
        ["UNKNOWN", "UNKNOWN", "SAT", "REPAIR_HINT"],
    )
    assert unknown.candidate_answer == "UNKNOWN"
    assert unknown.verifier_result == "UNKNOWN"
    assert unknown.reszero_applied is True
    assert round(sum(unknown.rewards), 12) == 0.0

    rejected = exp.jury_reward_for_case(
        _case("rejected", label=0),
        ["REPAIR_HINT", "REPAIR_HINT", "SAT", "UNKNOWN"],
    )
    assert rejected.candidate_answer == "REPAIR_HINT"
    assert rejected.verifier_result == "REJECTED"
    assert rejected.rewards == [-1.0, -1.0, -1.0, -1.0]


def test_scenario_learn_1384_run_writes_positive_headline_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1384: a positive held-out delta gates headline claims."""

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
    out_path = tmp_path / exp.OUTPUT_FILE

    artifact = exp.run(
        out_path=out_path,
        project_root=tmp_path,
        cases=cases,
        cached_pair_fn=lambda gpu_indices=(0, 1): _cached_specs(),
        rollout_generator=fake_rollout_generator,
        train_case_count=2,
        heldout_case_count=2,
        wall_budget_s=2400,
    )

    exp.validate_artifact(artifact)
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["training_steps_completed"] == 2
    assert artifact["jury_acceptance_rate"] == 1.0
    assert artifact["formal_reward_pass_rate"] == 1.0
    assert artifact["resZero_applied_count"] == 0
    assert artifact["grpo_v7_improvement_pp"] == 100.0
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == "grpo_v7_jury_rl_positive_improvement_100_0pp"


def test_req_learn_1383_wall_budget_terminal_artifact_sets_retirement() -> None:
    """REQ-LEARN-1383-6: wall budget exhaustion is terminal and retired."""

    artifact = exp.wall_budget_terminal_artifact(
        base_artifact=exp.base_artifact(project_root="/tmp/carnot", status="complete"),
        models_used=[],
        wall_time_used_s=2401.0,
        training_steps_completed=3,
        grpo_v7_improvement_pp=0.0,
    )

    exp.validate_artifact(artifact)
    assert artifact["wall_budget_exhausted"] is True
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["terminal_blocker"] == "wall_budget_exhausted"
