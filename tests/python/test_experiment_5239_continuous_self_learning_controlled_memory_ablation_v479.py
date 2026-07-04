"""Tests for Exp 5239 controlled typed-memory consumer ablation.

Spec refs: REQ-LEARN-5239, SCENARIO-LEARN-5239.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline import controlled_memory_ablation as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5239_spec_declares_controlled_five_arm_contract() -> None:
    """REQ-LEARN-5239: OpenSpec declares the five-arm controlled-memory contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5239") :]

    for marker in (
        "REQ-LEARN-5239",
        "SCENARIO-LEARN-5239",
        "no_memory",
        "best_constant",
        "per_query_random",
        "shuffled_memory",
        "aligned_memory",
        "controlled_nonparametric_typed_memory_ablation",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in section


def test_req_learn_5239_stream_uses_existing_heads_and_rollback_states() -> None:
    """REQ-LEARN-5239-1/5: the stream covers all typed heads and rollback entries."""

    memory = mod.load_memory(REPO)
    stream = mod.build_controlled_stream(memory)

    assert len(stream) == 6
    assert {task.expected_head for task in stream} == set(mod.MEMORY_HEADS)
    assert {task.expected_state for task in stream} == {"promoted", "rolled_back"}
    assert [task.expected_subject for task in stream] == list(mod.EXPECTED_ACTIONS)
    assert all(task.spec_refs == mod.SPEC_REFS for task in stream)


def test_scenario_learn_5239_five_arms_deconfound_aligned_memory() -> None:
    """SCENARIO-LEARN-5239: aligned memory beats shuffled and no-memory controls."""

    artifact = mod.build_result_artifact(
        memory=mod.load_memory(REPO),
        tests_run=[{"command": "fixture", "passed": True}],
    )
    metrics = artifact["arm_metrics"]

    assert artifact["arms"]["value"] == list(mod.ARM_NAMES)
    assert metrics["aligned_memory"]["accuracy"] == 1.0
    assert metrics["aligned_memory"]["accuracy"] > metrics["shuffled_memory"]["accuracy"]
    assert metrics["aligned_memory"]["accuracy"] > metrics["no_memory"]["accuracy"]
    assert artifact["aligned_vs_shuffled_delta"]["value"] > 0.0
    assert artifact["aligned_vs_no_memory_delta"]["value"] > 0.0
    assert artifact["degradation_detected"]["value"] is True
    assert artifact["retention_check_passed"]["value"] is True
    assert artifact["rollback_policy_exercised"]["value"] is True
    assert artifact["recommended_arc_memory_heads"]["value"] == [
        "provenance",
        "failures",
        "skills_rubrics",
    ]
    assert artifact["inference_substrate"]["value"] == (
        "controlled_nonparametric_typed_memory_ablation"
    )
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "controlled useful reuse" in artifact["honest_verdict"]["value"]


def test_req_learn_5239_constant_random_and_shuffled_controls_are_seeded() -> None:
    """REQ-LEARN-5239-2/3: control arms are deterministic and budget matched."""

    first = mod.evaluate_memory(mod.load_memory(REPO), seed=mod.RANDOM_SEED)
    second = mod.evaluate_memory(mod.load_memory(REPO), seed=mod.RANDOM_SEED)

    assert first == second
    assert first["budget"] == {
        "task_budget_per_arm": 6,
        "prior_slots_per_query": 1,
        "random_seed": mod.RANDOM_SEED,
    }
    assert first["arm_metrics"]["best_constant"]["selected_constant_subject"] in set(
        mod.EXPECTED_ACTIONS
    )
    assert first["arm_metrics"]["shuffled_memory"]["fixed_points"] == 0
    assert first["arm_metrics"]["per_query_random"]["selected_subjects"] == [
        row["selected_subject"] for row in first["arm_metrics"]["per_query_random"]["rows"]
    ]
    assert first["arm_metrics"]["no_memory"]["selected_subjects"] == [None] * 6


def test_req_learn_5239_result_schema_wraps_required_fields(tmp_path: Path) -> None:
    """REQ-LEARN-5239-4/6: run() writes the required principle-wrapped fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "passed": True}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]

    assert artifact["continuous_self_learning_task"]["value"] is True
    assert artifact["memory_heads_tested"]["value"] == list(mod.MEMORY_HEADS)
    assert artifact["controlled_stream_n"]["value"] == 6
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["nonparametric_memory_updates"]["value"] is True
    assert artifact["broad_self_distillation_used"]["value"] is False


def test_req_learn_5239_repository_artifact_is_stable_and_replayable() -> None:
    """REQ-LEARN-5239: checked-in artifact matches deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(memory=mod.load_memory(REPO), tests_run=[])

    assert result == replay
    assert result["schema"] == mod.SCHEMA
    assert result["experiment_id"] == 5239
    assert result["continuous_self_learning_task"]["value"] is True
    assert set(result["arms"]["value"]) == set(mod.ARM_NAMES)
    assert result["aligned_vs_shuffled_delta"]["value"] == replay[
        "aligned_vs_shuffled_delta"
    ]["value"]
    assert result["aligned_vs_no_memory_delta"]["value"] == replay[
        "aligned_vs_no_memory_delta"
    ]["value"]
