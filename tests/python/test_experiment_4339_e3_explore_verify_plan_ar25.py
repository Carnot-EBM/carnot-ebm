"""Tests for Exp 4339 ar25 E3 explore-verify-plan refinement.

Spec refs: REQ-PHASE4-078, SCENARIO-PHASE4-078.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import carnot.experiment_4339_e3_explore_verify_plan_ar25 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def test_req_phase4_078_spec_declares_exp4339_contract() -> None:
    """REQ-PHASE4-078: OpenSpec declares the ar25 explore-verify-plan contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-078" in spec
    assert "SCENARIO-PHASE4-078" in spec
    assert "experiment_4339_e3_explore_verify_plan_ar25.json" in spec
    assert "blocked_offline_env_missing_ar25" in spec
    assert "success_e3_ar25_L1_reproduced" in spec
    assert "complete_e3_ar25_partial_model_<acc>" in spec
    assert "explore_lemmas_collected" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_078_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-078: missing ar25 offline env blocks without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4339)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ar25"
    assert artifact["verifier_accuracy_per_round"] == []
    assert artifact["explore_lemmas_collected"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["offline_env_present"] is False
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_078_build_artifact_preserves_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-078: solve gates remain bare values while principles are preserved."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    plan = {
        "planned": True,
        "executed": False,
        "divergence_step": {"action": 7},
        "solution": [],
    }
    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[0.8875, 0.95],
        explore_lemmas_collected=12,
        world_model_path=model_path,
        plan_result=plan,
        reproduce_result={"game": "ar25", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="hidden_undo_stack_action7_gap",
        adaptive_tests_generated=4,
        random_seed=4339,
        duration_s=1.5,
    )

    assert artifact["honest_verdict"] == "complete_e3_ar25_partial_model_0.95"
    assert artifact["world_model_path"] == exp.WORLD_MODEL_RELATIVE_PATH
    assert artifact["world_model_sha256"] == exp.sha256_file(model_path)
    assert artifact["explore_lemmas_collected"] == 12
    assert artifact["adaptive_tests_generated"] == 4
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["plan_executed_detail"]["divergence_step"] == {"action": 7}
    assert artifact["residual_mismatch_class"] == "hidden_undo_stack_action7_gap"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_078_success_artifact_marks_reproduced_l1(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-078: reproduced L1 is the only success verdict."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path / "outside",
        verifier_accuracy_per_round=[0.8875, 0.9625],
        explore_lemmas_collected=8,
        world_model_path=model_path,
        plan_result={"planned": True, "executed": True, "level_up": True, "solution": ["1", "5"]},
        reproduce_result={"game": "ar25", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
        adaptive_tests_generated=3,
        random_seed=4339,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success_e3_ar25_L1_reproduced"
    assert artifact["world_model_path"] == str(model_path)
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert artifact["plan_executed_detail"]["divergence_step"] is None


def test_req_phase4_078_checksum_binds_explore_count_and_plan() -> None:
    """REQ-PHASE4-078: checksum binds model hash, plan, reproduction, and lemmas."""

    base = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.95],
        explore_lemmas_collected=7,
        random_seed=4339,
    )
    same = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.95],
        explore_lemmas_collected=7,
        random_seed=4339,
    )
    changed = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.95],
        explore_lemmas_collected=8,
        random_seed=4339,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_078_schema_errors_are_specific() -> None:
    """REQ-PHASE4-078: artifact schema validation catches non-bare or missing fields."""

    bad = {
        "honest_verdict": "complete_e3_ar25_partial_model_0.00",
        "verifier_accuracy_per_round": "not-list",
        "explore_lemmas_collected": "0",
        "world_model_path": exp.WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": "",
        "offline_reproduced": {"value": False},
        "reproduced_levels": "0",
        "plan_executed": None,
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4339,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "verifier_accuracy_per_round must be list" in errors
    assert "explore_lemmas_collected must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "plan_executed must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_078_run_experiment_blocks_when_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-078: runner writes blocked artifact when ar25 env is absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4339, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ar25"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()

