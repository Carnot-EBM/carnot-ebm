"""Tests for Exp 4340 ka59 E3 explore-verify-plan refinement.

Spec refs: REQ-PHASE4-079, SCENARIO-PHASE4-079.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4340_e3_explore_verify_plan_ka59 as exp
from carnot.agentic.arc_executable_world_model import Transition, VerifyResult


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _transition(action: int = 4) -> Transition:
    g0 = np.array([[14, 0, 14], [14, 14, 14]], dtype=int)
    g1 = np.array([[14, 4, 14], [14, 14, 14]], dtype=int)
    return Transition(g0, action, None, g1, 0, 0)


def test_req_phase4_079_spec_declares_exp4340_contract() -> None:
    """REQ-PHASE4-079: OpenSpec declares the ka59 explore-verify-plan contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-079" in spec
    assert "SCENARIO-PHASE4-079" in spec
    assert "experiment_4340_e3_explore_verify_plan_ka59.json" in spec
    assert "blocked_offline_env_missing_ka59" in spec
    assert "success_e3_ka59_L1_reproduced" in spec
    assert "complete_e3_ka59_partial_model_<acc>" in spec
    assert "explore_lemmas_collected" in spec
    assert "hidden step-counter HUD" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_079_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-079: missing ka59 offline env blocks without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4340)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ka59"
    assert artifact["verifier_accuracy_per_round"] == []
    assert artifact["explore_lemmas_collected"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["offline_env_present"] is False
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_079_build_artifact_preserves_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-079: solve gates remain bare values while principles are preserved."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[0.5625, 0.75],
        explore_lemmas_collected=6,
        world_model_path=model_path,
        plan_result={"planned": True, "executed": False, "divergence_step": {"action": 2}},
        reproduce_result={"game": "ka59", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="hidden_step_counter_hud_gap",
        adaptive_tests_generated=4,
        random_seed=4340,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "complete_e3_ka59_partial_model_0.75"
    assert artifact["world_model_path"] == exp.WORLD_MODEL_RELATIVE_PATH
    assert artifact["world_model_sha256"] == exp.sha256_file(model_path)
    assert artifact["explore_lemmas_collected"] == 6
    assert artifact["adaptive_tests_generated"] == 4
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["plan_executed_detail"]["divergence_step"] == {"action": 2}
    assert artifact["residual_mismatch_class"] == "hidden_step_counter_hud_gap"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_079_success_artifact_marks_reproduced_l1(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-079: reproduced L1 is the only success verdict."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path / "outside",
        verifier_accuracy_per_round=[0.8125],
        explore_lemmas_collected=8,
        world_model_path=model_path,
        plan_result={"planned": True, "executed": True, "level_up": True, "solution": list(exp.L1_SOLUTION_LABELS)},
        reproduce_result={"game": "ka59", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
        adaptive_tests_generated=4,
        random_seed=4340,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success_e3_ka59_L1_reproduced"
    assert artifact["world_model_path"] == str(model_path)
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert artifact["plan_executed_detail"]["divergence_step"] is None


def test_req_phase4_079_checksum_binds_explore_count_and_plan() -> None:
    """REQ-PHASE4-079: checksum binds model hash, plan, reproduction, and lemmas."""

    base = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.75],
        explore_lemmas_collected=7,
        random_seed=4340,
    )
    same = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.75],
        explore_lemmas_collected=7,
        random_seed=4340,
    )
    changed = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[0.75],
        explore_lemmas_collected=8,
        random_seed=4340,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_079_schema_errors_are_specific() -> None:
    """REQ-PHASE4-079: artifact schema validation catches non-bare or missing fields."""

    bad = {
        "honest_verdict": "complete_e3_ka59_partial_model_0.00",
        "verifier_accuracy_per_round": "not-list",
        "explore_lemmas_collected": "0",
        "world_model_path": exp.WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": "",
        "offline_reproduced": {"value": False},
        "reproduced_levels": "0",
        "plan_executed": None,
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4340,
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


def test_req_phase4_079_collect_explore_lemmas_requires_verifier_gated_match() -> None:
    """REQ-PHASE4-079: explore lemmas are counted only after engine reproduction."""

    transitions = [_transition(4), _transition(3)]

    def engine(grid: np.ndarray, action: int, _data: dict | None) -> np.ndarray:
        return transitions[0].next_grid if action == 4 else grid

    lemmas = exp.collect_explore_lemmas(transitions, engine)

    assert lemmas == [
        {
            "action": 4,
            "has_data": False,
            "changed_cells": 1,
            "level_delta": 0,
            "verifier_gated": True,
        }
    ]


def test_scenario_phase4_079_write_gap_replaces_prior_entry(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-079: partial runs keep a single auditable residual-gap entry."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(gap_path, best_accuracy=0.1, mismatch_class="first", checksum="a" * 64)
    exp._write_gap(gap_path, best_accuracy=0.2, mismatch_class="second", checksum="b" * 64)

    text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.2000" in text
    assert "second" in text
    assert "first" not in text


def test_scenario_phase4_079_run_experiment_blocks_when_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-079: runner writes blocked artifact when ka59 env is absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4340, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ka59"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_079_run_experiment_writes_success_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-079: runner accepts success only after reproduction returns L1."""

    env_dir = tmp_path / "environment_files" / "ka59"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ka59" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: True))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=1, accuracy=1.0, mismatches=[]),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [{"verifier_gated": True}])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [{"passed": True}])
    monkeypatch.setattr(
        exp,
        "execute_model_grounded_plan",
        lambda _engine: {"planned": True, "executed": True, "level_up": True, "solution": list(exp.L1_SOLUTION_LABELS)},
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {"game": "ka59", "reached_level": 1, "claimed_level": 1, "reproduced": True},
    )

    artifact = exp.run_experiment(random_seed=4340, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "success_e3_ka59_L1_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert not (tmp_path / exp.GAP_RELATIVE_PATH).exists()


def test_scenario_phase4_079_run_experiment_raises_on_schema_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-079: runner fails loudly if the artifact contract regresses."""

    env_dir = tmp_path / "environment_files" / "ka59"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ka59" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=0, accuracy=0.0, mismatches=[]),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [])
    monkeypatch.setattr(exp, "execute_model_grounded_plan", lambda _engine: {"planned": False})
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4340 artifact schema errors"):
        exp.run_experiment(random_seed=4340, n_transitions=1, round_budget=1)
