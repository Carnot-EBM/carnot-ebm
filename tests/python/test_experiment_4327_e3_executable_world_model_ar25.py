"""Tests for Exp 4327 ar25 E3 executable-world-model attempt.

Spec refs: REQ-PHASE4-074, SCENARIO-PHASE4-074.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4327_e3_executable_world_model_ar25 as exp
from carnot.agentic.arc_executable_world_model import Transition, VerifyResult


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _transition() -> Transition:
    g0 = np.array([[0, 1], [2, 3]], dtype=int)
    g1 = np.array([[0, 1], [2, 4]], dtype=int)
    return Transition(g0, 5, None, g1, 0, 0)


def test_req_phase4_074_spec_declares_exp4327_contract() -> None:
    """REQ-PHASE4-074: OpenSpec declares the ar25 E3 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-074" in spec
    assert "SCENARIO-PHASE4-074" in spec
    assert "experiment_4327_e3_executable_world_model_ar25.json" in spec
    assert "blocked_offline_env_missing_ar25" in spec
    assert "success_e3_ar25_L1_reproduced" in spec
    assert "complete_e3_ar25_partial_model_<acc>" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_074_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-074: missing ar25 offline env blocks without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4327)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ar25"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["offline_env_present"] is False
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_074_build_artifact_preserves_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-074: solve gates remain bare values while principles are preserved."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    reproduce_result = {
        "game": "ar25",
        "reached_level": 0,
        "claimed_level": 1,
        "reproduced": False,
    }
    plan = {"planned": True, "executed": False, "divergence_step": {"action": 1}}
    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[0.61, 0.75],
        world_model_path=model_path,
        plan_result=plan,
        reproduce_result=reproduce_result,
        residual_mismatch_class="translation_sign_rule_gap",
        random_seed=4327,
        duration_s=1.5,
    )

    assert artifact["honest_verdict"] == "complete_e3_ar25_partial_model_0.75"
    assert artifact["world_model_path"] == exp.WORLD_MODEL_RELATIVE_PATH
    assert artifact["world_model_sha256"] == exp.sha256_file(model_path)
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["plan_executed_detail"]["divergence_step"] == {"action": 1}
    assert artifact["residual_mismatch_class"] == "translation_sign_rule_gap"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_074_success_artifact_marks_reproduced_l1(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-074: reproduced L1 is the only success verdict."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path / "outside",
        verifier_accuracy_per_round=[0.95],
        world_model_path=model_path,
        plan_result={"planned": True, "executed": True, "level_up": True},
        reproduce_result={"game": "ar25", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
        random_seed=4327,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success_e3_ar25_L1_reproduced"
    assert artifact["world_model_path"] == str(model_path)
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert artifact["plan_executed_detail"]["divergence_step"] is None


def test_scenario_phase4_074_no_plan_artifact_keeps_execution_false(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-074: a no-plan partial does not fabricate execution."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[],
        world_model_path=model_path,
        plan_result=None,
        reproduce_result={"game": "ar25", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="no_goal_predicate_gap",
        random_seed=4327,
        duration_s=0.0,
    )

    assert artifact["plan_executed"] is False
    assert artifact["plan_executed_detail"]["divergence_step"] is None
    assert artifact["verifier_best_accuracy"] == 0.0


def test_req_phase4_074_checksum_is_stable_and_sensitive() -> None:
    """REQ-PHASE4-074: checksum binds the world model hash, plan, and reproduction."""

    base = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        random_seed=4327,
    )
    same = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        random_seed=4327,
    )
    changed = exp.compute_reproducibility_checksum(
        world_model_sha256="b" * 64,
        plan_result={"planned": True, "actions": [1, 2]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        random_seed=4327,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_074_residual_mismatch_classes_cover_known_gaps() -> None:
    """REQ-PHASE4-074: residual mismatch class records the partial-model gap."""

    assert exp.residual_mismatch_class([]) == "none"
    assert exp.residual_mismatch_class([{"error": "boom"}]) == "engine_runtime_error_gap"
    assert (
        exp.residual_mismatch_class([{"your_prediction_was_wrong_at": []}])
        == "model_predicted_identity_when_transition_changed_gap"
    )
    assert (
        exp.residual_mismatch_class([{"your_prediction_was_wrong_at": "wrong shape (1, 1)"}])
        == "world_model_shape_rule_gap"
    )
    assert (
        exp.residual_mismatch_class([{"action": 4}, {"action": 2}])
        == "missing_world_model_rule_gap_actions_2_4"
    )
    assert (
        exp.residual_mismatch_class([{"action": 7}, {"action": 1}])
        == "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )


def test_req_phase4_074_schema_errors_are_specific() -> None:
    """REQ-PHASE4-074: artifact schema validation catches non-bare or missing fields."""

    bad = {
        "honest_verdict": "complete_e3_ar25_partial_model_0.00",
        "verifier_accuracy_per_round": "not-list",
        "world_model_path": exp.WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": "",
        "offline_reproduced": {"value": False},
        "reproduced_levels": "0",
        "plan_executed": None,
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4327,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "verifier_accuracy_per_round must be list" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "plan_executed must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_074_write_gap_replaces_prior_entry(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-074: partial runs keep a single auditable residual-gap entry."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(gap_path, best_accuracy=0.1, mismatch_class="first", checksum="a" * 64)
    exp._write_gap(gap_path, best_accuracy=0.2, mismatch_class="second", checksum="b" * 64)

    text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.2000" in text
    assert "second" in text
    assert "first" not in text


def test_scenario_phase4_074_apply_noop_returns_frame() -> None:
    """SCENARIO-PHASE4-074: reproduction fallback apply is explicit and deterministic."""

    frame = object()

    assert exp._apply_noop(None, "noop", frame) is frame


def test_scenario_phase4_074_run_experiment_blocks_when_env_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-074: runner writes blocked artifact when ar25 env is absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4327, n_transitions=1)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ar25"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_074_run_experiment_writes_partial_with_stubs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-074: bounded run writes an honest partial when L1 is not reproduced."""

    env_dir = tmp_path / "environment_files" / "ar25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ar25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "ENV_DIR", tmp_path / "environment_files")
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1,
            n_correct=0,
            accuracy=0.0,
            mismatches=[{"action": 5, "true_change": [(1, 1, 3, 4)]}],
        ),
    )
    monkeypatch.setattr(
        exp.e3,
        "plan_and_execute",
        lambda *_args, **_kwargs: {"planned": False, "reason": "no plan to is_level_complete in model"},
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {"game": "ar25", "reached_level": 0, "claimed_level": 1, "reproduced": False},
    )

    artifact = exp.run_experiment(random_seed=4327, n_transitions=1)

    assert artifact["honest_verdict"] == "complete_e3_ar25_partial_model_0.00"
    assert artifact["verifier_accuracy_per_round"] == [0.0]
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["preconditions_checked"]["harness_import"] is True
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()
    assert "REQ-PHASE4-074" in (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_074_run_experiment_writes_success_with_stubs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-074: runner accepts success only after reproduction returns L1."""

    env_dir = tmp_path / "environment_files" / "ar25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ar25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: True))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1,
            n_correct=1,
            accuracy=1.0,
            mismatches=[],
        ),
    )
    monkeypatch.setattr(
        exp.e3,
        "plan_and_execute",
        lambda *_args, **_kwargs: {
            "planned": True,
            "executed": True,
            "level_up": True,
            "solution": ["noop"],
        },
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {"game": "ar25", "reached_level": 1, "claimed_level": 1, "reproduced": True},
    )

    artifact = exp.run_experiment(random_seed=4327, n_transitions=1)

    assert artifact["honest_verdict"] == "success_e3_ar25_L1_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert not (tmp_path / exp.GAP_RELATIVE_PATH).exists()


def test_scenario_phase4_074_run_experiment_raises_on_schema_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-074: runner fails loudly if the artifact contract regresses."""

    env_dir = tmp_path / "environment_files" / "ar25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ar25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1,
            n_correct=0,
            accuracy=0.0,
            mismatches=[],
        ),
    )
    monkeypatch.setattr(exp.e3, "plan_and_execute", lambda *_args, **_kwargs: {"planned": False})
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4327 artifact schema errors"):
        exp.run_experiment(random_seed=4327, n_transitions=1)
