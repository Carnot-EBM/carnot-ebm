"""Tests for Exp 4341 sc25 E3 explore-verify-plan reproduction.

Spec refs: REQ-PHASE4-080, SCENARIO-PHASE4-080.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4341_e3_sc25_reproduction as exp
from carnot.agentic.arc_executable_world_model import Transition, VerifyResult


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _transition(action: int = 6, data: dict[str, int] | None = None) -> Transition:
    g0 = np.zeros((4, 4), dtype=int)
    g1 = g0.copy()
    g1[1, 2] = 15
    return Transition(g0, action, data or {"x": 29, "y": 49}, g1, 0, 0)


def test_req_phase4_080_spec_declares_exp4341_contract() -> None:
    """REQ-PHASE4-080: OpenSpec declares the sc25 reproduction contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-080" in spec
    assert "SCENARIO-PHASE4-080" in spec
    assert "experiment_4341_e3_sc25_reproduction.json" in spec
    assert "blocked_offline_env_missing_sc25" in spec
    assert "success_e3_sc25_L1_reproduced" in spec
    assert "complete_e3_sc25_partial_model_<acc>" in spec
    assert "(24+5c,49+5r)" in spec
    assert "fireball-animation" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_080_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-080: missing sc25 offline env blocks without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4341)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_sc25"
    assert artifact["verifier_accuracy_per_round"] == []
    assert artifact["win_mechanic_cracked"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["offline_env_present"] is False
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_080_build_artifact_preserves_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-080: solve gates remain bare values while principles are preserved."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[0.5, 0.96],
        world_model_path=model_path,
        plan_result={
            "planned": True,
            "executed": True,
            "level_up": False,
            "solution": list(exp.L1_SOLUTION_LABELS),
        },
        reproduce_result={"game": "sc25", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="cast_pattern_clear_rule_gap",
        adaptive_tests_generated=2,
        explore_lemmas_collected=5,
        win_mechanic_cracked=True,
        random_seed=4341,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "complete_e3_sc25_partial_model_0.96"
    assert artifact["world_model_path"] == exp.WORLD_MODEL_RELATIVE_PATH
    assert artifact["world_model_sha256"] == exp.sha256_file(model_path)
    assert artifact["win_mechanic_cracked"] is True
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["residual_mismatch_class"] == "cast_pattern_clear_rule_gap"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_080_success_artifact_marks_reproduced_l1(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-080: reproduced L1 is the only success verdict."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    artifact = exp.build_artifact(
        repo=tmp_path / "outside",
        verifier_accuracy_per_round=[1.0],
        world_model_path=model_path,
        plan_result={
            "planned": True,
            "executed": True,
            "level_up": True,
            "solution": list(exp.L1_SOLUTION_LABELS),
        },
        reproduce_result={"game": "sc25", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
        adaptive_tests_generated=2,
        explore_lemmas_collected=5,
        win_mechanic_cracked=True,
        random_seed=4341,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success_e3_sc25_L1_reproduced"
    assert artifact["world_model_path"] == str(model_path)
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["win_mechanic_cracked"] is True


def test_req_phase4_080_checksum_binds_win_mechanic_and_plan() -> None:
    """REQ-PHASE4-080: checksum binds model hash, plan, reproduction, and mechanic flag."""

    base = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"solution": ["cell0,1", "move3"]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        win_mechanic_cracked=True,
        random_seed=4341,
    )
    same = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"solution": ["cell0,1", "move3"]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        win_mechanic_cracked=True,
        random_seed=4341,
    )
    changed = exp.compute_reproducibility_checksum(
        world_model_sha256="a" * 64,
        plan_result={"solution": ["cell0,1", "move3"]},
        reproduce_result={"reproduced": True, "reached_level": 1},
        verifier_accuracy_per_round=[1.0],
        win_mechanic_cracked=False,
        random_seed=4341,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_080_schema_errors_are_specific() -> None:
    """REQ-PHASE4-080: artifact schema validation catches non-bare or missing fields."""

    bad = {
        "honest_verdict": "complete_e3_sc25_partial_model_0.00",
        "verifier_accuracy_per_round": "not-list",
        "win_mechanic_cracked": {"value": False},
        "world_model_path": exp.WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": "",
        "offline_reproduced": {"value": False},
        "reproduced_levels": "0",
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4341,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "verifier_accuracy_per_round must be list" in errors
    assert "win_mechanic_cracked must be bare bool" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_req_phase4_080_collect_explore_lemmas_requires_verifier_gated_match() -> None:
    """REQ-PHASE4-080: explore lemmas are counted only after engine reproduction."""

    transitions = [_transition(6), _transition(3, None)]

    def engine(grid: np.ndarray, action: int, _data: dict | None) -> np.ndarray:
        return transitions[0].next_grid if action == 6 else grid

    lemmas = exp.collect_explore_lemmas(transitions, engine)

    assert lemmas == [
        {
            "action": 6,
            "has_data": True,
            "changed_cells": 1,
            "level_delta": 0,
            "verifier_gated": True,
        }
    ]


def test_req_phase4_080_labels_use_corrected_sc25_cast_grid_coords() -> None:
    """REQ-PHASE4-080: sc25 labels encode corrected offline cast-grid coordinates."""

    assert exp.label_to_action_data("warmup") == (5, None)
    assert exp.label_to_action_data("cell0,1") == (6, {"x": 29, "y": 49})
    assert exp.label_to_action_data("cell2,1") == (6, {"x": 29, "y": 59})
    assert exp.label_to_action_data("move3") == (3, None)
    with pytest.raises(ValueError, match="unknown sc25 label"):
        exp.label_to_action_data("bad")


def test_req_phase4_080_residual_gap_and_busy_helpers_cover_rule_classes() -> None:
    """REQ-PHASE4-080: residual mismatch classes identify missing world-model rules."""

    assert exp.residual_mismatch_class([]) == "none"
    assert exp.residual_mismatch_class([{"error": "boom"}]) == "engine_runtime_error_gap"
    assert (
        exp.residual_mismatch_class([{"your_prediction_was_wrong_at": []}])
        == "model_predicted_identity_when_transition_changed_gap"
    )
    assert (
        exp.residual_mismatch_class([{"action": 6, "your_prediction_was_wrong_at": [(1, 2, 3, 4)]}])
        == "missing_world_model_rule_gap_cast_pattern_clear_or_fireball_animation"
    )
    assert (
        exp.residual_mismatch_class([{"action": 3, "your_prediction_was_wrong_at": [(1, 2, 3, 4)]}])
        == "missing_world_model_rule_gap_actions_3"
    )

    class Game:
        eycwbtepcvs = False

    game = Game()
    assert exp._busy(game) is False
    setattr(game, exp.PHASES[0], {"acyylh": True})
    assert exp._busy(game) is True
    setattr(game, exp.PHASES[0], {"acyylh": False})
    game.eycwbtepcvs = True
    assert exp._busy(game) is True


def test_req_phase4_080_collect_explore_lemmas_handles_exceptions_duplicates_and_cap() -> None:
    """REQ-PHASE4-080: lemma collection is verifier-gated, deduplicated, and capped."""

    duplicate_a = _transition(6)
    duplicate_b = _transition(6)
    raise_transition = _transition(2, None)
    mismatch_transition = _transition(3, None)
    level_transition = _transition(4, None)
    level_transition.level_after = 1

    def engine(grid: np.ndarray, action: int, _data: dict | None) -> np.ndarray:
        if action == 2:
            raise RuntimeError("bad transition")
        if action == 3:
            return grid
        return duplicate_a.next_grid

    lemmas = exp.collect_explore_lemmas(
        [raise_transition, duplicate_a, duplicate_b, mismatch_transition, level_transition],
        engine,
        cap=2,
    )

    assert [lemma["action"] for lemma in lemmas] == [6, 4]
    assert lemmas[1]["level_delta"] == 1


def test_req_phase4_080_gap_writer_replaces_prior_entry(tmp_path: Path) -> None:
    """REQ-PHASE4-080: partial model gaps are written once and replace stale entries."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(gap_path, best_accuracy=0.5, mismatch_class="gap_a", checksum="a" * 64)
    first = gap_path.read_text(encoding="utf-8")
    assert "gap_a" in first

    exp._write_gap(gap_path, best_accuracy=0.75, mismatch_class="gap_b", checksum="b" * 64)
    second = gap_path.read_text(encoding="utf-8")
    assert "gap_a" not in second
    assert "gap_b" in second


def test_req_phase4_080_registry_update_records_success_without_touching_absent_blocks(tmp_path: Path) -> None:
    """REQ-PHASE4-080: sc25 registry updates only when a matching block exists."""

    missing = tmp_path / "missing.yaml"
    exp._update_registry_for_success(missing, world_model_sha256="a" * 64, checksum="b" * 64)
    assert not missing.exists()

    no_sc25 = tmp_path / "no_sc25.yaml"
    no_sc25.write_text("games:\n  - game: other\n", encoding="utf-8")
    exp._update_registry_for_success(no_sc25, world_model_sha256="a" * 64, checksum="b" * 64)
    assert "sc25" not in no_sc25.read_text(encoding="utf-8")

    already = tmp_path / "already.yaml"
    already.write_text(
        "reproducible_total_levels: 16\nreproducible_total_games: 13\n"
        "games:\n\n  - game: sc25\n    reproducibility: reproduced\n    levels_reproduced: 1\n",
        encoding="utf-8",
    )
    exp._update_registry_for_success(already, world_model_sha256="a" * 64, checksum="b" * 64)
    assert "world_model_sha256" not in already.read_text(encoding="utf-8")

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "reproducible_total_levels: 15\nreproducible_total_games: 12\n"
        "games:\n\n  - game: sc25\n    reproducibility: provisional\n    levels_reproduced: 0\n"
        "\n  # adapter-free sweep\n  - game: cd82\n    reproducibility: reproduced\n    levels_reproduced: 1\n",
        encoding="utf-8",
    )
    exp._update_registry_for_success(registry, world_model_sha256="a" * 64, checksum="b" * 64)
    text = registry.read_text(encoding="utf-8")
    assert "reproducible_total_levels: 16" in text
    assert "reproducible_total_games: 13" in text
    assert "reproducibility: reproduced" in text
    assert "world_model_sha256" in text


def test_scenario_phase4_080_run_experiment_blocks_when_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-080: runner writes blocked artifact when sc25 env is absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4341, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_sc25"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_080_run_experiment_writes_success_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-080: runner accepts success only after reproduction returns L1."""

    env_dir = tmp_path / "environment_files" / "sc25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "sc25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp, "collect_sc25_transitions", lambda **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: True))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=1, accuracy=1.0, mismatches=[]),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [{"verifier_gated": True}])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [{"passed": True}, {"passed": True}])
    monkeypatch.setattr(
        exp,
        "execute_model_grounded_plan",
        lambda _engine: {
            "planned": True,
            "executed": True,
            "level_up": True,
            "solution": list(exp.L1_SOLUTION_LABELS),
        },
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {"game": "sc25", "reached_level": 1, "claimed_level": 1, "reproduced": True},
    )

    artifact = exp.run_experiment(random_seed=4341, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "success_e3_sc25_L1_reproduced"
    assert artifact["offline_reproduced"] is True


def test_scenario_phase4_080_run_experiment_writes_partial_gap_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-080: unsolved runs record an honest verifier-gap partial."""

    env_dir = tmp_path / "environment_files" / "sc25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "sc25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp, "collect_sc25_transitions", lambda **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1,
            n_correct=0,
            accuracy=0.0,
            mismatches=[{"action": 6, "your_prediction_was_wrong_at": [(1, 2, 3, 4)]}],
        ),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [{"passed": False}])
    monkeypatch.setattr(
        exp,
        "execute_model_grounded_plan",
        lambda _engine: {
            "planned": True,
            "executed": True,
            "level_up": False,
            "solution": list(exp.L1_SOLUTION_LABELS),
        },
    )

    artifact = exp.run_experiment(random_seed=4341, n_transitions=1, round_budget=1)

    assert artifact["honest_verdict"] == "complete_e3_sc25_partial_model_0.00"
    assert artifact["offline_reproduced"] is False
    assert (tmp_path / exp.GAP_RELATIVE_PATH).exists()


def test_scenario_phase4_080_run_experiment_raises_on_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-080: malformed artifacts fail before being accepted."""

    env_dir = tmp_path / "environment_files" / "sc25"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "sc25" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "collect_sc25_transitions", lambda **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=1, accuracy=1.0, mismatches=[]),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [])
    monkeypatch.setattr(exp, "execute_model_grounded_plan", lambda _engine: {"level_up": False})
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["bad"])

    with pytest.raises(ValueError, match="schema errors"):
        exp.run_experiment(random_seed=4341, n_transitions=1, round_budget=1)
