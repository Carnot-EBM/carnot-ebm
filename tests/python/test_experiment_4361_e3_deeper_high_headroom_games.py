"""Tests for Exp 4361 E3 deeper high-headroom checkpoint run.

Spec refs: REQ-PHASE4-085, SCENARIO-PHASE4-085.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import carnot.experiment_4361_e3_deeper_high_headroom_games as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _row(
    game: str,
    *,
    prior: int | None = None,
    reached: int | None = None,
    accuracy: float = 0.5,
    advanced: bool = False,
) -> dict:
    prior_level = exp.PRIOR_BEST_LEVELS[game] if prior is None else prior
    reached_level = prior_level if reached is None else reached
    return {
        "game": game,
        "prior_best_level": prior_level,
        "new_reproduced_level": reached_level,
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "offline_reproduced": advanced,
        "reproduce_result": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": advanced,
        },
        "plan": ["mock"],
        "residual_win_mechanic_gap_class": "none" if advanced else "bounded_deepen_no_new_level",
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial",
        "world_model_path": exp.WORLD_MODEL_PATHS[game],
    }


def test_req_phase4_085_spec_declares_exp4361_contract() -> None:
    """REQ-PHASE4-085: OpenSpec declares the high-headroom deepen contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-085" in spec
    assert "SCENARIO-PHASE4-085" in spec
    assert "experiment_4361_e3_deeper_high_headroom_games.json" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_deeper_<targets>_reproduced" in spec
    assert "complete_e3_deeper_partial" in spec
    assert "sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap" in spec
    assert "program-editor solver toward L8" in spec
    assert "frame-based fresh_env branch-mode solver for L4" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_085_checksum_binds_rows_paths_and_seed() -> None:
    """REQ-PHASE4-085: checksum binds scorecard, paths, reproduction, and seed."""

    rows = [
        _row("sc25"),
        _row("tn36"),
        _row("lp85"),
        _row("tu93", reached=4, accuracy=1.0, advanced=True),
    ]
    hashes = {"a.py": "a" * 64}
    base = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4361,
    )
    same = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4361,
    )
    changed = exp.compute_reproducibility_checksum(
        per_target_scorecard=[{**rows[3], "new_reproduced_level": 3}],
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4361,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_085_build_artifact_counts_only_new_reproduced_levels(tmp_path: Path) -> None:
    """REQ-PHASE4-085: only levels beyond the prior best count as new progress."""

    adapter = tmp_path / "python" / "carnot" / "agentic" / "arc_game_adapters.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# adapters\n", encoding="utf-8")
    solver = tmp_path / "scripts" / "arc3_tn36_offline_solver.py"
    solver.parent.mkdir(parents=True)
    solver.write_text("# solver\n", encoding="utf-8")

    rows = [
        _row("sc25", accuracy=1.0),
        _row("tn36", reached=7, accuracy=0.875),
        _row("lp85", reached=4, accuracy=0.8),
        _row("tu93", reached=4, accuracy=1.0, advanced=True),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=33,
        world_model_paths=[str(adapter.relative_to(tmp_path)), str(solver.relative_to(tmp_path))],
        random_seed=4361,
        duration_s=2.5,
    )

    assert artifact["honest_verdict"] == "success_e3_deeper_tu93_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 33
    assert artifact["verifier_is_oracle"] is True
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_085_partial_artifact_preserves_all_targets(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-085: all-partial runs keep one row per target and bare gates."""

    rows = [
        _row("sc25", accuracy=0.75),
        _row("tn36", reached=7, accuracy=0.875),
        _row("lp85", reached=4, accuracy=0.6),
        _row("tu93", reached=3, accuracy=0.4),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=32,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4361,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["verifier_is_oracle"] is True
    assert isinstance(artifact["reproducible_total_levels"], int)
    assert isinstance(artifact["new_levels_reproduced"], int)
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_085_schema_errors_are_specific() -> None:
    """REQ-PHASE4-085: schema validation catches wrapped or malformed gate fields."""

    bad = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": "not-list",
        "reproducible_total_levels": {"value": 33},
        "new_levels_reproduced": "1",
        "world_model_paths": ["a.py"],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4361,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard must be list" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_req_phase4_085_schema_validation_covers_row_shape_errors() -> None:
    """REQ-PHASE4-085: malformed scorecard rows produce specific schema errors."""

    artifact = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": ["bad-row", {"game": "tn36", "offline_reproduced": "yes"}],
        "reproducible_total_levels": 32,
        "new_levels_reproduced": 0,
        "world_model_paths": [123],
        "verifier_is_oracle": "true",
        "preconditions_checked": {},
        "random_seed": 4361,
        "reproducibility_checksum": "a" * 64,
        "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_target_scorecard[0] must be dict" in errors
    assert "per_target_scorecard[1] missing prior_best_level" in errors
    assert "per_target_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be bare bool" in errors


def test_scenario_phase4_085_run_experiment_records_missing_envs_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-085: missing target envs block per target without fabrication."""

    env = tmp_path / "environment_files" / "tu93"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    adapter = tmp_path / exp.WORLD_MODEL_PATHS["tu93"]
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# adapters\n", encoding="utf-8")

    calls: list[str] = []

    def fake_tu93_runner(_repo: Path, _random_seed: int) -> dict:
        calls.append("tu93")
        return _row("tu93", reached=4, accuracy=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "tu93", fake_tu93_runner)

    artifact = exp.run_experiment(random_seed=4361)

    assert calls == ["tu93"]
    assert artifact["honest_verdict"] == "success_e3_deeper_tu93_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_target_scorecard"]] == [
        "blocked_offline_env_missing_sc25",
        "blocked_offline_env_missing_tn36",
        "blocked_offline_env_missing_lp85",
        "new_level_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_085_prior_artifact_row_covers_present_and_missing_inputs(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-085: existing sc25 L1 artifacts are partials unless they advance deeper."""

    sc25_result = tmp_path / "results" / "experiment_4341_e3_sc25_reproduction.json"
    sc25_result.parent.mkdir(parents=True)
    sc25_result.write_text(
        json.dumps(
            {
                "verifier_accuracy_per_round": [0.5, 1.0],
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "accepted_plan": ["cell0,1"],
            }
        ),
        encoding="utf-8",
    )

    present = exp._run_sc25_target(tmp_path, 4361)
    missing = exp._prior_artifact_row(
        repo=tmp_path,
        game="lp85",
        result_relative_path="results/missing.json",
        residual_gap="lp85_l5_search_path_not_offline_reproduced_reset_replay_gap",
    )

    assert present["verifier_accuracy"] == 1.0
    assert present["offline_reproduced"] is False
    assert present["plan"] == ["cell0,1"]
    assert present["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert missing["verifier_accuracy"] == 0.0
    assert missing["checkpoint_status"] == "honest_partial_prior_artifact_missing"


def test_req_phase4_085_tn36_runner_counts_only_new_reproduced_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: tn36 L8 counts only after the reproduction gate returns true."""

    class FakeSolver:
        @staticmethod
        def solve(max_level: int, cap: int):
            assert (max_level, cap) == (8, 500)
            return ([{"action": 6, "data": {"x": 1, "y": 2}}], 8)

    class FakeEnv:
        def step(self, action, data=None):
            return {"action": action, "data": data}

    def fake_reproduce(game, labels, apply, claimed_level):
        frame = apply(FakeEnv(), labels[0], None)
        assert frame["data"] == {"x": 1, "y": 2}
        return {
            "game": game,
            "reached_level": claimed_level,
            "claimed_level": claimed_level,
            "reproduced": True,
        }

    monkeypatch.setattr(exp, "_load_tn36_solver", lambda _repo: FakeSolver)
    monkeypatch.setattr(exp.arc_solver_kit, "reproduce", fake_reproduce)

    row = exp._run_tn36_target(tmp_path, 4361)

    assert row["game"] == "tn36"
    assert row["prior_best_level"] == 7
    assert row["new_reproduced_level"] == 8
    assert row["offline_reproduced"] is True
    assert row["trajectory_action_count"] == 1


def test_req_phase4_085_tn36_runner_keeps_l7_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: reproducing only tn36 L7 does not count as new L8 progress."""

    class FakeSolver:
        @staticmethod
        def solve(max_level: int, cap: int):
            return ([], 7)

    monkeypatch.setattr(exp, "_load_tn36_solver", lambda _repo: FakeSolver)
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {
            "game": "tn36",
            "reached_level": 7,
            "claimed_level": 7,
            "reproduced": True,
        },
    )

    row = exp._run_tn36_target(tmp_path, 4361)

    assert row["new_reproduced_level"] == 7
    assert row["offline_reproduced"] is False
    assert row["verifier_accuracy"] == 0.875
    assert row["residual_win_mechanic_gap_class"] == "tn36_l8_program_editor_maze_delta_gap"


def test_req_phase4_085_adaptered_runner_accepts_only_reproduction_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: lp85 L5 search hits are partials if reset replay rejects them."""

    def fake_solve_adaptered(game: str, target_level: int) -> dict:
        assert (game, target_level) == ("lp85", 5)
        return {
            "game": "lp85",
            "target": 5,
            "reached_level": 5,
            "moves": 66,
            "states_expanded": 962,
            "offline_reproduced": False,
            "solution_labels": ["a0"],
            "verifier_src": "learned_checkpoint",
            "reproduction_gate": {
                "game": "lp85",
                "reached_level": 4,
                "claimed_level": 5,
                "reproduced": False,
            },
        }

    monkeypatch.setattr(exp, "_solve_adaptered", fake_solve_adaptered)

    row = exp._run_lp85_target(Path("."), 4361)

    assert row["new_reproduced_level"] == 4
    assert row["searched_level"] == 5
    assert row["offline_reproduced"] is False
    assert row["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert row["residual_win_mechanic_gap_class"] == (
        "lp85_l5_search_path_not_offline_reproduced_reset_replay_gap"
    )


def test_req_phase4_085_solve_adaptered_wrapper_imports_loop_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: adaptered games route through the reusable loop solver."""

    module = types.ModuleType("scripts.arc_loop_solve")
    module.solve_adaptered = lambda game, target_level: {  # type: ignore[attr-defined]
        "game": game,
        "target": target_level,
    }
    monkeypatch.setitem(sys.modules, "scripts.arc_loop_solve", module)

    assert exp._solve_adaptered("tu93", 4) == {"game": "tu93", "target": 4}


def test_req_phase4_085_adaptered_runner_accepts_tu93_l4_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: tu93 L4 counts when fresh_env branch-mode reset replay succeeds."""

    def fake_solve_adaptered(game: str, target_level: int) -> dict:
        assert (game, target_level) == ("tu93", 4)
        return {
            "game": "tu93",
            "target": 4,
            "reached_level": 4,
            "moves": 64,
            "states_expanded": 7678,
            "offline_reproduced": True,
            "solution_labels": ["CLICK:1:2"],
            "verifier_src": "hand_verifier_cold_start",
            "reproduction_gate": {
                "game": "tu93",
                "reached_level": 4,
                "claimed_level": 4,
                "reproduced": True,
            },
        }

    monkeypatch.setattr(exp, "_solve_adaptered", fake_solve_adaptered)

    row = exp._run_tu93_target(Path("."), 4361)

    assert row["new_reproduced_level"] == 4
    assert row["offline_reproduced"] is True
    assert row["checkpoint_status"] == "new_level_reproduced"
    assert row["residual_win_mechanic_gap_class"] == "none"


def test_req_phase4_085_loader_registry_and_internal_schema_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-085: loader, registry parsing, and fail-closed schema stay deterministic."""

    solver_path = tmp_path / exp.WORLD_MODEL_PATHS["tn36"]
    solver_path.parent.mkdir(parents=True)
    solver_path.write_text("VALUE = 17\n", encoding="utf-8")

    module = exp._load_tn36_solver(tmp_path)
    assert module.VALUE == 17

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    with pytest.raises(ImportError, match="cannot load"):
        exp._load_tn36_solver(tmp_path)

    registry = tmp_path / exp.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text("reproducible_total_levels: 32\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) == 32
    registry.write_text("no total here\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) is None
    registry.unlink()
    assert exp._registry_total(tmp_path) is None

    checks = {
        "targets": {
            game: {"offline_env_present": False, "offline_env_path": str(tmp_path / game)}
            for game in exp.TARGET_ORDER
        },
        "harness_import": True,
        "solver_kit_import": True,
        "arc_loop_solve_import": True,
        "executable_world_model_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "preconditions", lambda _repo: checks)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4361 artifact schema errors"):
        exp.run_experiment(random_seed=4361)
