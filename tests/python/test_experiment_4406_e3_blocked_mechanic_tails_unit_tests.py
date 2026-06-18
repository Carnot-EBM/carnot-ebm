"""Tests for Exp 4406 ARC E3 named-register unit-test decomposition.

Spec refs: REQ-PHASE4-4406, SCENARIO-PHASE4-4406.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import carnot.experiment_4406_e3_blocked_mechanic_tails_unit_tests as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _load_world_model(game: str) -> Any:
    path = REPO / exp.WORLD_MODEL_PATHS[game]
    spec = importlib.util.spec_from_file_location(f"arc_e3_{game}_wm_test_4406", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "games": [
                    {"game": "ar25", "levels_reproduced": 1},
                    {"game": "ka59", "levels_reproduced": 1},
                    {"game": "ft09", "levels_reproduced": 1},
                ],
                "reproducible_total_levels": 34,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# gaps\n", encoding="utf-8")
    for game in exp.TARGET_ORDER:
        env_dir = tmp_path / "environment_files" / game
        env_dir.mkdir(parents=True)
        (env_dir / "fixture").write_text("present\n", encoding="utf-8")
        wm_dir = tmp_path / "results" / "arc_e3" / game
        wm_dir.mkdir(parents=True)
        (wm_dir / "world_model.py").write_text(
            "def transition_fixture():\n"
            f"    return {{'transition': '{game}:fixture', 'passed': True, 'expected': 'ok', 'observed': 'ok'}}\n",
            encoding="utf-8",
        )


def _passing_check(game: str) -> dict[str, object]:
    return {
        "game": game,
        "name": f"{game}_register_unit",
        "transition": f"{game}:register",
        "passed": True,
        "expected": "expected",
        "observed": "expected",
        "test_path": exp.UNIT_TEST_PATH,
        "world_model_path": exp.WORLD_MODEL_PATHS[game],
    }


def test_req_phase4_4406_spec_declares_named_register_contract() -> None:
    """REQ-PHASE4-4406: OpenSpec declares the named-register artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4406" in spec
    assert "SCENARIO-PHASE4-4406" in spec
    assert "action 7 restores the prior grid through an undo stack" in spec
    assert "object-relevance discriminator" in spec
    assert "coverage-balanced residual click transition" in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4406_ar25_action7_restores_prior_grid() -> None:
    """REQ-PHASE4-4406: ar25 ACTION7 restores the hidden undo-stack prior grid."""

    wm = _load_world_model("ar25")
    prior = np.full((12, 12), 9, dtype=int)
    prior[4:7, 4:7] = 5
    moved = wm.engine(prior, 4, None)

    restored = wm.engine(moved, 7, {"undo_stack": [prior.tolist()]})

    assert not np.array_equal(moved, prior)
    assert np.array_equal(restored, prior)
    assert wm.transition_fixture()["passed"] is True
    assert wm.transition_fixture()["transition"] == "ar25:L2:action7_undo_stack_restore"


def test_req_phase4_4406_ka59_hud_ticks_only_for_relevant_object() -> None:
    """REQ-PHASE4-4406: ka59 HUD ticks iff the relevant object changes state."""

    wm = _load_world_model("ka59")
    grid = np.ones((12, 12), dtype=int)
    grid[4:7, 4:7] = 14
    grid[5, 5] = 0
    grid[-1, -5:] = 4

    irrelevant = wm.engine(
        grid,
        6,
        {"changed_object_id": "decorative", "relevant_object_ids": ["agent", "pushed_block"]},
    )
    relevant = wm.engine(
        grid,
        6,
        {"changed_object_id": "pushed_block", "relevant_object_ids": ["agent", "pushed_block"]},
    )

    assert int(np.count_nonzero(irrelevant[-1] == 4)) == 5
    assert int(np.count_nonzero(relevant[-1] == 4)) == 4
    fixture = wm.transition_fixture()
    assert fixture["passed"] is True
    assert fixture["object_relevance_discriminator"]["selected_object_hypothesis"] == "agent_plus_second_movable_block"


def test_req_phase4_4406_ft09_residual_click_toggles_component_only() -> None:
    """REQ-PHASE4-4406: ft09 residual click keeps unrelated cells stable."""

    wm = _load_world_model("ft09")
    grid = np.zeros((8, 8), dtype=int)
    grid[2:5, 2:5] = 8
    grid[2, 4] = 0
    grid[6, 6] = 8
    expected = grid.copy()
    expected[2:5, 2:5] = np.where(expected[2:5, 2:5] == 8, 9, expected[2:5, 2:5])

    observed = wm.engine(grid, 6, {"x": 3, "y": 3})

    assert np.array_equal(observed, expected)
    assert observed[2, 4] == 0
    assert observed[6, 6] == 8
    assert wm.transition_fixture()["passed"] is True
    assert wm.transition_fixture()["transition"] == "ft09:L2:component_click_residual"


def test_req_phase4_4406_register_checks_execute_named_transitions() -> None:
    """REQ-PHASE4-4406: per-game checks execute the localized register tests."""

    checks = exp.run_register_checks(REPO)
    by_game = {game: [row for row in checks if row["game"] == game] for game in exp.TARGET_ORDER}

    assert all(by_game.values())
    assert all(row["passed"] is True for row in checks)
    assert any(row["transition"] == "ar25:L2:action7_undo_stack_restore" for row in by_game["ar25"])
    assert any(row["transition"] == "ka59:L2:object_relevant_hud_tick" for row in by_game["ka59"])
    assert any(row["transition"] == "ft09:L2:component_click_residual" for row in by_game["ft09"])


def test_scenario_phase4_4406_partial_artifact_records_register_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-4406: passing register tests without L2 reproduce is honest partial."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "register_checks_for_game", lambda _repo, game: [_passing_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=True)

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_ft09_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert [row["game"] for row in artifact["per_game_scorecard"]] == list(exp.TARGET_ORDER)
    assert all(row["register_unit_test_passed"] is True for row in artifact["per_game_scorecard"])
    assert all(row["offline_reproduced"] is False for row in artifact["per_game_scorecard"])
    assert artifact["verifier_is_oracle"] is True
    assert Path(artifact["artifact_path"]).exists()
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4406_counts_only_reproduced_new_level(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4406: passing register tests do not count without reproduce()."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "register_checks_for_game", lambda _repo, game: [_passing_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    def reproduction_runner(game: str, target_level: int) -> dict[str, object]:
        return {
            "game": game,
            "claimed_level": target_level,
            "reached_level": target_level if game == "ar25" else target_level - 1,
            "reproduced": game == "ar25",
        }

    artifact = exp.run_experiment(
        repo=tmp_path,
        reproduction_runner=reproduction_runner,
        write_artifact=False,
    )
    rows = {row["game"]: row for row in artifact["per_game_scorecard"]}

    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert rows["ar25"]["new_reproduced_level"] == 2
    assert rows["ka59"]["new_reproduced_level"] == 1


def test_req_phase4_4406_missing_env_blocks_one_game_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-PHASE4-4406: missing offline env records a blocked row only for that game."""

    _write_fake_repo(tmp_path)
    for child in (tmp_path / "environment_files" / "ft09").iterdir():
        child.unlink()
    monkeypatch.setattr(exp, "register_checks_for_game", lambda _repo, game: [_passing_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=False)
    rows = {row["game"]: row for row in artifact["per_game_scorecard"]}

    assert rows["ft09"]["checkpoint_status"] == "blocked_offline_env_missing_ft09"
    assert rows["ft09"]["register_unit_tests_total"] == 0
    assert rows["ar25"]["checkpoint_status"] == "honest_partial_register_tests_passed_reproduction_not_proven"


def test_req_phase4_4406_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4406: malformed artifacts report actionable schema errors."""

    bad = {
        "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
        "per_game_scorecard": [{"game": "ka59", "offline_reproduced": "false"}],
        "reproducible_total_levels": "34",
        "new_levels_reproduced": {"value": 0},
        "world_model_paths": ["ok.py", 12],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4406",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_game_scorecard[0] missing named_register" in errors
    assert "per_game_scorecard[0] missing prior_best_level" in errors
    assert "per_game_scorecard[0] missing register_unit_test_passed" in errors
    assert "per_game_scorecard[0].offline_reproduced must be bare bool" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors


def test_req_phase4_4406_helper_branches_are_covered(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4406: filesystem/import helper edge cases stay deterministic."""

    _write_fake_repo(tmp_path)
    prior_path = tmp_path / exp.PRIOR_ARTIFACT_RELATIVE_PATH
    prior_path.write_text(
        json.dumps(
            {
                "per_game_scorecard": [
                    {"game": "ar25", "verifier_accuracy": 0.5},
                    {"game": "ignored", "verifier_accuracy": 1.0},
                    "not-a-row",
                ]
            }
        ),
        encoding="utf-8",
    )

    assert exp.read_registry_total(tmp_path / "missing-root") == exp.PRIOR_REPRODUCIBLE_TOTAL_LEVELS
    assert exp.load_prior_scorecards(tmp_path)["ar25"]["verifier_accuracy"] == 0.5
    assert exp._object_relevance_from_checks([{"object_relevance_discriminator": {"ok": True}}]) == {
        "ok": True
    }
    failed_result, reproduced = exp._reproduction_result("ar25", 1, 2, False, None)
    assert reproduced is False
    assert failed_result["reason"] == "register_unit_test_failed_no_planning"

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    try:
        exp._load_module_from_path(tmp_path, "missing.py", "missing_module")
    except ImportError as exc:
        assert "cannot load" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ImportError")


def test_req_phase4_4406_import_and_git_preconditions_are_reported(monkeypatch, tmp_path: Path) -> None:
    """REQ-PHASE4-4406: preconditions expose import and conductor status."""

    calls: list[str] = []

    def fake_import(name: str) -> object:
        calls.append(name)
        if name.endswith("arc_solver_kit"):
            raise RuntimeError("blocked")
        return object()

    monkeypatch.setattr(exp.importlib, "import_module", fake_import)
    imports = exp._imports_ok()
    assert imports == {"harness_import": True, "solver_kit_import": False}
    assert calls

    assert exp._research_conductor_modified(tmp_path) is False
    (tmp_path / ".git").mkdir()

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=" M scripts/research_conductor.py\n")

    monkeypatch.setattr(exp.subprocess, "run", fake_run)
    assert exp._research_conductor_modified(tmp_path) is True

    def raising_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", raising_run)
    assert exp._research_conductor_modified(tmp_path) is False


def test_req_phase4_4406_schema_edge_cases_and_error_gate(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4406: schema and run-error branches are explicit."""

    assert "missing honest_verdict" in exp.artifact_schema_errors({})
    assert "per_game_scorecard must be list" in exp.artifact_schema_errors(
        {
            "honest_verdict": "x",
            "per_game_scorecard": {},
            "reproducible_total_levels": 34,
            "new_levels_reproduced": 0,
            "world_model_paths": [],
            "verifier_is_oracle": True,
            "preconditions_checked": {},
            "random_seed": 4406,
            "reproducibility_checksum": "0" * 64,
            "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
        }
    )
    assert "per_game_scorecard[0] must be dict" in exp.artifact_schema_errors(
        {
            "honest_verdict": "x",
            "per_game_scorecard": ["bad"],
            "reproducible_total_levels": 34,
            "new_levels_reproduced": 0,
            "world_model_paths": [],
            "verifier_is_oracle": True,
            "preconditions_checked": {},
            "random_seed": 4406,
            "reproducibility_checksum": "0" * 64,
            "field_principles": {},
        }
    )
    assert "field_principles missing" in exp.artifact_schema_errors(
        {
            "honest_verdict": "x",
            "per_game_scorecard": [],
            "reproducible_total_levels": 34,
            "new_levels_reproduced": 0,
            "world_model_paths": [],
            "verifier_is_oracle": True,
            "preconditions_checked": {},
            "random_seed": 4406,
            "reproducibility_checksum": "0" * 64,
            "field_principles": None,
        }
    )

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "register_checks_for_game", lambda _repo, game: [_passing_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    try:
        exp.run_experiment(repo=tmp_path, write_artifact=False)
    except ValueError as exc:
        assert "forced" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ValueError")


def test_req_phase4_4406_verifier_gap_checkpoint_appends_once(tmp_path: Path) -> None:
    """REQ-PHASE4-4406: residual named-register gaps are recorded idempotently."""

    path = tmp_path / "ops" / "verifier_gaps.md"
    path.parent.mkdir()
    path.write_text("# gaps\n", encoding="utf-8")
    rows = [
        {
            "game": "ka59",
            "target_level": 2,
            "offline_reproduced": False,
            "residual_gap_class": "ka59_l2_object_relevance_step_counter_hud_register_gap",
            "register_unit_test_passed": True,
            "register_unit_tests_total": 1,
        }
    ]

    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    text = path.read_text(encoding="utf-8")

    assert text.count("exp4406-gap-ka59-l2:start") == 1
    assert "ka59_l2_object_relevance_step_counter_hud_register_gap" in text


def test_req_phase4_4406_verifier_gap_checkpoint_skip_paths(tmp_path: Path) -> None:
    """REQ-PHASE4-4406: gap checkpoint skips missing files and reproduced rows."""

    exp.write_verifier_gap_checkpoint(tmp_path, [{"game": "ar25", "offline_reproduced": False}])
    path = tmp_path / "ops" / "verifier_gaps.md"
    path.parent.mkdir()
    path.write_text("# gaps\n", encoding="utf-8")

    exp.write_verifier_gap_checkpoint(
        tmp_path,
        [{"game": "ar25", "target_level": 2, "offline_reproduced": True}],
    )

    assert path.read_text(encoding="utf-8") == "# gaps\n"
