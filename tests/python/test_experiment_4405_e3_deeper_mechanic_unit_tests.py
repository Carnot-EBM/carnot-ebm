"""Tests for Exp 4405 ARC E3 mechanic-unit-test decomposition.

Spec refs: REQ-PHASE4-4405, SCENARIO-PHASE4-4405.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import carnot.experiment_4405_e3_deeper_mechanic_unit_tests as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _write_fake_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "games": [
                    {"game": "lp85", "levels_reproduced": 5},
                    {"game": "tu93", "levels_reproduced": 4},
                    {"game": "tn36", "levels_reproduced": 7},
                    {"game": "tr87", "levels_reproduced": 6},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_4394_e3_deeper_fidelity_gate.json").write_text(
        json.dumps(
            {
                "per_target_scorecard": [
                    {
                        "game": "lp85",
                        "lookahead_fidelity": 0.833333,
                        "verifier_accuracy": 0.833333,
                        "plan": [],
                    },
                    {
                        "game": "tu93",
                        "lookahead_fidelity": 0.8,
                        "verifier_accuracy": 0.8,
                        "plan": [],
                    },
                    {
                        "game": "tn36",
                        "lookahead_fidelity": 0.875,
                        "verifier_accuracy": 0.875,
                        "plan": ["mock-plan"],
                    },
                    {
                        "game": "tr87",
                        "lookahead_fidelity": 0.857143,
                        "verifier_accuracy": 0.857143,
                        "plan": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    for game in exp.TARGET_ORDER:
        env_dir = tmp_path / "environment_files" / game
        env_dir.mkdir(parents=True)
        (env_dir / "fixture").write_text("present\n", encoding="utf-8")
    gaps = tmp_path / "ops" / "verifier_gaps.md"
    gaps.write_text("# gaps\n", encoding="utf-8")


def _check(game: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "name": f"{game}_transition_unit",
        "transition": f"{game}:mismatch0",
        "passed": passed,
        "expected": "expected",
        "observed": "expected" if passed else "wrong",
        "test_path": "tests/python/test_experiment_4405_e3_deeper_mechanic_unit_tests.py",
    }


def test_req_phase4_4405_spec_declares_mechanic_decomposition_contract() -> None:
    """REQ-PHASE4-4405: OpenSpec declares the mechanic-unit-test artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4405" in spec
    assert "SCENARIO-PHASE4-4405" in spec
    assert "experiment_4405_e3_deeper_mechanic_unit_tests.json" in spec
    assert "mechanic_unit_tests_passed" in spec
    assert "residual_failing_mechanic" in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4405_mechanic_checks_execute_named_transitions() -> None:
    """REQ-PHASE4-4405: per-game checks assert the localized transition semantics."""

    checks = exp.run_mechanic_checks(REPO)
    by_game = {game: [row for row in checks if row["game"] == game] for game in exp.TARGET_ORDER}

    assert all(by_game.values())
    assert all(row["passed"] is True for row in checks)
    assert any(row["transition"] == "lp85:L6:button_permutation_slot_mapping" for row in by_game["lp85"])
    assert any(row["transition"] == "tu93:L5:fresh_env_branch_move" for row in by_game["tu93"])
    assert any(row["transition"] == "tn36:L8:sxhtkytekm_program_editor_run" for row in by_game["tn36"])
    assert any(row["transition"] == "tr87:L7:two_pass_greedy_rewrite" for row in by_game["tr87"])


def test_scenario_phase4_4405_partial_run_preserves_all_target_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-4405: no reproduced level is an honest mechanic-test partial."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "mechanic_checks_for_game", lambda _repo, game: [_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(write_artifact=True)

    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert all(row["mechanic_unit_tests_passed"] == 1 for row in artifact["per_target_scorecard"])
    assert all(row["offline_reproduced"] is False for row in artifact["per_target_scorecard"])
    assert artifact["verifier_is_oracle"] is True
    assert Path(artifact["artifact_path"]).exists()
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4405_counts_only_reproduced_new_level(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4405: passing unit tests do not count unless reproduce gates the level."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "mechanic_checks_for_game", lambda _repo, game: [_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    def reproduction_runner(game: str, target_level: int) -> dict[str, object]:
        return {
            "game": game,
            "claimed_level": target_level,
            "reached_level": target_level if game == "lp85" else target_level - 1,
            "reproduced": game == "lp85",
        }

    artifact = exp.run_experiment(reproduction_runner=reproduction_runner, write_artifact=False)
    rows = {row["game"]: row for row in artifact["per_target_scorecard"]}

    assert artifact["honest_verdict"] == "success_e3_deeper_lp85_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert rows["lp85"]["new_reproduced_level"] == 6
    assert rows["tu93"]["new_reproduced_level"] == 4


def test_req_phase4_4405_missing_env_blocks_one_target_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-PHASE4-4405: missing offline env records a blocked row only for that target."""

    _write_fake_repo(tmp_path)
    for child in (tmp_path / "environment_files" / "tr87").iterdir():
        child.unlink()
    monkeypatch.setattr(exp, "mechanic_checks_for_game", lambda _repo, game: [_check(game)])
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=False)
    rows = {row["game"]: row for row in artifact["per_target_scorecard"]}

    assert rows["tr87"]["checkpoint_status"] == "blocked_offline_env_missing_tr87"
    assert rows["tr87"]["mechanic_unit_tests_total"] == 0
    assert rows["lp85"]["checkpoint_status"] == "honest_partial_mechanic_tests_passed_reproduction_not_proven"


def test_req_phase4_4405_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4405: malformed artifacts report actionable schema errors."""

    bad = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": [{"game": "lp85", "offline_reproduced": "false"}],
        "reproducible_total_levels": "34",
        "new_levels_reproduced": {"value": 0},
        "world_model_paths": ["ok.py", 12],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4405",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard[0] missing prior_best_level" in errors
    assert "per_target_scorecard[0] missing mechanic_unit_tests_passed" in errors
    assert "per_target_scorecard[0].offline_reproduced must be bare bool" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors


def test_req_phase4_4405_verifier_gap_checkpoint_appends_once(tmp_path: Path) -> None:
    """REQ-PHASE4-4405: residual failing mechanics are recorded idempotently."""

    path = tmp_path / "ops" / "verifier_gaps.md"
    path.parent.mkdir()
    path.write_text("# gaps\n", encoding="utf-8")
    rows = [
        {
            "game": "lp85",
            "target_level": 6,
            "offline_reproduced": False,
            "residual_failing_mechanic": "lp85_l6_residual_gap",
            "mechanic_unit_tests_passed": 1,
            "mechanic_unit_tests_total": 1,
        }
    ]

    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    text = path.read_text(encoding="utf-8")

    assert text.count("exp4405-gap-lp85-l6:start") == 1
    assert "lp85_l6_residual_gap" in text
