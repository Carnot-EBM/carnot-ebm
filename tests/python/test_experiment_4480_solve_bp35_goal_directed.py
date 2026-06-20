"""Tests for Exp 4480 bp35 goal-directed navigation solve.

Spec refs: REQ-REPORT-4480, SCENARIO-REPORT-4480.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from carnot import experiment_4480_solve_bp35_goal_directed as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "arc_solver_kit_importable": True,
        "offline_arcade_reachable": True,
        "target_env_present": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / "environment_files" / "bp35" / "0a0ad940").mkdir(parents=True)
    (root / mod.ARC_REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "lf52",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                    },
                    {
                        "game": "bp35",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                        "dead_ends": [
                            {
                                "gap_id": mod.BP35_GAP_ID,
                                "status": "open",
                                "failure_mode": "untargeted_graph_explore_timeout",
                            }
                        ],
                    },
                ],
                "reproducible_total_levels": 46,
                "reproducible_total_games": 22,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_req_report_4480_spec_declares_bp35_goal_directed_contract() -> None:
    """REQ-REPORT-4480: OpenSpec declares the bp35 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4480" in spec
    assert "SCENARIO-REPORT-4480" in spec
    assert "goal-directed navigation solver" in spec
    assert "shape-changing avatar" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4480_goal_distance_and_state_key_are_shape_aware() -> None:
    """REQ-REPORT-4480: bp35 heuristic descends to the goal and keys avatar shape."""

    far = {
        "avatar": {
            "position": [7, 20],
            "shape_offsets": [[0, 0]],
            "image": "player_right",
            "facing_right": True,
            "gravity_up": True,
            "move_phase": 4,
        },
        "goal": {"position": [3, 7], "color": 14},
        "removable_blockers": [{"position": [7, 19], "name": "qclfkhjnaac"}],
    }
    near = {
        **far,
        "avatar": {**far["avatar"], "position": [4, 7], "image": "player_left"},
    }
    reshaped = {
        **far,
        "avatar": {**far["avatar"], "shape_offsets": [[0, 0], [1, 0]], "image": "player_left"},
    }

    assert mod.bp35_goal_distance(far) > mod.bp35_goal_distance(near)
    assert mod.bp35_state_key(far) != mod.bp35_state_key(reshaped)
    assert "shape_offsets" in mod.bp35_state_key_features()
    assert "local_removable_blockers" in mod.bp35_state_key_features()


def test_scenario_report_4480_default_bp35_plan_reproduces_l1() -> None:
    """SCENARIO-REPORT-4480: real bp35 labels pass arc_solver_kit.reproduce()."""

    result = mod.solve_bp35_goal_directed()
    gate = mod.reproduce_bp35_solution(result["solution"])

    assert result["grounded"] is True
    assert result["goal_region"]["color"] == 14
    assert result["shape_aware_state_key"] is True
    assert result["uses_goal_distance_heuristic"] is True
    assert len(result["solution"]) == 17
    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 1


def test_scenario_report_4480_run_banks_bp35_and_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4480: reproduced bp35 L1 updates the artifact and registry."""

    _write_fixture_repo(tmp_path)
    solution = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in mod.BP35_L1_ACTION_ROWS]
    reproduced_calls: list[list[str]] = []

    def solver() -> dict[str, Any]:
        return {
            "operator": mod.SOLVER_OPERATOR,
            "game": "bp35",
            "grounded": True,
            "solution": solution,
            "goal_region": {"position": [3, 7], "color": 14, "source": "fixture"},
            "states_expanded": 17,
            "uses_goal_distance_heuristic": True,
            "shape_aware_state_key": True,
            "state_key_features": mod.bp35_state_key_features(),
            "residual": "",
            "trace": [],
        }

    def reproduce(labels: Sequence[str]) -> dict[str, Any]:
        reproduced_calls.append(list(labels))
        return {
            "game": "bp35",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        solver_fn=solver,
        reproduce_fn=reproduce,
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
    )

    assert reproduced_calls == [solution]
    assert artifact["honest_verdict"] == "success: bp35_L1_goal_directed_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "bp35"
    assert artifact["goal_region_identified"] is True
    assert artifact["goal_directed_solver_built"] is True
    assert artifact["shape_aware_state_key"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["reproducible_total_levels"] == 47
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproduction_result"]["reproduced"] is True
    registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    bp35 = next(row for row in registry["games"] if row["game"] == "bp35")
    assert bp35["reproducibility"] == "reproduced"
    assert bp35["levels_reproduced"] == 1
    assert bp35["latest_exp4480_solve_bp35"]["artifact"] == mod.RESULT_RELATIVE_PATH
    assert bp35["dead_ends"][0]["status"] == "filled"
    assert registry["reproducible_total_levels"] == 47
    assert registry["reproducible_total_games"] == 23


def test_req_report_4480_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4480: blocked resources stop before solving and schema rejects fabrication."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_arcade_reachable": False, "ok": False},
        solver_fn=lambda: calls.append("solver") or {},
        reproduce_fn=lambda _solution: pytest.fail("blocked run must not reproduce"),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    bad = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "target_game": "",
        "goal_region_identified": "true",
        "goal_directed_solver_built": "true",
        "shape_aware_state_key": "true",
        "offline_reproduced": "false",
        "reproduced_levels": "0",
        "reproducible_total_levels": "47",
        "preconditions_checked": [],
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "solution_labels": {},
        "reproduction_result": [],
        "random_seed": "4480",
        "reproducibility_checksum": "bad",
        "submitted_to_leaderboard": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be bp35" in errors
    assert "goal_region_identified must be bare bool" in errors
    assert "goal_directed_solver_built must be bare bool" in errors
    assert "shape_aware_state_key must be bare bool" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "preconditions_checked must be dict" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "solution_labels must be list" in errors
    assert "reproduction_result must be dict" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4480" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)
