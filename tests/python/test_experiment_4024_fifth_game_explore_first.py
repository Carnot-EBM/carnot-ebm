"""Tests for Exp 4024 ARC-AGI-3 fifth-game explore-first continuation.

Spec refs: REQ-PHASE4-032, SCENARIO-PHASE4-032.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import carnot.agentic.arc_exp4024_fifth_game_explore_first as helper
from carnot.agentic.arc_exp4024_fifth_game_explore_first import (
    REQUIRED_ARTIFACT_FIELDS,
    Cd82Plan,
    ExperimentOutcome,
    artifact_schema_errors,
    build_artifact,
    build_cd82_l1_plan,
    load_environment_baselines,
    select_new_candidate_from_survey,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4024_fifth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _l1_target() -> np.ndarray:
    target = np.zeros((10, 10), dtype=np.int16)
    target[5:10, :] = 15
    return target


def _outcome(*, solved: bool = True) -> ExperimentOutcome:
    return ExperimentOutcome(
        target_game="cd82-fb555c5d",
        selected_candidate_reason="selected cd82 from survey by smallest remaining L0 baseline",
        prior_total_games_solved=5,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=5 if solved else -1,
        exploration_actions_used=4,
        induced_mechanic="cd82 region-fill pattern matching",
        verification_decisions=[
            {
                "phase": "verify",
                "predicted_goal_after_action": solved,
                "retained": solved,
            }
        ],
        phase_trace=[
            {"phase": "observe", "level_completed": 0},
            {"phase": "explore", "action": 4, "level_completed": 0},
            {"phase": "induce", "mechanic": "region_fill"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "action": 5, "level_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_032_spec_declares_exp4024_contract() -> None:
    """REQ-PHASE4-032: OpenSpec declares Exp 4024 and the required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-032" in spec
    assert "SCENARIO-PHASE4-032" in spec
    assert "experiment_4024_fifth_game_explore_first.json" in spec
    assert "results/arc3_win_condition_survey.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_032_selects_cd82_from_remaining_non_spatial_survey() -> None:
    """REQ-PHASE4-032: target selection excludes solved games and picks the easiest new candidate."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_new_candidate_from_survey(survey, baselines)

    assert selected.game == "cd82"
    assert selected.game_id == "cd82-fb555c5d"
    assert selected.baseline_actions == 55
    assert selected.non_spatial is True
    assert "r11l" in selected.excluded_solved_games
    assert "tn36" in selected.excluded_solved_games
    assert selected.selection_reason.startswith("selected: cd82")


def test_req_phase4_032_cd82_plan_is_induced_from_target_difference() -> None:
    """REQ-PHASE4-032: the cd82 L1 plan is a verified region fill, not a blind click list."""

    current = np.zeros((10, 10), dtype=np.int16)
    plan = build_cd82_l1_plan(
        active_index=0,
        selected_color=15,
        current_canvas=current,
        target_canvas=_l1_target(),
    )

    assert isinstance(plan, Cd82Plan)
    assert plan.region_index == 4
    assert plan.fill_color == 15
    assert plan.actions == [4, 2, 2, 3, 5]
    assert plan.exploration_actions == [4, 2, 2, 3]
    assert plan.commit_action == 5
    assert plan.predicted_goal_after_commit is True
    assert np.array_equal(plan.predicted_canvas, _l1_target())


def test_req_phase4_032_cd82_plan_rejects_uncovered_target() -> None:
    """REQ-PHASE4-032: unsupported target diffs fail closed before action execution."""

    target = np.zeros((10, 10), dtype=np.int16)
    target[2, 3] = 9

    with pytest.raises(ValueError, match="single cd82 region fill"):
        build_cd82_l1_plan(
            active_index=0,
            selected_color=15,
            current_canvas=np.zeros((10, 10), dtype=np.int16),
            target_canvas=target,
        )


def test_req_phase4_032_artifact_schema_requires_required_bare_fields() -> None:
    """REQ-PHASE4-032: Exp 4024 success artifacts expose bare monotonic counters."""

    artifact = build_artifact(
        _outcome(),
        random_seed=4024,
        duration_s=1.25,
        inference_substrate="test_substrate",
    )

    assert artifact["honest_verdict"] == "success: fifth_game_solved_cd82-fb555c5d_at_action_5"
    assert artifact["game_solved"] is True
    assert artifact["total_games_solved"] == 6
    assert artifact["real_env_confirmed"] is True
    assert artifact["level_completed"] == 1
    assert artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["game_solved"] = "true"
    bad["total_games_solved"] = 5.0
    bad["real_env_confirmed"] = 1
    bad["inference_substrate"] = 7

    errors = artifact_schema_errors(bad)

    assert any("honest_verdict" in err for err in errors)
    assert any("game_solved" in err for err in errors)
    assert any("total_games_solved" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)


def test_scenario_phase4_032_no_solve_and_blocked_artifacts() -> None:
    """SCENARIO-PHASE4-032: no-solve and blocked artifacts keep honest counters."""

    no_solve = build_artifact(
        _outcome(solved=False),
        random_seed=4024,
        duration_s=0.5,
        inference_substrate="test_substrate",
    )

    assert no_solve["honest_verdict"] == (
        "complete: fifth_game_no_solve_cd82-fb555c5d_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 5
    assert no_solve["real_env_confirmed"] is False
    assert artifact_schema_errors(no_solve) == []

    blocked = helper.blocked_artifact(
        random_seed=4024,
        duration_s=0.0,
        inference_substrate="test_substrate",
    )

    assert blocked["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 5
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []


class _Sprite:
    def __init__(self, name: str, pixels: np.ndarray) -> None:
        self.name = name
        self.pixels = pixels


class _Level:
    def __init__(self) -> None:
        self.canvas = _Sprite("xytrjjbyib", np.zeros((10, 10), dtype=np.int16))
        self.target = _Sprite("eoqnvkspoa-pqwme1-1", _l1_target())

    def get_sprites_by_name(self, name: str) -> list[_Sprite]:
        return [self.canvas] if name == "xytrjjbyib" else []

    def get_sprites(self) -> list[_Sprite]:
        return [self.canvas, self.target]


class _Game:
    def __init__(self) -> None:
        self.xwmfgtlso = 0
        self.knqmgavuh = 15
        self._action_count = 0
        self.current_level = _Level()
        self.levels_completed = 0


class _Frame:
    def __init__(self, levels_completed: int) -> None:
        self.levels_completed = levels_completed
        self.level_completed = levels_completed
        self.state = "GameState.NOT_FINISHED"
        self.available_actions = [1, 2, 3, 4, 5, 6]
        self.frame = np.zeros((64, 64), dtype=np.int16)


class _Cd82Env:
    def __init__(self) -> None:
        self._game = _Game()

    def reset(self) -> _Frame:
        return _Frame(self._game.levels_completed)

    def step(self, action: object, data: dict[str, int] | None = None) -> _Frame:
        action_id = int(getattr(action, "value", action))
        self._game._action_count += 1
        if action_id in {1, 2, 3, 4}:
            self._game.xwmfgtlso = helper.move_basket_index(self._game.xwmfgtlso, action_id)
        if action_id == 5:
            plan = build_cd82_l1_plan(
                active_index=self._game.xwmfgtlso,
                selected_color=self._game.knqmgavuh,
                current_canvas=self._game.current_level.canvas.pixels,
                target_canvas=self._game.current_level.target.pixels,
            )
            self._game.current_level.canvas.pixels = plan.predicted_canvas
            if plan.predicted_goal_after_commit:
                self._game.levels_completed = 1
        return _Frame(self._game.levels_completed)


class _Arc:
    def make(self, game_id: str) -> _Cd82Env:
        assert game_id == "cd82-fb555c5d"
        return _Cd82Env()


def test_scenario_phase4_032_script_writes_success_from_level_counter(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-032: success requires real-env level_completed confirmation."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "cd82" / "fb555c5d"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "game_id": "cd82-fb555c5d",
                "baseline_actions": [55, 8, 41, 21, 23, 23],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: _Arc())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: fifth_game_solved_cd82-fb555c5d_at_action_5"
    assert artifact["game_solved"] is True
    assert artifact["total_games_solved"] == 6
    assert artifact["real_env_confirmed"] is True
    assert artifact["phase_trace"][0]["phase"] == "observe"
    assert artifact["phase_trace"][-1]["level_completed"] == 1
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["level_completed"] == 1


def test_scenario_phase4_032_script_writes_blocked_when_arcade_missing(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-032: unavailable offline Arcade writes a blocked artifact."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unavailable() -> object:
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == (
        "blocked_arc_offline_env_unavailable"
    )
