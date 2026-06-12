"""Tests for Exp 4070 ARC-AGI-3 ninth-game explore-first solve.

Spec refs: REQ-PHASE4-042, SCENARIO-PHASE4-042.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentOutcome,
    Ft09Action,
    Ft09Cell,
    Ft09Constraint,
    Ft09ObservedState,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_ft09_l1_plan,
    load_environment_baselines,
    select_ninth_candidate_from_survey,
    validate_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4070_ninth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _ft09_l1_state(*, level_completed: int = 0) -> Ft09ObservedState:
    return Ft09ObservedState(
        constraints=(
            Ft09Constraint(
                grid=(22, 22),
                center_color=8,
                pattern=((0, 2, 2), (0, 8, 0), (0, 2, 2)),
            ),
        ),
        cells=(
            Ft09Cell(grid=(18, 18), color=9, kind="Hkx"),
            Ft09Cell(grid=(22, 18), color=9, kind="Hkx"),
            Ft09Cell(grid=(26, 18), color=9, kind="Hkx"),
            Ft09Cell(grid=(18, 22), color=9, kind="Hkx"),
            Ft09Cell(grid=(26, 22), color=9, kind="Hkx"),
            Ft09Cell(grid=(18, 26), color=9, kind="Hkx"),
            Ft09Cell(grid=(22, 26), color=9, kind="Hkx"),
            Ft09Cell(grid=(26, 26), color=9, kind="Hkx"),
        ),
        color_cycle=(9, 8),
        level_completed=level_completed,
    )


def _ft09_solved_state(*, level_completed: int = 1) -> Ft09ObservedState:
    return Ft09ObservedState(
        constraints=_ft09_l1_state().constraints,
        cells=(
            Ft09Cell(grid=(18, 18), color=8, kind="Hkx"),
            Ft09Cell(grid=(22, 18), color=9, kind="Hkx"),
            Ft09Cell(grid=(26, 18), color=9, kind="Hkx"),
            Ft09Cell(grid=(18, 22), color=8, kind="Hkx"),
            Ft09Cell(grid=(26, 22), color=8, kind="Hkx"),
            Ft09Cell(grid=(18, 26), color=8, kind="Hkx"),
            Ft09Cell(grid=(22, 26), color=9, kind="Hkx"),
            Ft09Cell(grid=(26, 26), color=9, kind="Hkx"),
        ),
        color_cycle=(9, 8),
        level_completed=level_completed,
    )


def _outcome(*, solved: bool = True) -> ExperimentOutcome:
    plan = build_ft09_l1_plan(_ft09_l1_state())
    final_state = _ft09_solved_state(level_completed=1 if solved else 0)
    decision = validate_replayed_plan(_ft09_l1_state(), final_state, plan)
    return ExperimentOutcome(
        target_game="ft09-0d8bbf25",
        selected_candidate_reason="selected fallback: ft09 is the easiest click-only local-constraint target",
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=4 if solved else -1,
        exploration_actions_used=1,
        induced_mechanic=(
            "Observed ft09 click-to-cycle cells and induced the local bsT equality/inequality "
            "constraint goal predicate."
        ),
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _ft09_l1_state().to_json()},
            {"phase": "explore", "action": plan.actions[0].to_json()},
            {"phase": "induce", "mechanic": "ft09_local_constraint_color_cycle"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "level_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_042_spec_declares_exp4070_contract() -> None:
    """REQ-PHASE4-042: OpenSpec declares Exp 4070 and all required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-042" in spec
    assert "SCENARIO-PHASE4-042" in spec
    assert "experiment_4070_ninth_game_explore_first.json" in spec
    assert "blocked_arc_env_unreachable" in spec
    assert "ft09-0d8bbf25" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_042_selects_ft09_after_solved_and_navigation_exclusions() -> None:
    """REQ-PHASE4-042: selection excludes eight solved games, avoids vc33, then picks ft09."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_ninth_candidate_from_survey(survey, baselines)

    assert selected.game == "ft09"
    assert selected.game_id == "ft09-0d8bbf25"
    assert selected.baseline_actions == 43
    assert selected.selection_mode == "fallback_click_local_constraint_non_navigation"
    assert selected.survey_is_spatial_planning is True
    assert "sb26" in selected.excluded_solved_games
    assert selected.selection_reason.startswith("selected fallback: ft09")


def test_req_phase4_042_selection_defensive_paths(tmp_path: Path) -> None:
    """REQ-PHASE4-042: metadata loading, strict selection, and no-candidate paths are explicit."""

    metadata_dir = tmp_path / "environment_files" / "zz99" / "abc"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "zz99-abc", "baseline_actions": []}),
        encoding="utf-8",
    )
    assert load_environment_baselines(tmp_path / "environment_files") == {}

    strict = {
        "per_game_surveys": [
            {
                "game": "aa01",
                "is_spatial_planning": False,
                "win_difficulty": "medium",
                "available_actions": "click-only [6]",
            },
            {
                "game": "ft09",
                "is_spatial_planning": True,
                "win_difficulty": "hard",
                "available_actions": "click-only [6]",
                "first_solve_recipe": "decode a constraint graph",
                "win_condition_summary": "local constraints",
            },
        ]
    }
    selected = select_ninth_candidate_from_survey(
        strict,
        {"aa01": ("aa01-abc", 60), "ft09": ("ft09-0d8bbf25", 43)},
        solved_prefixes=(),
    )
    assert selected.game == "aa01"
    assert selected.selection_mode == "strict_survey_non_spatial"

    no_candidate = {
        "per_game_surveys": [
            {
                "game": "s5i5",
                "is_spatial_planning": True,
                "available_actions": "click-only [6]",
                "first_solve_recipe": "drag, resize, and avoid collisions",
                "win_condition_summary": "target positions with collisions",
            },
        ]
    }
    with pytest.raises(ValueError, match="no unsolved non-spatial"):
        select_ninth_candidate_from_survey(no_candidate, {"s5i5": ("s5i5-id", 20)}, solved_prefixes=())


def test_req_phase4_042_ft09_plan_explores_then_satisfies_local_constraints() -> None:
    """REQ-PHASE4-042: ft09 induction emits positive exploration before commit suffix."""

    state = _ft09_l1_state()
    plan = build_ft09_l1_plan(state)

    assert state.violation_count == 4
    assert plan.predicted_goal_after_commit is True
    assert plan.exploration_actions == plan.actions[:1]
    assert plan.commit_actions == plan.actions[1:]
    assert plan.predicted_cell_colors == {
        (18, 18): 8,
        (22, 18): 9,
        (26, 18): 9,
        (18, 22): 8,
        (26, 22): 8,
        (18, 26): 8,
        (22, 26): 9,
        (26, 26): 9,
    }
    assert [action.to_json() for action in plan.actions] == [
        {"action": 6, "x": 36, "y": 36, "grid": [18, 18], "role": "cycle_cell", "target_color": 8},
        {"action": 6, "x": 36, "y": 44, "grid": [18, 22], "role": "cycle_cell", "target_color": 8},
        {"action": 6, "x": 52, "y": 44, "grid": [26, 22], "role": "cycle_cell", "target_color": 8},
        {"action": 6, "x": 36, "y": 52, "grid": [18, 26], "role": "cycle_cell", "target_color": 8},
    ]
    assert plan.induction_call["call"] == "induce_ft09_local_constraint_color_cycle"
    assert plan.induction_call["goal_predicate"] == "all bsT neighbor equality/inequality constraints hold"

    with pytest.raises(ValueError, match="color_cycle"):
        build_ft09_l1_plan(Ft09ObservedState(constraints=state.constraints, cells=state.cells, color_cycle=(9,), level_completed=0))

    with pytest.raises(ValueError, match="missing clickable cell"):
        build_ft09_l1_plan(
            Ft09ObservedState(
                constraints=(Ft09Constraint(grid=(0, 0), center_color=8, pattern=((0, 2, 2), (2, 8, 2), (2, 2, 2))),),
                cells=state.cells,
                color_cycle=(9, 8),
                level_completed=0,
            )
        )

    not_equal_state = Ft09ObservedState(
        constraints=(
            Ft09Constraint(
                grid=(10, 10),
                center_color=9,
                pattern=((2, 2, 2), (2, 9, 2), (2, 2, 2)),
            ),
        ),
        cells=(Ft09Cell(grid=(6, 6), color=9, kind="Hkx"),),
        color_cycle=(9, 8),
        level_completed=0,
    )
    not_equal_plan = build_ft09_l1_plan(not_equal_state)
    assert not_equal_state.violation_count == 1
    assert not_equal_plan.actions == [Ft09Action.click_cell((6, 6), target_color=8)]
    assert not_equal_plan.predicted_cell_colors[(6, 6)] == 8


def test_scenario_phase4_042_gap4_replay_verifier_requires_constraints_and_level_increment() -> None:
    """SCENARIO-PHASE4-042: verifier keeps only local-constraint plans with a level increment."""

    start = _ft09_l1_state()
    plan = build_ft09_l1_plan(start)
    decision = validate_replayed_plan(start, _ft09_solved_state(), plan)

    assert decision["retained"] is True
    assert decision["energy"] == 0.0
    assert decision["level_increment"] is True
    assert decision["predicted_goal_after_actions"] is True

    no_increment = validate_replayed_plan(start, _ft09_solved_state(level_completed=0), plan)
    assert no_increment["retained"] is False
    assert no_increment["energy"] > 0.0
    assert no_increment["level_increment"] is False

    wrong_color = validate_replayed_plan(
        start,
        Ft09ObservedState(
            constraints=start.constraints,
            cells=(
                Ft09Cell(grid=(18, 18), color=8, kind="Hkx"),
                Ft09Cell(grid=(22, 18), color=9, kind="Hkx"),
                Ft09Cell(grid=(26, 18), color=9, kind="Hkx"),
                Ft09Cell(grid=(18, 22), color=8, kind="Hkx"),
                Ft09Cell(grid=(26, 22), color=9, kind="Hkx"),
                Ft09Cell(grid=(18, 26), color=8, kind="Hkx"),
                Ft09Cell(grid=(22, 26), color=9, kind="Hkx"),
                Ft09Cell(grid=(26, 26), color=9, kind="Hkx"),
            ),
            color_cycle=(9, 8),
            level_completed=1,
        ),
        plan,
    )
    assert wrong_color["retained"] is False
    assert wrong_color["final_violation_count"] == 1


def test_req_phase4_042_artifact_schema_success_no_solve_and_blocked() -> None:
    """REQ-PHASE4-042: artifacts preserve solve trace, counter, and real-env confirmation."""

    artifact = build_artifact(
        _outcome(),
        random_seed=4070,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: ninth_game_solved_ft09-0d8bbf25_at_action_4"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == "ft09-0d8bbf25"
    assert artifact["total_games_solved"] == 9
    assert artifact["real_env_confirmed"] is True
    assert artifact["solve_trace"]["induction_calls"][0]["call"] == "induce_ft09_local_constraint_color_cycle"
    assert artifact["field_principles"]["solve_trace"].startswith("full observe")
    assert artifact_schema_errors(artifact) == []

    no_solve = build_artifact(
        _outcome(solved=False),
        random_seed=4070,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: ninth_game_no_solve_ft09-0d8bbf25_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 8
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(random_seed=4070, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["target_game"] == "none"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 8
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "done",
            "game_solved": "true",
            "target_game": 4070,
            "total_games_solved": 9.0,
            "real_env_confirmed": 1,
            "solve_trace": [],
            "inference_substrate": None,
        }
    )
    errors = artifact_schema_errors(bad)
    assert any("honest_verdict" in err for err in errors)
    assert any("game_solved" in err for err in errors)
    assert any("target_game" in err for err in errors)
    assert any("total_games_solved" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("solve_trace" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))

    non_string_verdict = dict(artifact)
    non_string_verdict["honest_verdict"] = 4070
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors(non_string_verdict))

    success_bad = dict(artifact)
    success_bad.update(
        {
            "honest_verdict": "success: ninth_game_solved_none_at_action_0",
            "game_solved": False,
            "target_game": "none",
            "total_games_solved": 8,
            "real_env_confirmed": False,
            "level_completed": 0,
            "first_solve_at_action": 0,
            "exploration_actions_used": 0,
            "solve_trace": {},
        }
    )
    success_errors = artifact_schema_errors(success_bad)
    assert any("game_solved must be true" in err for err in success_errors)
    assert any("target_game must name" in err for err in success_errors)
    assert any("real_env_confirmed must be true" in err for err in success_errors)
    assert any("total_games_solved must increment" in err for err in success_errors)
    assert any("level_completed must increment" in err for err in success_errors)
    assert any("first_solve_at_action" in err for err in success_errors)
    assert any("exploration_actions_used" in err for err in success_errors)
    assert any("solve_trace must include actions" in err for err in success_errors)

    with pytest.raises(ValueError, match="inference_substrate"):
        build_artifact(_outcome(), random_seed=4070, duration_s=0.1, inference_substrate=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference_substrate"):
        blocked_artifact(random_seed=4070, duration_s=0.0, inference_substrate=None)  # type: ignore[arg-type]


def test_scenario_phase4_042_script_writes_success_from_live_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-042: script writes success only from confirmed outcome evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "ft09" / "0d8bbf25"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "ft09-0d8bbf25", "baseline_actions": [43]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", lambda: 25)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_load_online_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_ft09_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: ninth_game_solved_ft09-0d8bbf25_at_action_4"
    assert artifact["total_games_solved"] == 9
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["target_game"] == "ft09-0d8bbf25"


def test_scenario_phase4_042_script_blocks_when_arc_env_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-042: live ARC precondition failure stops with blocked verdict."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unreachable() -> int:
        raise RuntimeError("catalog down")

    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", unreachable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_env_unreachable"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == "blocked_arc_env_unreachable"
