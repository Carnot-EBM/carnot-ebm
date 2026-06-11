"""Tests for Exp 4049 ARC-AGI-3 eighth-game explore-first solve.

Spec refs: REQ-PHASE4-041, SCENARIO-PHASE4-041.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_exp4049_eighth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    ExperimentOutcome,
    Sb26Action,
    Sb26ClickTarget,
    Sb26ObservedState,
    Sb26Slot,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_sb26_l1_plan,
    load_environment_baselines,
    select_eighth_candidate_from_survey,
    validate_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4049_eighth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _sb26_l1_state(*, level_completed: int = 0) -> Sb26ObservedState:
    return Sb26ObservedState(
        target_colors=(9, 14, 11, 15),
        slots=(
            Sb26Slot(x=20, y=27, color=None),
            Sb26Slot(x=26, y=27, color=None),
            Sb26Slot(x=32, y=27, color=None),
            Sb26Slot(x=38, y=27, color=None),
        ),
        items=(
            Sb26ClickTarget(x=17, y=56, color=14, name="lngftsryyw"),
            Sb26ClickTarget(x=25, y=56, color=15, name="lngftsryyw"),
            Sb26ClickTarget(x=33, y=56, color=9, name="lngftsryyw"),
            Sb26ClickTarget(x=41, y=56, color=11, name="lngftsryyw"),
        ),
        level_completed=level_completed,
    )


def _sb26_solved_state(*, level_completed: int = 1) -> Sb26ObservedState:
    return Sb26ObservedState(
        target_colors=(9, 14, 11, 15),
        slots=(
            Sb26Slot(x=20, y=27, color=9),
            Sb26Slot(x=26, y=27, color=14),
            Sb26Slot(x=32, y=27, color=11),
            Sb26Slot(x=38, y=27, color=15),
        ),
        items=(),
        level_completed=level_completed,
    )


def _outcome(*, solved: bool = True) -> ExperimentOutcome:
    plan = build_sb26_l1_plan(_sb26_l1_state())
    final_state = _sb26_solved_state(level_completed=1 if solved else 0)
    decision = validate_replayed_plan(_sb26_l1_state(), final_state, plan)
    return ExperimentOutcome(
        target_game="sb26-7fbdac44",
        selected_candidate_reason=(
            "selected fallback: sb26 is the lowest-baseline click-sequence "
            "color-matching candidate after strict non-spatial rows were exhausted"
        ),
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=9 if solved else -1,
        exploration_actions_used=2,
        induced_mechanic=(
            "Observed sb26 item selection plus slot placement before validation; "
            "induced left-to-right target-color matching followed by ACTION5 validation."
        ),
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _sb26_l1_state().to_json()},
            {"phase": "explore", "action": plan.actions[0].to_json()},
            {"phase": "explore", "action": plan.actions[1].to_json()},
            {"phase": "induce", "mechanic": "sb26_color_sequence_slot_matching"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "level_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_041_spec_declares_exp4049_contract() -> None:
    """REQ-PHASE4-041: OpenSpec declares Exp 4049 and all required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-041" in spec
    assert "SCENARIO-PHASE4-041" in spec
    assert "experiment_4049_eighth_game_explore_first.json" in spec
    assert "blocked_arc_env_unreachable" in spec
    assert "sb26-7fbdac44" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_041_selects_sb26_when_strict_non_spatial_set_is_exhausted() -> None:
    """REQ-PHASE4-041: selection excludes seven solved games, avoids vc33, then picks sb26."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_eighth_candidate_from_survey(survey, baselines)

    assert selected.game == "sb26"
    assert selected.game_id == "sb26-7fbdac44"
    assert selected.baseline_actions == 18
    assert selected.selection_mode == "fallback_click_sequence_non_navigation"
    assert selected.survey_is_spatial_planning is True
    assert "dc22" in selected.excluded_solved_games
    assert selected.selection_reason.startswith("selected fallback: sb26")


def test_req_phase4_041_selection_defensive_paths(tmp_path: Path) -> None:
    """REQ-PHASE4-041: metadata loading, strict selection, and no-candidate paths are explicit."""

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
                "game": "bb02",
                "is_spatial_planning": True,
                "win_difficulty": "hard",
                "available_actions": "click-only [6] + 2 additional actions [5, 7]",
                "first_solve_recipe": "match a sequence",
                "win_condition_summary": "Match a sequence",
            },
        ]
    }
    selected = select_eighth_candidate_from_survey(
        strict,
        {"aa01": ("aa01-abc", 40), "bb02": ("bb02-def", 10)},
        solved_prefixes=(),
    )
    assert selected.game == "aa01"
    assert selected.selection_mode == "strict_survey_non_spatial"

    no_candidate = {
        "per_game_surveys": [
            {
                "game": "vc33",
                "is_spatial_planning": False,
                "available_actions": "click-only [6]",
            },
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
        select_eighth_candidate_from_survey(no_candidate, {"s5i5": ("s5i5-id", 20)}, solved_prefixes=())


def test_req_phase4_041_sb26_plan_explores_then_matches_target_sequence() -> None:
    """REQ-PHASE4-041: sb26 induction emits positive exploration before commit suffix."""

    plan = build_sb26_l1_plan(_sb26_l1_state())

    assert plan.predicted_slot_colors == (9, 14, 11, 15)
    assert plan.predicted_goal_after_commit is True
    assert plan.exploration_actions == plan.actions[:2]
    assert plan.commit_actions == plan.actions[2:]
    assert [action.to_json() for action in plan.actions] == [
        {"action": 6, "x": 36, "y": 59, "sprite": "lngftsryyw", "role": "select_item", "color": 9},
        {"action": 6, "x": 23, "y": 30, "role": "place_slot", "color": 9},
        {"action": 6, "x": 20, "y": 59, "sprite": "lngftsryyw", "role": "select_item", "color": 14},
        {"action": 6, "x": 29, "y": 30, "role": "place_slot", "color": 14},
        {"action": 6, "x": 44, "y": 59, "sprite": "lngftsryyw", "role": "select_item", "color": 11},
        {"action": 6, "x": 35, "y": 30, "role": "place_slot", "color": 11},
        {"action": 6, "x": 28, "y": 59, "sprite": "lngftsryyw", "role": "select_item", "color": 15},
        {"action": 6, "x": 41, "y": 30, "role": "place_slot", "color": 15},
        {"action": 5, "role": "validate"},
    ]
    assert plan.induction_call["call"] == "induce_sb26_color_sequence_slot_matching"
    assert plan.induction_call["goal_predicate"] == "slot_colors == target_colors and ACTION5 increments level counter"
    assert _sb26_l1_state().remaining_mismatches == 4
    assert _sb26_l1_state().to_json()["loose_item_colors"] == [14, 15, 9, 11]

    with pytest.raises(ValueError, match="no available item"):
        build_sb26_l1_plan(
            Sb26ObservedState(
                target_colors=(1,),
                slots=(Sb26Slot(x=0, y=0, color=None),),
                items=(Sb26ClickTarget(x=10, y=10, color=2, name="item"),),
                level_completed=0,
            )
        )
    with pytest.raises(ValueError, match="slot count"):
        build_sb26_l1_plan(
            Sb26ObservedState(
                target_colors=(1, 2),
                slots=(Sb26Slot(x=0, y=0, color=None),),
                items=(Sb26ClickTarget(x=10, y=10, color=1, name="item"),),
                level_completed=0,
            )
        )


def test_scenario_phase4_041_gap4_replay_verifier_requires_sequence_and_level_increment() -> None:
    """SCENARIO-PHASE4-041: verifier keeps only replayed plans with a level increment."""

    start = _sb26_l1_state()
    plan = build_sb26_l1_plan(start)
    decision = validate_replayed_plan(start, _sb26_solved_state(), plan)

    assert decision["retained"] is True
    assert decision["energy"] == 0.0
    assert decision["level_increment"] is True
    assert decision["predicted_goal_after_actions"] is True

    no_increment = validate_replayed_plan(start, _sb26_solved_state(level_completed=0), plan)
    assert no_increment["retained"] is False
    assert no_increment["energy"] > 0.0
    assert no_increment["level_increment"] is False

    wrong_slots = validate_replayed_plan(
        start,
        Sb26ObservedState(
            target_colors=(9, 14, 11, 15),
            slots=(
                Sb26Slot(x=20, y=27, color=14),
                Sb26Slot(x=26, y=27, color=9),
                Sb26Slot(x=32, y=27, color=11),
                Sb26Slot(x=38, y=27, color=15),
            ),
            items=(),
            level_completed=1,
        ),
        plan,
    )
    assert wrong_slots["retained"] is False
    assert wrong_slots["slot_sequence_matches"] is False


def test_req_phase4_041_artifact_schema_success_no_solve_and_blocked() -> None:
    """REQ-PHASE4-041: artifacts preserve solve trace, counter, and real-env confirmation."""

    artifact = build_artifact(
        _outcome(),
        random_seed=4049,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: eighth_game_solved_sb26-7fbdac44_at_action_9"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == "sb26-7fbdac44"
    assert artifact["total_games_solved"] == 8
    assert artifact["real_env_confirmed"] is True
    assert artifact["solve_trace"]["induction_calls"][0]["call"] == "induce_sb26_color_sequence_slot_matching"
    assert artifact["field_principles"]["solve_trace"].startswith("full observe")
    assert artifact_schema_errors(artifact) == []

    no_solve = build_artifact(
        _outcome(solved=False),
        random_seed=4049,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: eighth_game_no_solve_sb26-7fbdac44_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 7
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(random_seed=4049, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["target_game"] == "none"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 7
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "done",
            "game_solved": "true",
            "target_game": 4049,
            "total_games_solved": 8.0,
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
    non_string_verdict["honest_verdict"] = 4049
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors(non_string_verdict))

    success_bad = dict(artifact)
    success_bad.update(
        {
            "honest_verdict": "success: eighth_game_solved_none_at_action_0",
            "game_solved": False,
            "target_game": "none",
            "total_games_solved": 7,
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
        build_artifact(_outcome(), random_seed=4049, duration_s=0.1, inference_substrate=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference_substrate"):
        blocked_artifact(random_seed=4049, duration_s=0.0, inference_substrate=None)  # type: ignore[arg-type]


def test_scenario_phase4_041_script_writes_success_from_live_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-041: script writes success only from confirmed outcome evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "sb26" / "7fbdac44"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "sb26-7fbdac44", "baseline_actions": [18]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", lambda: 25)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_load_online_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_sb26_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: eighth_game_solved_sb26-7fbdac44_at_action_9"
    assert artifact["total_games_solved"] == 8
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["target_game"] == "sb26-7fbdac44"


def test_scenario_phase4_041_script_blocks_when_arc_env_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-041: live ARC precondition failure stops with blocked verdict."""

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
