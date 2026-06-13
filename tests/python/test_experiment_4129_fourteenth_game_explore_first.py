"""Tests for Exp 4129 ARC-AGI-3 fourteenth-game explore-first solve.

Spec refs: REQ-PHASE4-049, SCENARIO-PHASE4-049.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_exp4129_fourteenth_game_explore_first as arc4129
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines
from carnot.agentic.arc_exp4129_fourteenth_game_explore_first import (
    BP35_GAME_ID,
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    Bp35ObservedState,
    Bp35Outcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_bp35_l1_plan,
    compute_actions_vs_baseline,
    select_exp4129_candidate_from_survey,
    validate_bp35_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4129_fourteenth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _state(*, level_completed: int = 0, player: tuple[int, int] = (3, 23)) -> Bp35ObservedState:
    return Bp35ObservedState(
        player_position=player,
        gem_position=(3, 7),
        gravity_direction="up",
        level_completed=level_completed,
        grid_size=(11, 36),
        removable_blocks=((7, 19), (4, 16), (4, 15), (5, 9)),
    )


def _candidate() -> arc4129.SelectedCandidate:
    return arc4129.SelectedCandidate(
        game="bp35",
        game_id=BP35_GAME_ID,
        baseline_actions=21,
        survey_is_spatial_planning=True,
        win_difficulty="hard",
        selection_mode="fallback_lowest_baseline_direct_observable_after_strict_nonspatial_exhausted",
        selection_reason=(
            "selected fallback: bp35 is the lowest-baseline remaining directly observable "
            "offline fixture, L0 baseline_actions=21"
        ),
        excluded_solved_games=arc4129.SOLVED_PREFIXES_BEFORE_FOURTEENTH,
    )


def _outcome(*, solved: bool = True) -> Bp35Outcome:
    plan = build_bp35_l1_plan(_state())
    final_state = _state(level_completed=1 if solved else 0, player=(3, 37) if solved else (4, 7))
    decision = validate_bp35_replayed_plan(_state(), final_state, plan)
    return Bp35Outcome(
        target_game=BP35_GAME_ID,
        selected_candidate_reason=_candidate().selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=16 if solved else -1,
        exploration_actions_used=len(plan.exploration_actions),
        induced_mechanic=(
            "Observed BP35 horizontal movement, upward falling through gaps, direct block removal, "
            "and gem contact level advancement."
        ),
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _state().to_json()},
            {"phase": "explore", "actions": [action.to_json() for action in plan.exploration_actions]},
            {"phase": "induce", "mechanic": "bp35_upward_gravity_gem_route"},
            decision,
            {"phase": "act", "levels_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_049_spec_declares_exp4129_contract() -> None:
    """REQ-PHASE4-049: OpenSpec declares Exp 4129 and required principle fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-049" in spec
    assert "SCENARIO-PHASE4-049" in spec
    assert "experiment_4129_fourteenth_game_explore_first.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert BP35_GAME_ID in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_games_solved", "levels_completed", "real_env_confirmed"):
        assert field in spec


def test_req_phase4_049_selects_bp35_after_strict_nonspatial_exhaustion() -> None:
    """REQ-PHASE4-049: selection chooses BP35 after strict non-spatial survey rows are exhausted."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_exp4129_candidate_from_survey(survey, baselines)

    assert selected.game == "bp35"
    assert selected.game_id == BP35_GAME_ID
    assert selected.baseline_actions == 21
    assert selected.selection_mode == "fallback_lowest_baseline_direct_observable_after_strict_nonspatial_exhausted"
    assert selected.survey_is_spatial_planning is True
    assert "tu93" in selected.excluded_solved_games
    assert arc4129.strict_nonspatial_candidates_exhausted(survey, baselines) is True

    strict_survey = {
        "per_game_surveys": [
            {
                "game": "aa01",
                "is_spatial_planning": False,
                "win_difficulty": "easy",
                "available_actions": "click-only",
                "win_condition_summary": "target match",
            },
            {
                "game": "bb02",
                "is_spatial_planning": False,
                "win_difficulty": "easy",
                "available_actions": "click-only",
                "win_condition_summary": "target match",
            },
        ]
    }
    strict_selected = select_exp4129_candidate_from_survey(
        strict_survey,
        {"aa01": ("aa01-game", 8), "bb02": ("bb02-game", 4)},
        solved_prefixes=(),
    )
    assert strict_selected.game == "bb02"
    assert strict_selected.selection_mode == "strict_survey_non_spatial"

    with pytest.raises(ValueError, match="no unsolved survey candidates"):
        select_exp4129_candidate_from_survey({"per_game_surveys": []}, {}, solved_prefixes=())


def test_req_phase4_049_bp35_plan_explores_then_validates_commit_suffix() -> None:
    """REQ-PHASE4-049: BP35 induction emits exploration before held-out validation."""

    plan = build_bp35_l1_plan(_state())

    assert [action.action for action in plan.actions] == [4, 4, 4, 4, 6, 3, 3, 6, 3, 6, 3, 4, 4, 6, 3, 3]
    assert [action.role for action in plan.actions[:5]] == [
        "move_right",
        "move_right",
        "move_right",
        "move_right_and_fall",
        "remove_overhead_block_and_fall",
    ]
    assert len(plan.exploration_actions) == 5
    assert len(plan.commit_actions) == 11
    assert [action.grid for action in plan.actions if action.action == 6] == [(7, 19), (4, 16), (4, 15), (5, 9)]
    assert plan.predicted_first_solve_at_action == 16
    assert plan.predicted_goal_after_commit is True
    assert plan.induction_call["goal_predicate"] == "player reaches the fjlzdjxhant gem and level counter increments"

    retained = validate_bp35_replayed_plan(_state(), _state(level_completed=1, player=(3, 37)), plan)
    assert retained["retained"] is True
    assert retained["heldout_transition_count"] == len(plan.commit_actions)
    assert retained["level_increment"] is True
    assert retained["predicted_goal_after_actions"] is True

    rejected = validate_bp35_replayed_plan(_state(), _state(level_completed=0, player=(3, 7)), plan)
    assert rejected["retained"] is False
    assert rejected["energy"] > 0.0

    with pytest.raises(ValueError, match="BP35 plan expects upward gravity"):
        build_bp35_l1_plan(Bp35ObservedState((3, 23), (3, 7), "down", 0, (11, 36), ()))
    with pytest.raises(ValueError, match="expected BP35 first-level start"):
        build_bp35_l1_plan(_state(player=(1, 1)))


def test_scenario_phase4_049_success_and_blocked_artifacts_validate() -> None:
    """SCENARIO-PHASE4-049: success increments only with real-env-confirmed evidence."""

    artifact = build_artifact(
        _outcome(solved=True),
        _candidate(),
        random_seed=4129,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: fourteenth_game_solved_bp35-0a0ad940_at_action_16"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == BP35_GAME_ID
    assert artifact["total_games_solved"] == 13
    assert artifact["levels_completed"] == 1
    assert artifact["first_solve_at_action"] == 16
    assert artifact["actions_vs_baseline"] == 0.7619
    assert artifact["real_env_confirmed"] is True
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert artifact["requirements"] == ["REQ-PHASE4-049", "SCENARIO-PHASE4-049"]
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact

    no_solve = build_artifact(
        _outcome(solved=False),
        _candidate(),
        random_seed=4129,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == "complete: fourteenth_game_no_solve_bp35-0a0ad940_level_counter_did_not_increment"
    assert no_solve["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert no_solve["real_env_confirmed"] is False
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(
        target_game=BP35_GAME_ID,
        random_seed=4129,
        duration_s=0.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert blocked["levels_completed"] == 0
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    assert compute_actions_vs_baseline(16, 21, solved=True) == 0.7619
    assert compute_actions_vs_baseline(-1, 21, solved=False) == 0.0
    with pytest.raises(ValueError, match="baseline_actions"):
        compute_actions_vs_baseline(16, 0, solved=True)
    with pytest.raises(ValueError, match="first_solve_at_action"):
        compute_actions_vs_baseline(0, 21, solved=True)


def test_scenario_phase4_049_schema_rejects_fabricated_success(monkeypatch) -> None:
    """SCENARIO-PHASE4-049: malformed success artifacts cannot inflate the solved count."""

    assert any("missing required field levels_completed" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4129}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("game_solved must be a bare bool" in err for err in artifact_schema_errors({"game_solved": "yes"}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4129}))
    assert any("total_games_solved must be a bare int" in err for err in artifact_schema_errors({"total_games_solved": "13"}))
    assert any("levels_completed must be a bare int" in err for err in artifact_schema_errors({"levels_completed": True}))
    assert any("first_solve_at_action must be a bare int" in err for err in artifact_schema_errors({"first_solve_at_action": 1.0}))
    assert any("actions_vs_baseline must be a bare float" in err for err in artifact_schema_errors({"actions_vs_baseline": "0.7"}))
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 1}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("inference_substrate must equal" in err for err in artifact_schema_errors({"inference_substrate": "wrong"}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))

    bad_success = {
        "honest_verdict": "success: fourteenth_game_solved_bp35-0a0ad940_at_action_16",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": 12,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "actions_vs_baseline": 0.0,
        "real_env_confirmed": False,
        "solve_trace": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    errors = artifact_schema_errors(bad_success)
    assert any("game_solved must be true" in err for err in errors)
    assert any("target_game must name" in err for err in errors)
    assert any("real_env_confirmed must be true" in err for err in errors)
    assert any("total_games_solved must increment" in err for err in errors)
    assert any("levels_completed must increment" in err for err in errors)
    assert any("first_solve_at_action must be positive" in err for err in errors)
    assert any("actions_vs_baseline must be positive" in err for err in errors)
    assert any("solve_trace must include actions" in err for err in errors)

    monkeypatch.setattr(arc4129, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(solved=True), _candidate(), random_seed=4129, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game=BP35_GAME_ID, random_seed=4129, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)


def test_scenario_phase4_049_script_writes_success_from_fake_offline_env(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-049: runner writes success when the offline env confirms BP35 L1."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "bp35" / "0a0ad940"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": BP35_GAME_ID, "baseline_actions": [21]}),
        encoding="utf-8",
    )
    metadata_dir.joinpath("bp35.py").write_text("# synthetic fixture marker\n", encoding="utf-8")

    candidate = _candidate()
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: type("Arcade", (), {"get_environments": lambda self: [BP35_GAME_ID]})())
    monkeypatch.setattr(exp, "_run_bp35_explore_first", lambda arcade, selected, arc_env_count: _outcome(solved=True))

    artifact = exp.run(write=True)

    assert candidate.game_id == artifact["target_game"]
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["total_games_solved"] == 13
    written = json.loads((tmp_path / "results" / "experiment_4129_fourteenth_game_explore_first.json").read_text())
    assert written == artifact
