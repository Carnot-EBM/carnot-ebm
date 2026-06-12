"""Tests for Exp 4110 ARC-AGI-3 twelfth-game explore-first solve.

Spec refs: REQ-PHASE4-047, SCENARIO-PHASE4-047.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_exp4110_twelfth_game_explore_first as arc4110
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines
from carnot.agentic.arc_exp4110_twelfth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    SelectedCandidate,
    Tu93Action,
    Tu93ObservedState,
    Tu93Outcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_tu93_l1_plan,
    compute_actions_vs_baseline,
    observe_tu93_state_from_env,
    select_exp4110_candidate_from_survey,
    validate_tu93_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4110_twelfth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _state(*, level_completed: int = 0, at_target: bool = False) -> Tu93ObservedState:
    pixels = [[0 for _ in range(13)] for _ in range(7)]
    pixels[0][3] = 2
    pixels[0][9] = 2
    pixels[3][12] = 2
    return Tu93ObservedState(
        player_position=(15, 9) if at_target else (3, 3),
        target_position=(15, 9),
        map_origin=(3, 3),
        map_pixels=tuple(tuple(row) for row in pixels),
        remaining_steps=50,
        level_completed=level_completed,
    )


def _candidate() -> SelectedCandidate:
    return SelectedCandidate(
        game="tu93",
        game_id="tu93-0768757b",
        baseline_actions=19,
        survey_is_spatial_planning=True,
        win_difficulty="hard",
        selection_mode="fallback_lowest_baseline_grid_markov_after_strict_nonspatial_exhausted",
        selection_reason=(
            "selected fallback: tu93 is the lowest-baseline remaining grid-Markov offline fixture, "
            "L0 baseline_actions=19"
        ),
        excluded_solved_games=(
            "r11l",
            "lp85",
            "sc25",
            "su15",
            "tn36",
            "cd82",
            "dc22",
            "sb26",
            "ft09",
            "s5i5",
        ),
    )


def _outcome(*, solved: bool = True) -> Tu93Outcome:
    plan = build_tu93_l1_plan(_state())
    final_state = _state(level_completed=1 if solved else 0, at_target=solved)
    decision = validate_tu93_replayed_plan(_state(), final_state, plan)
    return Tu93Outcome(
        target_game="tu93-0768757b",
        selected_candidate_reason=_candidate().selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=18 if solved else -1,
        exploration_actions_used=len(plan.exploration_actions),
        induced_mechanic="Observed TU93 lattice movement: accepted directions move the player one node.",
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _state().to_json()},
            {"phase": "explore", "actions": [action.to_json() for action in plan.exploration_actions]},
            {"phase": "induce", "mechanic": "tu93_lattice_navigation_to_target"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "levels_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_047_spec_declares_exp4110_contract() -> None:
    """REQ-PHASE4-047: OpenSpec declares Exp 4110 and required principle fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-047" in spec
    assert "SCENARIO-PHASE4-047" in spec
    assert "experiment_4110_twelfth_game_explore_first.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "tu93-0768757b" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_games_solved", "levels_completed", "real_env_confirmed"):
        assert field in spec


def test_req_phase4_047_selects_tu93_after_strict_nonspatial_exhaustion() -> None:
    """REQ-PHASE4-047: selection chooses the lowest-baseline remaining offline fixture."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_exp4110_candidate_from_survey(survey, baselines)

    assert selected.game == "tu93"
    assert selected.game_id == "tu93-0768757b"
    assert selected.baseline_actions == 19
    assert selected.selection_mode == "fallback_lowest_baseline_grid_markov_after_strict_nonspatial_exhausted"
    assert selected.survey_is_spatial_planning is True
    assert "vc33" not in selected.excluded_solved_games
    assert "s5i5" in selected.excluded_solved_games

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
    strict_selected = select_exp4110_candidate_from_survey(
        strict_survey,
        {"aa01": ("aa01-game", 8), "bb02": ("bb02-game", 4)},
        solved_prefixes=(),
    )
    assert strict_selected.game == "bb02"
    assert strict_selected.selection_mode == "strict_survey_non_spatial"

    with pytest.raises(ValueError, match="no unsolved survey candidates"):
        select_exp4110_candidate_from_survey({"per_game_surveys": []}, {}, solved_prefixes=())


def test_req_phase4_047_tu93_plan_explores_then_validates_commit_suffix() -> None:
    """REQ-PHASE4-047: TU93 induction emits exploration before held-out validation."""

    plan = build_tu93_l1_plan(_state())

    assert [action.action for action in plan.actions] == [4, 4, 2]
    assert [action.direction for action in plan.actions] == ["right", "right", "down"]
    assert len(plan.exploration_actions) == 2
    assert plan.commit_actions == plan.actions[2:]
    assert plan.predicted_player_position == (15, 9)
    assert plan.predicted_goal_after_commit is True
    assert plan.induction_call["goal_predicate"] == "player sprite top-left equals the visible target top-left"

    decision = validate_tu93_replayed_plan(_state(), _state(level_completed=1, at_target=True), plan)
    assert decision["retained"] is True
    assert decision["heldout_transition_count"] == 1
    assert decision["final_player_at_target"] is True
    assert decision["level_increment"] is True

    rejected = validate_tu93_replayed_plan(_state(), _state(level_completed=0, at_target=True), plan)
    assert rejected["retained"] is False
    assert rejected["energy"] > 0.0

    mismatched = validate_tu93_replayed_plan(_state(), _state(level_completed=1, at_target=False), plan)
    assert mismatched["retained"] is False
    assert mismatched["final_player_at_target"] is False

    blocked = _state()
    blocked_pixels = tuple(tuple(0 for _ in row) for row in blocked.map_pixels)
    with pytest.raises(ValueError, match="no TU93 path"):
        build_tu93_l1_plan(
            Tu93ObservedState(
                player_position=blocked.player_position,
                target_position=blocked.target_position,
                map_origin=blocked.map_origin,
                map_pixels=blocked_pixels,
                remaining_steps=50,
                level_completed=0,
            )
        )
    with pytest.raises(ValueError, match="already at target"):
        build_tu93_l1_plan(_state(at_target=True))
    assert arc4110._valid_move(_state(), (12, 0), 4) is None
    edge_pixels = [[0 for _ in range(10)]]
    edge_pixels[0][9] = 2
    edge_state = Tu93ObservedState(
        player_position=(3, 3),
        target_position=(9, 3),
        map_origin=(3, 3),
        map_pixels=tuple(tuple(row) for row in edge_pixels),
        remaining_steps=50,
        level_completed=0,
    )
    assert arc4110._valid_move(edge_state, (6, 0), 4) is None


def test_req_phase4_047_observes_tu93_engine_state() -> None:
    """REQ-PHASE4-047: observed state derives player, target, and lattice map."""

    class Sprite:
        def __init__(self, name: str, x: int, y: int, pixels: list[list[int]]) -> None:
            self.name = name
            self.x = x
            self.y = y
            self.pixels = pixels

    player = Sprite("player", 3, 3, [[9]])
    target = Sprite("target", 15, 9, [[1]])
    grid = Sprite("grid", 3, 3, [list(row) for row in _state().map_pixels])

    class Level:
        def get_sprites_by_tag(self, tag: str) -> list[Sprite]:
            return {
                "0017unajnymcki": [player],
                "0015msvpvzxhqf": [target],
                "0005uvnhiglpvh": [grid],
            }.get(tag, [])

    env = type(
        "Env",
        (),
        {
            "_game": type(
                "Game",
                (),
                {
                    "current_level": Level(),
                    "ksulgrfyqx": type("Steps", (), {"current_steps": 49})(),
                },
            )()
        },
    )()

    observed = observe_tu93_state_from_env(env, level_completed=0)

    assert observed.player_position == (3, 3)
    assert observed.target_position == (15, 9)
    assert observed.map_origin == (3, 3)
    assert observed.remaining_steps == 49
    assert observed.to_json()["map_size"] == [7, 13]

    empty_env = type("Env", (), {"_game": type("Game", (), {"current_level": Level()})()})()
    player_only = Level()
    player_only.get_sprites_by_tag = lambda tag: [player] if tag == "0017unajnymcki" else []
    empty_env._game.current_level = player_only
    with pytest.raises(ValueError, match="requires player, target, and map"):
        observe_tu93_state_from_env(empty_env, level_completed=0)


def test_scenario_phase4_047_artifact_has_required_success_fields() -> None:
    """SCENARIO-PHASE4-047: success artifact reports the monotonic 11->12 increment."""

    artifact = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4110,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: twelfth_game_solved_tu93-0768757b_at_action_18"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == "tu93-0768757b"
    assert artifact["total_games_solved"] == 12
    assert artifact["levels_completed"] == 1
    assert artifact["first_solve_at_action"] == 18
    assert artifact["actions_vs_baseline"] == 0.9474
    assert artifact["real_env_confirmed"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["experiment"] == "experiment_4110_twelfth_game_explore_first"
    assert artifact["candidate_baseline_actions"] == 19
    assert artifact["requirements"] == ["REQ-PHASE4-047", "SCENARIO-PHASE4-047"]
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_phase4_047_no_solve_blocked_and_schema_validation(monkeypatch) -> None:
    """REQ-PHASE4-047: no-solve and blocked artifacts do not inflate the count."""

    no_solve = build_artifact(
        _outcome(solved=False),
        _candidate(),
        random_seed=4110,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: twelfth_game_no_solve_tu93-0768757b_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 11
    assert no_solve["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(
        target_game="tu93-0768757b",
        random_seed=4110,
        duration_s=0.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 11
    assert blocked["levels_completed"] == 0
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    assert Tu93Action(4, "right").to_json() == {"action": 4, "direction": "right", "role": "move_player"}
    assert compute_actions_vs_baseline(18, 19, solved=True) == 0.9474
    assert compute_actions_vs_baseline(-1, 19, solved=False) == 0.0
    with pytest.raises(ValueError, match="baseline_actions"):
        compute_actions_vs_baseline(18, 0, solved=True)
    with pytest.raises(ValueError, match="first_solve_at_action"):
        compute_actions_vs_baseline(0, 19, solved=True)

    assert any("missing required field levels_completed" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4110}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("levels_completed must be a bare int" in err for err in artifact_schema_errors({"levels_completed": "1"}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4110}))
    assert any(
        "actions_vs_baseline must be a bare float" in err
        for err in artifact_schema_errors({"actions_vs_baseline": "0.9474"})
    )

    bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4110,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    bad.update(
        {
            "game_solved": "yes",
            "target_game": "",
            "total_games_solved": "12",
            "levels_completed": 0,
            "first_solve_at_action": 0.0,
            "actions_vs_baseline": 0.0,
            "real_env_confirmed": "true",
            "inference_substrate": "wrong",
            "requirements": [],
            "solve_trace": {},
        }
    )
    bad_errors = artifact_schema_errors(bad)
    assert any("game_solved must be a bare bool" in err for err in bad_errors)
    assert any("target_game must name" in err for err in bad_errors)
    assert any("total_games_solved must be a bare int" in err for err in bad_errors)
    assert any("first_solve_at_action must be a bare int" in err for err in bad_errors)
    assert any("actions_vs_baseline must be positive" in err for err in bad_errors)
    assert any("real_env_confirmed must be a bare bool" in err for err in bad_errors)
    assert any("inference_substrate must equal" in err for err in bad_errors)
    assert any("requirements must include" in err for err in bad_errors)
    assert any("solve_trace must include" in err for err in bad_errors)

    monkeypatch.setattr(arc4110, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        arc4110.build_artifact(
            _outcome(),
            _candidate(),
            random_seed=4110,
            duration_s=1.0,
            inference_substrate=INFERENCE_SUBSTRATE,
        )
    with pytest.raises(ValueError, match="forced schema error"):
        arc4110.blocked_artifact(
            target_game="tu93-0768757b",
            random_seed=4110,
            duration_s=0.0,
            inference_substrate=INFERENCE_SUBSTRATE,
        )


def test_scenario_phase4_047_script_writes_success_from_confirmed_outcome(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-047: runner writes success only from confirmed offline evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "tu93" / "0768757b"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "tu93-0768757b", "baseline_actions": [19]}),
        encoding="utf-8",
    )
    metadata_dir.joinpath("tu93.py").write_text("# synthetic offline fixture\n", encoding="utf-8")
    for prefix, baseline in {
        "bp35": 21,
        "ls20": 22,
    }.items():
        root = tmp_path / "environment_files" / prefix / "fixture"
        root.mkdir(parents=True)
        root.joinpath("metadata.json").write_text(
            json.dumps({"game_id": f"{prefix}-fixture", "baseline_actions": [baseline]}),
            encoding="utf-8",
        )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_tu93_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: twelfth_game_solved_tu93-0768757b_at_action_18"
    assert artifact["actions_vs_baseline"] == 0.9474
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == (
        "experiment_4110_twelfth_game_explore_first"
    )


def test_scenario_phase4_047_script_blocks_when_fixture_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-047: missing offline fixture stops with the required blocked verdict."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "tu93" / "0768757b"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "tu93-0768757b", "baseline_actions": [19]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact["target_game"] == "tu93-0768757b"
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["total_games_solved"] == 11
