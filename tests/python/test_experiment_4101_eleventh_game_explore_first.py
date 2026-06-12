"""Tests for Exp 4101 ARC-AGI-3 eleventh-game explore-first solve.

Spec refs: REQ-PHASE4-046, SCENARIO-PHASE4-046.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_exp4101_eleventh_game_explore_first as arc4101
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines
from carnot.agentic.arc_exp4101_eleventh_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    S5I5Action,
    S5I5Item,
    S5I5ObservedState,
    S5I5Outcome,
    SelectedCandidate,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_s5i5_l1_plan,
    compute_actions_vs_baseline,
    observe_s5i5_state_from_env,
    select_exp4101_candidate_from_survey,
    validate_s5i5_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4101_eleventh_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _state(*, level_completed: int = 0, at_target: bool = False) -> S5I5ObservedState:
    horizontal_current = (51, 9) if at_target else (30, 9)
    vertical_current = (9, 51) if at_target else (9, 33)
    return S5I5ObservedState(
        items=(
            S5I5Item(
                item_index=0,
                placeholder_name="horizontal",
                current_position=horizontal_current,
                target_position=(51, 9),
                control_name="horizontal-control",
                control_point=(48, 21),
                step_delta=(3, 0),
                clicks_needed=0 if at_target else 7,
            ),
            S5I5Item(
                item_index=1,
                placeholder_name="vertical",
                current_position=vertical_current,
                target_position=(9, 51),
                control_name="vertical-control",
                control_point=(24, 47),
                step_delta=(0, 3),
                clicks_needed=0 if at_target else 6,
            ),
        ),
        level_completed=level_completed,
    )


def _candidate() -> SelectedCandidate:
    return SelectedCandidate(
        game="s5i5",
        game_id="s5i5-18d95033",
        baseline_actions=20,
        survey_is_spatial_planning=True,
        win_difficulty="hard",
        selection_mode="fallback_click_only_lowest_baseline_after_strict_nonspatial_exhausted",
        selection_reason="selected fallback: s5i5 is the lowest-baseline unsolved click-only fixture, L0 baseline_actions=20",
        excluded_solved_games=("r11l", "lp85", "sc25", "su15", "tn36", "cd82", "dc22", "sb26", "ft09"),
    )


class _Sprite:
    def __init__(self, name: str, x: int, y: int, width: int, height: int) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __hash__(self) -> int:
        return id(self)


def _outcome(*, solved: bool = True) -> S5I5Outcome:
    plan = build_s5i5_l1_plan(_state())
    final_state = _state(level_completed=1 if solved else 0, at_target=solved)
    decision = validate_s5i5_replayed_plan(_state(), final_state, plan)
    return S5I5Outcome(
        target_game="s5i5-18d95033",
        selected_candidate_reason=_candidate().selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=13 if solved else -1,
        exploration_actions_used=len(plan.exploration_actions),
        induced_mechanic="Observed s5i5 resize-control clicks move linked placeholders by one tile toward targets.",
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _state().to_json()},
            {"phase": "explore", "actions": [action.to_json() for action in plan.exploration_actions]},
            {"phase": "induce", "mechanic": "s5i5_resize_linked_placeholders"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "levels_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_046_spec_declares_exp4101_contract() -> None:
    """REQ-PHASE4-046: OpenSpec declares Exp 4101 and required principle fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-046" in spec
    assert "SCENARIO-PHASE4-046" in spec
    assert "experiment_4101_eleventh_game_explore_first.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "s5i5-18d95033" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_games_solved", "levels_completed", "real_env_confirmed"):
        assert field in spec


def test_req_phase4_046_selects_s5i5_after_strict_nonspatial_exhaustion() -> None:
    """REQ-PHASE4-046: selection chooses the lowest-baseline unsolved click-only fallback."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_exp4101_candidate_from_survey(survey, baselines)

    assert selected.game == "s5i5"
    assert selected.game_id == "s5i5-18d95033"
    assert selected.baseline_actions == 20
    assert selected.selection_mode == "fallback_click_only_lowest_baseline_after_strict_nonspatial_exhausted"
    assert selected.survey_is_spatial_planning is True
    assert "vc33" not in selected.excluded_solved_games
    assert "ft09" in selected.excluded_solved_games

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
    strict_selected = select_exp4101_candidate_from_survey(
        strict_survey,
        {"aa01": ("aa01-game", 8), "bb02": ("bb02-game", 4)},
        solved_prefixes=(),
    )
    assert strict_selected.game == "bb02"
    assert strict_selected.selection_mode == "strict_survey_non_spatial"

    no_target_survey = {
        "per_game_surveys": [
            {
                "game": "cc03",
                "is_spatial_planning": True,
                "available_actions": "click-only",
                "win_condition_summary": "match colors",
            }
        ]
    }
    with pytest.raises(ValueError, match="no unsolved strict non-spatial"):
        select_exp4101_candidate_from_survey(no_target_survey, {"cc03": ("cc03-game", 2)}, solved_prefixes=())
    with pytest.raises(ValueError, match="no unsolved strict non-spatial"):
        select_exp4101_candidate_from_survey({"per_game_surveys": []}, {}, solved_prefixes=())


def test_req_phase4_046_s5i5_plan_explores_then_validates_commit_suffix() -> None:
    """REQ-PHASE4-046: s5i5 induction emits exploration before held-out validation."""

    plan = build_s5i5_l1_plan(_state())

    assert len(plan.actions) == 13
    assert len(plan.exploration_actions) == 2
    assert len(plan.commit_actions) == 11
    assert [action.point for action in plan.exploration_actions] == [(48, 21), (24, 47)]
    assert plan.actions.count(S5I5Action.click((48, 21), control_name="horizontal-control", item_index=0)) == 7
    assert plan.actions.count(S5I5Action.click((24, 47), control_name="vertical-control", item_index=1)) == 6
    assert plan.predicted_goal_after_commit is True
    assert plan.predicted_positions == {0: (51, 9), 1: (9, 51)}
    assert plan.induction_call["mechanic"] == (
        "clicking the high side of each observed resize control moves its linked placeholder by one tile"
    )

    decision = validate_s5i5_replayed_plan(_state(), _state(level_completed=1, at_target=True), plan)
    assert decision["retained"] is True
    assert decision["heldout_transition_count"] == 11
    assert decision["level_increment"] is True
    assert decision["predicted_goal_after_actions"] is True

    rejected = validate_s5i5_replayed_plan(_state(), _state(level_completed=0, at_target=True), plan)
    assert rejected["retained"] is False
    assert rejected["energy"] > 0.0

    mismatched = validate_s5i5_replayed_plan(_state(), _state(level_completed=1, at_target=False), plan)
    assert mismatched["retained"] is False
    assert mismatched["final_targets_satisfied"] is False

    with pytest.raises(ValueError, match="at least one"):
        build_s5i5_l1_plan(S5I5ObservedState(items=(), level_completed=0))

    negative_click_state = S5I5ObservedState(
        items=(
            S5I5Item(
                item_index=0,
                placeholder_name="bad",
                current_position=(0, 0),
                target_position=(0, 0),
                control_name="bad-control",
                control_point=(0, 0),
                step_delta=(0, 0),
                clicks_needed=-1,
            ),
        ),
        level_completed=0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        build_s5i5_l1_plan(negative_click_state)


def test_req_phase4_046_observation_defensive_branches() -> None:
    """REQ-PHASE4-046: malformed observed s5i5 links fail before action."""

    class Level:
        def __init__(self, tags: dict[str, list[_Sprite]]) -> None:
            self.tags = tags

        def get_sprites_by_tag(self, tag: str) -> list[_Sprite]:
            return self.tags.get(tag, [])

    def env_for(
        placeholders: list[_Sprite],
        targets: list[_Sprite],
        *,
        links: dict[_Sprite, _Sprite] | None = None,
    ) -> object:
        pigtralzpb: dict[_Sprite, list[_Sprite]] = {}
        uricqfoplr: dict[_Sprite, set[_Sprite]] = {}
        for index, (control, placeholder) in enumerate((links or {}).items()):
            movable = _Sprite(f"movable-{index}", placeholder.x, placeholder.y, placeholder.width, placeholder.height)
            pigtralzpb[control] = [movable]
            uricqfoplr[movable] = {placeholder}
        return type(
            "Env",
            (),
            {
                "_game": type(
                    "Game",
                    (),
                    {
                        "current_level": Level(
                            {
                                "0064ocqkuqacti": placeholders,
                                "0087vvmblxkzdi": targets,
                            }
                        ),
                        "pigtralzpb": pigtralzpb,
                        "uricqfoplr": uricqfoplr,
                    },
                )()
            },
        )()

    with pytest.raises(ValueError, match="requires placeholders and targets"):
        observe_s5i5_state_from_env(env_for([], []), level_completed=0)

    lonely = _Sprite("lonely", 0, 0, 3, 3)
    target = _Sprite("target", 3, 0, 3, 3)
    with pytest.raises(ValueError, match="no controlled placeholders"):
        observe_s5i5_state_from_env(env_for([lonely], [target]), level_completed=0)

    uncontrolled = _Sprite("uncontrolled", 0, 9, 3, 3)
    controlled = _Sprite("controlled", 0, 0, 3, 3)
    control = _Sprite("control", 0, 0, 9, 3)
    observed = observe_s5i5_state_from_env(
        env_for(
            [uncontrolled, controlled],
            [_Sprite("ignored-target", 3, 9, 3, 3), target],
            links={control: controlled},
        ),
        level_completed=0,
    )
    assert len(observed.items) == 1

    diagonal = _Sprite("diagonal", 0, 0, 3, 3)
    with pytest.raises(ValueError, match="diagonal movement"):
        observe_s5i5_state_from_env(
            env_for([diagonal], [_Sprite("diagonal-target", 3, 3, 3, 3)], links={control: diagonal}),
            level_completed=0,
        )

    unaligned = _Sprite("unaligned", 0, 0, 2, 2)
    with pytest.raises(ValueError, match="not step-aligned"):
        observe_s5i5_state_from_env(
            env_for([unaligned], [_Sprite("unaligned-target", 3, 0, 2, 2)], links={control: unaligned}),
            level_completed=0,
        )

    used_target = _Sprite("used", 3, 0, 3, 3)
    with pytest.raises(ValueError, match="no target available"):
        arc4101._nearest_axis_target((0, 0), [used_target], {used_target})


def test_req_phase4_046_observes_s5i5_engine_links() -> None:
    """REQ-PHASE4-046: observed state is derived from control-placeholder-target links."""

    class Sprite:
        def __init__(self, name: str, x: int, y: int, width: int, height: int) -> None:
            self.name = name
            self.x = x
            self.y = y
            self.width = width
            self.height = height

        def __hash__(self) -> int:
            return id(self)

    horizontal_control = Sprite("horizontal-control", 36, 18, 13, 7)
    vertical_control = Sprite("vertical-control", 21, 35, 7, 13)
    horizontal_movable = Sprite("horizontal-movable", 27, 9, 6, 3)
    vertical_movable = Sprite("vertical-movable", 9, 27, 3, 9)
    horizontal_placeholder = Sprite("horizontal", 30, 9, 3, 3)
    vertical_placeholder = Sprite("vertical", 9, 33, 3, 3)
    horizontal_target = Sprite("horizontal-target", 51, 9, 3, 3)
    vertical_target = Sprite("vertical-target", 9, 51, 3, 3)

    class Level:
        def get_sprites_by_tag(self, tag: str) -> list[Sprite]:
            return {
                "0064ocqkuqacti": [vertical_placeholder, horizontal_placeholder],
                "0087vvmblxkzdi": [vertical_target, horizontal_target],
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
                    "pigtralzpb": {
                        horizontal_control: [horizontal_movable],
                        vertical_control: [vertical_movable],
                    },
                    "uricqfoplr": {
                        horizontal_movable: {horizontal_placeholder},
                        vertical_movable: {vertical_placeholder},
                    },
                },
            )()
        },
    )()

    observed = observe_s5i5_state_from_env(env, level_completed=0)

    assert observed.level_completed == 0
    assert [item.to_json() for item in observed.items] == [
        {
            "clicks_needed": 7,
            "control_name": "horizontal-control",
            "control_point": [48, 21],
            "current_position": [30, 9],
            "item_index": 0,
            "placeholder_name": "horizontal",
            "step_delta": [3, 0],
            "target_position": [51, 9],
        },
        {
            "clicks_needed": 6,
            "control_name": "vertical-control",
            "control_point": [24, 47],
            "current_position": [9, 33],
            "item_index": 1,
            "placeholder_name": "vertical",
            "step_delta": [0, 3],
            "target_position": [9, 51],
        },
    ]


def test_scenario_phase4_046_artifact_has_required_success_fields() -> None:
    """SCENARIO-PHASE4-046: success artifact reports the monotonic 10->11 increment."""

    artifact = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4101,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: eleventh_game_solved_s5i5-18d95033_at_action_13"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == "s5i5-18d95033"
    assert artifact["total_games_solved"] == 11
    assert artifact["levels_completed"] == 1
    assert artifact["first_solve_at_action"] == 13
    assert artifact["actions_vs_baseline"] == 0.65
    assert artifact["real_env_confirmed"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["experiment"] == "experiment_4101_eleventh_game_explore_first"
    assert artifact["candidate_baseline_actions"] == 20
    assert artifact["requirements"] == ["REQ-PHASE4-046", "SCENARIO-PHASE4-046"]
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_phase4_046_no_solve_blocked_and_schema_validation() -> None:
    """REQ-PHASE4-046: no-solve and blocked artifacts do not inflate the count."""

    no_solve = build_artifact(
        _outcome(solved=False),
        _candidate(),
        random_seed=4101,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: eleventh_game_no_solve_s5i5-18d95033_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 10
    assert no_solve["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(
        target_game="s5i5-18d95033",
        random_seed=4101,
        duration_s=0.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 10
    assert blocked["levels_completed"] == 0
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    assert compute_actions_vs_baseline(13, 20, solved=True) == 0.65
    assert compute_actions_vs_baseline(-1, 20, solved=False) == 0.0
    with pytest.raises(ValueError, match="baseline_actions"):
        compute_actions_vs_baseline(13, 0, solved=True)
    with pytest.raises(ValueError, match="first_solve_at_action"):
        compute_actions_vs_baseline(0, 20, solved=True)

    errors = artifact_schema_errors({})
    assert any("missing required field levels_completed" in err for err in errors)
    assert any("missing required field first_solve_at_action" in err for err in errors)
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4101}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("levels_completed must be a bare int" in err for err in artifact_schema_errors({"levels_completed": "1"}))
    assert any(
        "actions_vs_baseline must be a bare float" in err
        for err in artifact_schema_errors({"actions_vs_baseline": "0.65"})
    )

    bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4101,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    bad.update(
        {
            "game_solved": "yes",
            "target_game": 4101,
            "total_games_solved": "11",
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
    assert any("target_game must be a string" in err for err in bad_errors)
    assert any("total_games_solved must be a bare int" in err for err in bad_errors)
    assert any("first_solve_at_action must be a bare int" in err for err in bad_errors)
    assert any("real_env_confirmed must be a bare bool" in err for err in bad_errors)
    assert any("inference_substrate must equal" in err for err in bad_errors)
    assert any("requirements must include" in err for err in bad_errors)
    assert any("levels_completed must increment" in err for err in bad_errors)
    assert any("actions_vs_baseline must be positive" in err for err in bad_errors)
    assert any("solve_trace must include" in err for err in bad_errors)

    success_with_blank_target = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4101,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    success_with_blank_target["target_game"] = ""
    assert any("target_game must name" in err for err in artifact_schema_errors(success_with_blank_target))


def test_req_phase4_046_artifact_builders_raise_on_internal_schema_errors(monkeypatch) -> None:
    """REQ-PHASE4-046: artifact constructors fail closed when schema validation fails."""

    monkeypatch.setattr(arc4101, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        arc4101.build_artifact(
            _outcome(),
            _candidate(),
            random_seed=4101,
            duration_s=1.0,
            inference_substrate=INFERENCE_SUBSTRATE,
        )
    with pytest.raises(ValueError, match="forced schema error"):
        arc4101.blocked_artifact(
            target_game="s5i5-18d95033",
            random_seed=4101,
            duration_s=0.0,
            inference_substrate=INFERENCE_SUBSTRATE,
        )


def test_scenario_phase4_046_script_writes_success_from_confirmed_outcome(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-046: runner writes success only from confirmed offline evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "s5i5" / "18d95033"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "s5i5-18d95033", "baseline_actions": [20]}),
        encoding="utf-8",
    )
    metadata_dir.joinpath("s5i5.py").write_text("# synthetic offline fixture\n", encoding="utf-8")
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_s5i5_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: eleventh_game_solved_s5i5-18d95033_at_action_13"
    assert artifact["actions_vs_baseline"] == 0.65
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == (
        "experiment_4101_eleventh_game_explore_first"
    )


def test_scenario_phase4_046_script_blocks_when_fixture_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-046: missing offline fixture stops with the required blocked verdict."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "s5i5" / "18d95033"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "s5i5-18d95033", "baseline_actions": [20]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    assert artifact["target_game"] == "s5i5-18d95033"
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["total_games_solved"] == 10
