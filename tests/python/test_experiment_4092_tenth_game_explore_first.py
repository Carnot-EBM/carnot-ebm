"""Tests for Exp 4092 ARC-AGI-3 tenth-game explore-first solve.

Spec refs: REQ-PHASE4-045, SCENARIO-PHASE4-045.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines
from carnot.agentic.arc_exp4092_tenth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    R11LGroup,
    R11LOutcome,
    R11LPiece,
    R11LObservedState,
    SelectedCandidate,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_r11l_l1_plan,
    compute_actions_vs_baseline,
    observe_r11l_state_from_env,
    select_exp4092_candidate_from_survey,
    validate_r11l_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4092_tenth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _state(*, level_completed: int = 0) -> R11LObservedState:
    return R11LObservedState(
        groups=(
            R11LGroup(
                group_id="pumlzd",
                target_center=(20, 38),
                target_satisfied=False,
                pieces=(
                    R11LPiece(piece_index=0, center=(36, 7)),
                    R11LPiece(piece_index=1, center=(59, 27)),
                ),
            ),
        ),
        level_completed=level_completed,
    )


def _candidate() -> SelectedCandidate:
    return SelectedCandidate(
        game="r11l",
        game_id="r11l-495a7899",
        baseline_actions=22,
        survey_is_spatial_planning=False,
        win_difficulty="medium",
        selection_mode="preferred_consensus_top_pick",
        selection_reason="selected preferred: r11l is the consensus top pick, directly observable, L0 baseline_actions=22",
        excluded_solved_games=("lp85", "sc25", "su15", "tn36", "cd82", "dc22", "sb26", "ft09"),
    )


def _outcome(*, solved: bool = True) -> R11LOutcome:
    plan = build_r11l_l1_plan(_state())
    final_state = R11LObservedState(groups=_state().groups, level_completed=1 if solved else 0)
    decision = validate_r11l_replayed_plan(_state(), final_state, plan)
    return R11LOutcome(
        target_game="r11l-495a7899",
        selected_candidate_reason=_candidate().selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=4 if solved else -1,
        exploration_actions_used=len(plan.exploration_actions),
        induced_mechanic="Observed r11l click-select then click-place sprite movement.",
        verification_decisions=[decision],
        phase_trace=[
            {"phase": "observe", "state": _state().to_json()},
            {"phase": "explore", "actions": [action.to_json() for action in plan.exploration_actions]},
            {"phase": "induce", "mechanic": "r11l_click_select_place"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "level_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=plan.actions,
        arc_env_count=25,
        induction_calls=[plan.induction_call],
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_045_spec_declares_exp4092_contract() -> None:
    """REQ-PHASE4-045: OpenSpec declares Exp 4092 and all required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-045" in spec
    assert "SCENARIO-PHASE4-045" in spec
    assert "experiment_4092_tenth_game_explore_first.json" in spec
    assert "blocked_arc_agi3_unreachable" in spec
    assert "r11l-495a7899" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_045_selects_r11l_consensus_top_pick() -> None:
    """REQ-PHASE4-045: selection chooses R11L and avoids vc33."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_exp4092_candidate_from_survey(survey, baselines)

    assert selected.game == "r11l"
    assert selected.game_id == "r11l-495a7899"
    assert selected.baseline_actions == 22
    assert selected.selection_mode == "preferred_consensus_top_pick"
    assert selected.survey_is_spatial_planning is False
    assert "vc33" not in selected.excluded_solved_games
    assert "ft09" in selected.excluded_solved_games


def test_req_phase4_045_r11l_plan_explores_then_validates_commit_suffix() -> None:
    """REQ-PHASE4-045: r11l induction emits exploration before the held-out commit suffix."""

    plan = build_r11l_l1_plan(_state())

    assert len(plan.exploration_actions) == 2
    assert plan.commit_actions == plan.actions[2:]
    assert plan.predicted_goal_after_commit is True
    assert [action.to_json() for action in plan.actions] == [
        {
            "action": 6,
            "x": 7,
            "y": 36,
            "point": [36, 7],
            "role": "select_piece",
            "group_id": "pumlzd",
            "piece_index": 0,
        },
        {
            "action": 6,
            "x": 34,
            "y": 20,
            "point": [20, 34],
            "role": "place_piece",
            "group_id": "pumlzd",
            "piece_index": 0,
        },
        {
            "action": 6,
            "x": 27,
            "y": 59,
            "point": [59, 27],
            "role": "select_piece",
            "group_id": "pumlzd",
            "piece_index": 1,
        },
        {
            "action": 6,
            "x": 42,
            "y": 20,
            "point": [20, 42],
            "role": "place_piece",
            "group_id": "pumlzd",
            "piece_index": 1,
        },
    ]
    decision = validate_r11l_replayed_plan(_state(), R11LObservedState(groups=_state().groups, level_completed=1), plan)
    assert decision["retained"] is True
    assert decision["heldout_transition_count"] == 2

    rejected = validate_r11l_replayed_plan(_state(), R11LObservedState(groups=_state().groups, level_completed=0), plan)
    assert rejected["retained"] is False
    assert rejected["energy"] > 0.0

    with pytest.raises(ValueError, match="at least one r11l"):
        build_r11l_l1_plan(R11LObservedState(groups=(), level_completed=0))

    with pytest.raises(ValueError, match="has no observed pieces"):
        build_r11l_l1_plan(
            R11LObservedState(
                groups=(R11LGroup(group_id="empty", target_center=(1, 1), target_satisfied=False, pieces=()),),
                level_completed=0,
            )
        )

    single_piece_plan = build_r11l_l1_plan(
        R11LObservedState(
            groups=(
                R11LGroup(
                    group_id="single",
                    target_center=(10, 10),
                    target_satisfied=False,
                    pieces=(R11LPiece(piece_index=0, center=(1, 1)),),
                ),
            ),
            level_completed=0,
        )
    )
    assert single_piece_plan.actions[1].point == (10, 10)

    odd_piece_plan = build_r11l_l1_plan(
        R11LObservedState(
            groups=(
                R11LGroup(
                    group_id="odd",
                    target_center=(10, 10),
                    target_satisfied=False,
                    pieces=(
                        R11LPiece(piece_index=0, center=(1, 1)),
                        R11LPiece(piece_index=1, center=(2, 2)),
                        R11LPiece(piece_index=2, center=(3, 3)),
                    ),
                ),
            ),
            level_completed=0,
        )
    )
    assert odd_piece_plan.actions[-1].point == (10, 10)


def test_req_phase4_045_observes_r11l_engine_groups() -> None:
    """REQ-PHASE4-045: observed state derives the target predicate substrate from groups."""

    class Sprite:
        def __init__(self, name: str, y: int, x: int, height: int = 4, width: int = 4) -> None:
            self.name = name
            self.y = y
            self.x = x
            self.height = height
            self.width = width

        def collides_with(self, other: object) -> bool:
            return bool(getattr(self, "collides", False) and other is not None)

    target = Sprite("flkdtg-pumlzd", 18, 36, height=8, width=8)
    composite = Sprite("roefwu-pumlzd", 18, 36, height=6, width=6)
    piece = Sprite("roefwulewcui-pumlzd", 34, 5, height=4, width=4)
    target_without_composite = Sprite("flkdtg-solo", 40, 50, height=8, width=10)
    env = type(
        "Env",
        (),
        {
            "_game": type(
                "Game",
                (),
                {
                    "kacotwgjcyq": {
                        "missing": {
                            "gosubdcyegamj": None,
                            "roduyfsmiznvg": None,
                            "lecfirgqbwunn": [],
                        },
                        "pumlzd": {
                            "gosubdcyegamj": target,
                            "roduyfsmiznvg": composite,
                            "lecfirgqbwunn": [piece],
                        },
                        "solo": {
                            "gosubdcyegamj": target_without_composite,
                            "roduyfsmiznvg": None,
                            "lecfirgqbwunn": [],
                        },
                    }
                },
            )()
        },
    )()

    observed = observe_r11l_state_from_env(env, level_completed=0)

    assert observed.level_completed == 0
    by_id = {group.group_id: group for group in observed.groups}
    assert by_id["pumlzd"].target_center == (21, 39)
    assert by_id["pumlzd"].pieces[0].center == (36, 7)
    assert by_id["solo"].target_center == (44, 55)


def test_scenario_phase4_045_artifact_has_required_success_fields() -> None:
    """SCENARIO-PHASE4-045: success artifact reports the monotonic 9->10 increment."""

    artifact = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4092,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: tenth_game_solved_r11l-495a7899_at_action_4"
    assert artifact["game_solved"] is True
    assert artifact["total_games_solved"] == 10
    assert artifact["first_solve_at_action"] == 4
    assert artifact["actions_vs_baseline"] == 0.1818
    assert artifact["real_env_confirmed"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["experiment"] == "experiment_4092_tenth_game_explore_first"
    assert artifact["candidate_baseline_actions"] == 22
    assert artifact["requirements"] == ["REQ-PHASE4-045", "SCENARIO-PHASE4-045"]
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_phase4_045_no_solve_blocked_and_schema_validation() -> None:
    """REQ-PHASE4-045: no-solve and blocked artifacts do not inflate the game count."""

    no_solve = build_artifact(
        _outcome(solved=False),
        _candidate(),
        random_seed=4092,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: tenth_game_no_solve_r11l-495a7899_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 9
    assert no_solve["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(random_seed=4092, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    assert blocked["honest_verdict"] == "blocked_arc_agi3_unreachable"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 9
    assert blocked["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(blocked) == []

    assert compute_actions_vs_baseline(4, 22, solved=True) == 0.1818
    assert compute_actions_vs_baseline(-1, 22, solved=False) == 0.0
    with pytest.raises(ValueError, match="baseline_actions"):
        compute_actions_vs_baseline(4, 0, solved=True)
    with pytest.raises(ValueError, match="first_solve_at_action"):
        compute_actions_vs_baseline(0, 22, solved=True)

    errors = artifact_schema_errors({})
    assert any("missing required field actions_vs_baseline" in err for err in errors)
    assert any("missing required field first_solve_at_action" in err for err in errors)

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="r11l consensus"):
        select_exp4092_candidate_from_survey(survey, {})

    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 1}))
    assert any(
        "honest_verdict must start with" in err
        for err in artifact_schema_errors({"honest_verdict": "invalid"})
    )

    success_bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4092,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    success_bad.update(
        {
            "game_solved": "yes",
            "total_games_solved": "10",
            "first_solve_at_action": 0.0,
            "actions_vs_baseline": "0.1818",
            "real_env_confirmed": "true",
            "inference_substrate": "wrong",
            "requirements": [],
            "level_completed": 0,
            "solve_trace": {},
        }
    )
    success_errors = artifact_schema_errors(success_bad)
    assert any("game_solved must be a bare bool" in err for err in success_errors)
    assert any("total_games_solved must be a bare int" in err for err in success_errors)
    assert any("first_solve_at_action must be a bare int" in err for err in success_errors)
    assert any("actions_vs_baseline must be a bare float" in err for err in success_errors)
    assert any("real_env_confirmed must be a bare bool" in err for err in success_errors)
    assert any("inference_substrate must equal" in err for err in success_errors)
    assert any("requirements must include" in err for err in success_errors)
    assert any("game_solved must be true for success" in err for err in success_errors)
    assert any("real_env_confirmed must be true for success" in err for err in success_errors)
    assert any("total_games_solved must increment" in err for err in success_errors)
    assert any("level_completed must increment" in err for err in success_errors)
    assert any("first_solve_at_action must be positive for success" in err for err in success_errors)
    assert any("solve_trace must include" in err for err in success_errors)

    ratio_bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4092,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    ratio_bad["actions_vs_baseline"] = 0.0
    assert any("actions_vs_baseline must be positive" in err for err in artifact_schema_errors(ratio_bad))

    with pytest.raises(ValueError, match="inference_substrate"):
        build_artifact(
            _outcome(),
            _candidate(),
            random_seed=4092,
            duration_s=1.0,
            inference_substrate="wrong",
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        blocked_artifact(random_seed=4092, duration_s=0.0, inference_substrate="wrong")


def test_scenario_phase4_045_script_writes_success_from_confirmed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-045: runner writes success only from confirmed live evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "r11l" / "495a7899"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "r11l-495a7899", "baseline_actions": [22]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", lambda: 25)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_load_online_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_r11l_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: tenth_game_solved_r11l-495a7899_at_action_4"
    assert artifact["actions_vs_baseline"] == 0.1818
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == (
        "experiment_4092_tenth_game_explore_first"
    )


def test_scenario_phase4_045_script_blocks_when_arc_sdk_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-045: live ARC SDK precondition failure stops with blocked verdict."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unreachable() -> int:
        raise RuntimeError("catalog down")

    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", unreachable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_agi3_unreachable"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["actions_vs_baseline"] == 0.0
