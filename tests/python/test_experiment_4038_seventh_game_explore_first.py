"""Tests for Exp 4038 ARC-AGI-3 seventh-game explore-first solve.

Spec refs: REQ-PHASE4-038, SCENARIO-PHASE4-038.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_exp4038_seventh_game_explore_first as helper
from carnot.agentic.arc_exp4038_seventh_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    Dc22Action,
    Dc22State,
    ExperimentOutcome,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    dc22_default_exploration_actions,
    load_environment_baselines,
    select_seventh_candidate_from_survey,
    validate_replayed_plan,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4038_seventh_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _outcome(*, solved: bool = True) -> ExperimentOutcome:
    actions = [
        *dc22_default_exploration_actions(),
        Dc22Action.key(1),
        Dc22Action.key(1),
        Dc22Action.click(48, 36, sprite="buezna-blrmbx", grid=(48, 26)),
    ]
    return ExperimentOutcome(
        target_game="dc22-fdcac232",
        selected_candidate_reason="selected: dc22 is unsolved and survey non-spatial",
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=20 if solved else -1,
        exploration_actions_used=2,
        induced_mechanic=(
            "Observed ACTION1 moves jfva upward and buezna clicks toggle bridge blockers; "
            "goal predicate is jfva position equals goknoi position."
        ),
        verification_decisions=[
            {
                "phase": "verify",
                "retained": solved,
                "predicted_goal_after_actions": solved,
                "energy": 0.0 if solved else 10.0,
            }
        ],
        phase_trace=[
            {"phase": "observe", "level_completed": 0},
            {"phase": "explore", "action": {"action": 1}, "level_completed": 0},
            {"phase": "induce", "mechanic": "dc22_navigation_toggle_goal"},
            {"phase": "verify", "retained": solved},
            {"phase": "act", "level_completed": 1 if solved else 0},
        ],
        real_env_confirmed=solved,
        action_plan=actions,
        arc_env_count=25,
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_038_spec_declares_exp4038_contract() -> None:
    """REQ-PHASE4-038: OpenSpec declares Exp 4038 and the mandatory artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-038" in spec
    assert "SCENARIO-PHASE4-038" in spec
    assert "experiment_4038_seventh_game_explore_first.json" in spec
    assert "blocked_arc_env_unreachable" in spec
    assert "dc22-fdcac232" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_038_selects_dc22_from_remaining_non_spatial_survey() -> None:
    """REQ-PHASE4-038: target selection excludes six solved games and avoids vc33."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_seventh_candidate_from_survey(survey, baselines)

    assert selected.game == "dc22"
    assert selected.game_id == "dc22-fdcac232"
    assert selected.baseline_actions == 59
    assert selected.non_spatial is True
    assert "cd82" in selected.excluded_solved_games
    assert "vc33" not in selected.excluded_solved_games
    assert selected.selection_reason.startswith("selected: dc22")


def test_req_phase4_038_selection_and_baseline_defensive_paths(tmp_path: Path) -> None:
    """REQ-PHASE4-038: malformed metadata and no-candidate surveys fail closed."""

    empty_metadata_dir = tmp_path / "environment_files" / "zz99" / "abc"
    empty_metadata_dir.mkdir(parents=True)
    empty_metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "zz99-abc", "baseline_actions": []}),
        encoding="utf-8",
    )

    assert load_environment_baselines(tmp_path / "environment_files") == {}

    survey = {
        "per_game_surveys": [
            {"game": "vc33", "is_spatial_planning": False},
            {"game": "dc22", "is_spatial_planning": True},
            {"game": "xx01", "is_spatial_planning": False},
        ]
    }
    with pytest.raises(ValueError, match="no unsolved non-spatial"):
        select_seventh_candidate_from_survey(survey, {})


def test_req_phase4_038_default_exploration_is_positive_before_commit() -> None:
    """REQ-PHASE4-038: the plan spends real exploration actions before verification."""

    actions = dc22_default_exploration_actions()

    assert actions == [
        Dc22Action.key(1),
        Dc22Action.click(48, 36, sprite="buezna-blrmbx", grid=(48, 26)),
    ]
    assert [row.to_json() for row in actions] == [
        {"action": 1},
        {"action": 6, "x": 48, "y": 36, "sprite": "buezna-blrmbx", "grid": [48, 26]},
    ]


def test_scenario_phase4_038_gap4_replay_verifier_requires_level_increment() -> None:
    """SCENARIO-PHASE4-038: GAP-4 replay validation accepts only a confirmed level-up."""

    start = Dc22State(player=(10, 28), goal=(24, 10), level_completed=0, blocker_signature=("b-open",))
    mid = Dc22State(player=(10, 26), goal=(24, 10), level_completed=0, blocker_signature=("b-open",))
    solved = Dc22State(player=(24, 10), goal=(24, 10), level_completed=1, blocker_signature=("b-open",))
    actions = [Dc22Action.key(1), Dc22Action.key(4)]

    decision = validate_replayed_plan(start, [start, mid, solved], actions, start_level_completed=0)

    assert decision["retained"] is True
    assert decision["predicted_goal_after_actions"] is True
    assert decision["energy"] == 0.0
    assert decision["final_distance"] == 0
    assert start.to_json()["distance_to_goal"] == 32

    rejected = validate_replayed_plan(start, [start, mid, mid], actions, start_level_completed=0)
    assert rejected["retained"] is False
    assert rejected["predicted_goal_after_actions"] is False
    assert rejected["energy"] > 0.0

    with pytest.raises(ValueError, match="one more state"):
        validate_replayed_plan(start, [start], actions, start_level_completed=0)
    with pytest.raises(ValueError, match="start from the supplied"):
        validate_replayed_plan(start, [mid, mid, solved], actions, start_level_completed=0)


def test_req_phase4_038_artifact_schema_success_no_solve_and_blocked() -> None:
    """REQ-PHASE4-038: artifacts preserve bare counters and real-env confirmation."""

    artifact = build_artifact(
        _outcome(),
        random_seed=4038,
        duration_s=1.25,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: seventh_game_solved_dc22-fdcac232_at_action_20"
    assert artifact["game_solved"] is True
    assert artifact["target_game"] == "dc22-fdcac232"
    assert artifact["total_games_solved"] == 7
    assert artifact["real_env_confirmed"] is True
    assert artifact["field_principles"]["total_games_solved"].startswith("monotonic ARC accuracy")
    assert artifact_schema_errors(artifact) == []

    no_solve = build_artifact(
        _outcome(solved=False),
        random_seed=4038,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert no_solve["honest_verdict"] == (
        "complete: seventh_game_no_solve_dc22-fdcac232_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 6
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(random_seed=4038, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["target_game"] == "none"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 6
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    bad = dict(artifact)
    bad["honest_verdict"] = "done"
    bad["game_solved"] = "true"
    bad["target_game"] = 4038
    bad["total_games_solved"] = 7.0
    bad["real_env_confirmed"] = 1
    bad["inference_substrate"] = None
    errors = artifact_schema_errors(bad)

    assert any("honest_verdict" in err for err in errors)
    assert any("game_solved" in err for err in errors)
    assert any("target_game" in err for err in errors)
    assert any("total_games_solved" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)

    missing_errors = artifact_schema_errors({})
    assert any("missing required field honest_verdict" in err for err in missing_errors)

    non_string_verdict = dict(artifact)
    non_string_verdict["honest_verdict"] = 4038
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors(non_string_verdict))

    success_bad = dict(artifact)
    success_bad.update(
        {
            "honest_verdict": "success: seventh_game_solved_none_at_action_0",
            "game_solved": False,
            "target_game": "none",
            "total_games_solved": 6,
            "real_env_confirmed": False,
            "level_completed": 0,
            "first_solve_at_action": 0,
            "exploration_actions_used": 0,
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

    with pytest.raises(ValueError, match="inference_substrate"):
        build_artifact(_outcome(), random_seed=4038, duration_s=0.1, inference_substrate=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference_substrate"):
        blocked_artifact(random_seed=4038, duration_s=0.0, inference_substrate=None)  # type: ignore[arg-type]


def test_scenario_phase4_038_script_writes_success_from_real_env_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-038: the script writes success only from the outcome confirmation."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "environment_files" / "dc22" / "fdcac232"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "dc22-fdcac232", "baseline_actions": [59]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", lambda: 25)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_load_online_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_dc22_explore_first", lambda *args, **kwargs: _outcome())

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: seventh_game_solved_dc22-fdcac232_at_action_20"
    assert artifact["total_games_solved"] == 7
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["target_game"] == "dc22-fdcac232"


def test_scenario_phase4_038_script_blocks_when_arc_env_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-038: live ARC precondition failure stops with blocked verdict."""

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
