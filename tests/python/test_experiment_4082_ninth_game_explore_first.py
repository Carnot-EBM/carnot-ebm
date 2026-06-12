"""Tests for Exp 4082 ARC-AGI-3 ninth-game explore-first retry.

Spec refs: REQ-PHASE4-044, SCENARIO-PHASE4-044.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_exp4070_ninth_game_explore_first import (
    ExperimentOutcome,
    Ft09Action,
    Ft09Cell,
    Ft09Constraint,
    Ft09ObservedState,
    SelectedCandidate,
    build_ft09_l1_plan,
    load_environment_baselines,
    validate_replayed_plan,
)
from carnot.agentic.arc_exp4082_ninth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    compute_actions_vs_baseline,
    select_exp4082_candidate_from_survey,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4082_ninth_game_explore_first as exp  # noqa: E402


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


def _candidate() -> SelectedCandidate:
    return SelectedCandidate(
        game="ft09",
        game_id="ft09-0d8bbf25",
        baseline_actions=43,
        survey_is_spatial_planning=True,
        win_difficulty="hard",
        selection_mode="fallback_click_local_constraint_non_navigation",
        selection_reason="selected fallback: ft09 is unsolved, win_difficulty=hard, L0 baseline_actions=43",
        excluded_solved_games=("r11l", "lp85", "sc25", "su15", "tn36", "cd82", "dc22", "sb26"),
    )


def _outcome(*, solved: bool = True) -> ExperimentOutcome:
    plan = build_ft09_l1_plan(_ft09_l1_state())
    final_state = _ft09_solved_state(level_completed=1 if solved else 0)
    decision = validate_replayed_plan(_ft09_l1_state(), final_state, plan)
    return ExperimentOutcome(
        target_game="ft09-0d8bbf25",
        selected_candidate_reason=_candidate().selection_reason,
        prior_total_games_solved=PRIOR_TOTAL_GAMES_SOLVED,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=4 if solved else -1,
        exploration_actions_used=1,
        induced_mechanic="Observed ft09 click-to-cycle cells and induced the local constraint goal predicate.",
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


def test_req_phase4_044_spec_declares_exp4082_contract() -> None:
    """REQ-PHASE4-044: OpenSpec declares Exp 4082 and its required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-044" in spec
    assert "SCENARIO-PHASE4-044" in spec
    assert "experiment_4082_ninth_game_explore_first.json" in spec
    assert "ft09-0d8bbf25" in spec
    assert "actions_vs_baseline" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_044_selects_ft09_after_strict_nonspatial_exhaustion() -> None:
    """REQ-PHASE4-044: selection avoids vc33 and picks the lowest local-constraint fallback."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_exp4082_candidate_from_survey(survey, baselines)

    assert selected.game == "ft09"
    assert selected.game_id == "ft09-0d8bbf25"
    assert selected.baseline_actions == 43
    assert selected.selection_mode == "fallback_click_local_constraint_non_navigation"
    assert "vc33" not in selected.excluded_solved_games
    assert "sb26" in selected.excluded_solved_games


def test_scenario_phase4_044_artifact_has_required_success_fields() -> None:
    """SCENARIO-PHASE4-044: success artifact reports the monotonic 8->9 increment."""

    artifact = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4082,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )

    assert artifact["honest_verdict"] == "success: ninth_game_solved_ft09-0d8bbf25_at_action_4"
    assert artifact["game_solved"] is True
    assert artifact["total_games_solved"] == 9
    assert artifact["first_solve_at_action"] == 4
    assert artifact["actions_vs_baseline"] == 0.093
    assert artifact["real_env_confirmed"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["experiment"] == "experiment_4082_ninth_game_explore_first"
    assert artifact["candidate_baseline_actions"] == 43
    assert artifact["requirements"] == ["REQ-PHASE4-044", "SCENARIO-PHASE4-044"]
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_phase4_044_no_solve_blocked_and_schema_validation() -> None:
    """REQ-PHASE4-044: no-solve and blocked artifacts stay terminal and schema-valid."""

    no_solve = build_artifact(
        _outcome(solved=False),
        _candidate(),
        random_seed=4082,
        duration_s=0.5,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    assert no_solve["honest_verdict"] == (
        "complete: ninth_game_no_solve_ft09-0d8bbf25_level_counter_did_not_increment"
    )
    assert no_solve["game_solved"] is False
    assert no_solve["total_games_solved"] == 8
    assert no_solve["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(random_seed=4082, duration_s=0.0, inference_substrate=INFERENCE_SUBSTRATE)
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["game_solved"] is False
    assert blocked["total_games_solved"] == 8
    assert blocked["actions_vs_baseline"] == 0.0
    assert artifact_schema_errors(blocked) == []

    assert compute_actions_vs_baseline(4, 43, solved=True) == 0.093
    assert compute_actions_vs_baseline(-1, 43, solved=False) == 0.0
    with pytest.raises(ValueError, match="baseline_actions"):
        compute_actions_vs_baseline(4, 0, solved=True)
    with pytest.raises(ValueError, match="first_solve_at_action"):
        compute_actions_vs_baseline(0, 43, solved=True)

    errors = artifact_schema_errors({})
    assert any("missing required field actions_vs_baseline" in err for err in errors)
    assert any("missing required field first_solve_at_action" in err for err in errors)

    bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4082,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    bad["actions_vs_baseline"] = "0.093"
    bad["first_solve_at_action"] = 4.0
    bad["inference_substrate"] = "wrong"
    bad["requirements"] = []
    bad_errors = artifact_schema_errors(bad)
    assert any("actions_vs_baseline must be a bare float" in err for err in bad_errors)
    assert any("first_solve_at_action must be a bare int" in err for err in bad_errors)
    assert any("inference_substrate must equal" in err for err in bad_errors)
    assert any("requirements must include" in err for err in bad_errors)

    success_bad = build_artifact(
        _outcome(),
        _candidate(),
        random_seed=4082,
        duration_s=1.0,
        inference_substrate=INFERENCE_SUBSTRATE,
    )
    success_bad["actions_vs_baseline"] = 0.0
    success_bad["first_solve_at_action"] = 0
    success_errors = artifact_schema_errors(success_bad)
    assert any("actions_vs_baseline must be positive for success" in err for err in success_errors)
    assert any("first_solve_at_action must be positive for success" in err for err in success_errors)

    with pytest.raises(ValueError, match="inference_substrate"):
        build_artifact(
            _outcome(),
            _candidate(),
            random_seed=4082,
            duration_s=1.0,
            inference_substrate="wrong",
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        blocked_artifact(random_seed=4082, duration_s=0.0, inference_substrate="wrong")


def test_scenario_phase4_044_script_writes_success_from_confirmed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-044: runner writes success only from confirmed outcome evidence."""

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
    assert artifact["actions_vs_baseline"] == 0.093
    assert artifact["real_env_confirmed"] is True
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == (
        "experiment_4082_ninth_game_explore_first"
    )


def test_scenario_phase4_044_script_blocks_when_arc_env_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-044: live ARC precondition failure stops with blocked verdict."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unreachable() -> int:
        raise RuntimeError("catalog down")

    monkeypatch.setattr(exp, "_confirm_arc_env_reachable", unreachable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_env_unreachable"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["actions_vs_baseline"] == 0.0
