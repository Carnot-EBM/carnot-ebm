"""Tests for Exp 4151 ARC-AGI-3 incremental non-spatial progress.

Spec refs: REQ-PHASE4-051, SCENARIO-PHASE4-051.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4151_arc_incremental_progress as exp
from carnot.experiment_4151_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    FirstLevelOutcome,
    SelectedTarget,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_selection_evidence,
    load_environment_baselines,
    select_next_unsolved_nonspatial,
    validate_gap4_heldout_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _target() -> SelectedTarget:
    return SelectedTarget(
        game="zz99",
        game_id="zz99-abcd1234",
        baseline_actions=9,
        survey_rank=0,
        selection_mode="strict_survey_non_spatial",
        selection_reason="selected zz99 as the next unsolved strict non-spatial survey target",
        excluded_solved_games=exp.SOLVED_PREFIXES_BEFORE_4151,
    )


def _outcome(*, solved: bool, target_game: str = "zz99-abcd1234") -> FirstLevelOutcome:
    decision = validate_gap4_heldout_replay(
        start_level=0,
        final_level=1 if solved else 0,
        heldout_transition_count=2,
        predicted_level_after_actions=1,
    )
    return FirstLevelOutcome(
        target_game=target_game,
        final_level_completed=1 if solved else 0,
        first_solve_at_action=5 if solved else -1,
        exploration_actions_used=3,
        real_env_confirmed=solved,
        verifier_validated=solved,
        verification_decisions=[decision],
        action_plan=[{"action": 6, "role": "validated_click"}] if solved else [],
        phase_trace=[
            {"phase": "observe", "target_game": target_game},
            {"phase": "explore", "actions": 3},
            {"phase": "induce", "mechanic": "observed_click_match"},
            decision,
            {"phase": "act", "levels_completed": 1 if solved else 0},
        ],
        induced_mechanic="observed click-to-match first-level mechanic",
        failure_reason="" if solved else "level_counter_did_not_increment",
    )


def test_req_phase4_051_spec_declares_exp4151_contract() -> None:
    """REQ-PHASE4-051: OpenSpec declares the Exp 4151 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-051" in spec
    assert "SCENARIO-PHASE4-051" in spec
    assert "experiment_4151_arc_incremental_progress.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "complete: fifteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_games_solved", "levels_completed", "real_env_confirmed"):
        assert field in spec


def test_req_phase4_051_selects_none_when_current_strict_nonspatial_pool_is_exhausted(tmp_path: Path) -> None:
    """REQ-PHASE4-051: the real survey has no unbanked strict non-spatial target."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    selected = select_next_unsolved_nonspatial(survey, baselines)
    evidence = build_selection_evidence(survey, baselines)

    assert selected is None
    assert {row["game"] for row in evidence["strict_nonspatial_rows"]} >= {
        "r11l",
        "sc25",
        "lp85",
        "tn36",
        "cd82",
        "dc22",
    }
    assert all(row["already_solved"] for row in evidence["strict_nonspatial_rows"] if row["has_baseline"])
    assert evidence["unsolved_strict_nonspatial_count"] == 0

    bad_json = tmp_path / "environment_files" / "aa01" / "aaaa1111"
    bad_json.mkdir(parents=True)
    bad_json.joinpath("metadata.json").write_text("{bad json", encoding="utf-8")
    bad_id = tmp_path / "environment_files" / "bb02" / "bbbb2222"
    bad_id.mkdir(parents=True)
    bad_id.joinpath("metadata.json").write_text(json.dumps({"game_id": "bb02"}), encoding="utf-8")
    good = tmp_path / "environment_files" / "cc03" / "cccc3333"
    good.mkdir(parents=True)
    good.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "cc03-cccc3333", "baseline_actions": []}),
        encoding="utf-8",
    )
    assert load_environment_baselines(tmp_path / "environment_files") == {"cc03": ("cc03-cccc3333", 0)}

    strict_survey = {
        "top_pick": "zz99",
        "ranked_targets": [],
        "per_game_surveys": [{"game": "zz99", "is_spatial_planning": False, "win_difficulty": "easy"}],
    }
    strict = select_next_unsolved_nonspatial(
        strict_survey,
        {"zz99": ("zz99-abcd1234", 9)},
        solved_prefixes=(),
    )
    assert strict == _target().__class__(
        game="zz99",
        game_id="zz99-abcd1234",
        baseline_actions=9,
        survey_rank=0,
        selection_mode="strict_survey_non_spatial",
        selection_reason="selected zz99 as the next unsolved strict non-spatial survey target",
        excluded_solved_games=(),
    )

    assert exp._fixture_available("bad") is False
    default_outcome = exp._run_selected_first_level(object(), strict)
    assert default_outcome.target_game == "zz99-abcd1234"
    assert default_outcome.failure_reason == "no_offline_driver_for_selected_strict_nonspatial_target"


def test_scenario_phase4_051_success_no_solve_and_blocked_artifacts_validate() -> None:
    """SCENARIO-PHASE4-051: only verified real-env evidence increments the game count."""

    success = build_artifact(
        _outcome(solved=True),
        _target(),
        random_seed=4151,
        duration_s=0.25,
        selection_evidence={"unsolved_strict_nonspatial_count": 1},
    )

    assert success["honest_verdict"] == "success: fifteenth_game_solved_zz99-abcd1234_at_action_5"
    assert success["game_solved"] is True
    assert success["total_games_solved"] == 14
    assert success["prior_total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert success["levels_completed"] == 1
    assert success["first_solve_at_action"] == 5
    assert success["real_env_confirmed"] is True
    assert success["field_principles"]["honest_verdict"] == "Terminal-prefixed. An honest no-solve is a COMPLETE verdict."
    assert success["field_principles"]["total_games_solved"].startswith("The monotonic progress metric")
    assert success["requirements"] == REQUIREMENTS
    assert artifact_schema_errors(success) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in success

    no_solve = build_artifact(
        FirstLevelOutcome(
            target_game="none",
            final_level_completed=0,
            first_solve_at_action=-1,
            exploration_actions_used=0,
            real_env_confirmed=False,
            verifier_validated=False,
            verification_decisions=[],
            action_plan=[],
            phase_trace=[{"phase": "observe", "source": "strict_nonspatial_pool_exhausted"}],
            induced_mechanic="none",
            failure_reason="no_unsolved_strict_nonspatial_candidates",
        ),
        None,
        random_seed=4151,
        duration_s=0.25,
        selection_evidence={"unsolved_strict_nonspatial_count": 0},
    )
    assert no_solve["honest_verdict"] == "complete: fifteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates"
    assert no_solve["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert no_solve["game_solved"] is False
    assert no_solve["real_env_confirmed"] is False
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(target_game="zz99-abcd1234", random_seed=4151, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert blocked["levels_completed"] == 0
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_051_schema_rejects_malformed_or_fabricated_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-051: schema guards prevent fabricated solved-game increments."""

    retained = validate_gap4_heldout_replay(0, 1, 2, 1)
    rejected = validate_gap4_heldout_replay(0, 0, 2, 1)
    no_heldout = validate_gap4_heldout_replay(0, 1, 0, 1)
    assert retained["retained"] is True
    assert retained["energy"] == 0.0
    assert rejected["retained"] is False
    assert no_heldout["retained"] is False

    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4151}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("total_games_solved must be a bare int" in err for err in artifact_schema_errors({"total_games_solved": True}))
    assert any("game_solved must be a bare bool" in err for err in artifact_schema_errors({"game_solved": "yes"}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4151}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("inference_substrate must equal" in err for err in artifact_schema_errors({"inference_substrate": "wrong"}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))
    assert any("field_principles must be a dict" in err for err in artifact_schema_errors({"field_principles": []}))
    assert any(
        "field_principles missing real_env_confirmed" in err
        for err in artifact_schema_errors({"field_principles": {"honest_verdict": "x"}})
    )

    bad_success = {
        "honest_verdict": "success: fifteenth_game_solved_zz99-abcd1234_at_action_5",
        "game_solved": False,
        "target_game": "none",
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "prior_total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "levels_completed": 0,
        "first_solve_at_action": -1,
        "real_env_confirmed": False,
        "verifier_validated": False,
        "solve_trace": {},
        "phase_trace": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "requirements": REQUIREMENTS,
        "field_principles": {
            "honest_verdict": "x",
            "total_games_solved": "x",
            "levels_completed": "x",
            "real_env_confirmed": "x",
        },
    }
    errors = artifact_schema_errors(bad_success)
    assert any("game_solved must be true" in err for err in errors)
    assert any("target_game must name" in err for err in errors)
    assert any("real_env_confirmed must be true" in err for err in errors)
    assert any("verifier_validated must be true" in err for err in errors)
    assert any("total_games_solved must increment" in err for err in errors)
    assert any("levels_completed must increment" in err for err in errors)
    assert any("first_solve_at_action must be positive" in err for err in errors)
    assert any("solve_trace must include phase_trace" in err for err in errors)

    bad_complete = {
        **bad_success,
        "honest_verdict": "complete: fifteenth_game_no_solve_zz99-abcd1234_x",
        "game_solved": True,
        "total_games_solved": 14,
        "real_env_confirmed": True,
    }
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_games_solved must remain" in err for err in complete_errors)
    assert any("game_solved must be false" in err for err in complete_errors)
    assert any("real_env_confirmed must be false" in err for err in complete_errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(solved=True), _target(), random_seed=4151, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game="zz99-abcd1234", random_seed=4151, duration_s=0.0)


def test_scenario_phase4_051_runner_writes_no_solve_success_and_blocked_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-051: runner writes stable terminal artifacts for each path."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4151_arc_incremental_progress.json").exists()

    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    no_baselines = exp.run(write=True)
    assert no_baselines["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir = tmp_path / "environment_files" / "r11l" / "495a7899"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "r11l-495a7899", "baseline_actions": [22]}),
        encoding="utf-8",
    )
    fixture_dir.joinpath("r11l.py").write_text("# marker\n", encoding="utf-8")

    no_solve = exp.run(write=True)
    assert no_solve["honest_verdict"] == "complete: fifteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates"
    assert no_solve["target_game"] == "none"
    written = json.loads((tmp_path / "results" / "experiment_4151_arc_incremental_progress.json").read_text())
    assert written == no_solve

    synthetic_survey = {
        "ranked_targets": [{"game": "zz99"}],
        "per_game_surveys": [{"game": "zz99", "is_spatial_planning": False}],
    }
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(synthetic_survey),
        encoding="utf-8",
    )
    missing_fixture = exp.run(write=True)
    assert missing_fixture["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert missing_fixture["target_game"] == "zz99"

    zz_dir = tmp_path / "environment_files" / "zz99" / "abcd1234"
    zz_dir.mkdir(parents=True)
    zz_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "zz99-abcd1234", "baseline_actions": [9]}),
        encoding="utf-8",
    )
    missing_fixture_py = exp.run(write=True)
    assert missing_fixture_py["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert missing_fixture_py["target_game"] == "zz99-abcd1234"

    zz_dir.joinpath("zz99.py").write_text("# marker\n", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_selected_first_level", lambda arcade, target: _outcome(solved=True))

    success = exp.run(write=True)
    assert success["honest_verdict"] == "success: fifteenth_game_solved_zz99-abcd1234_at_action_5"
    assert success["total_games_solved"] == 14

    monkeypatch.setattr(exp, "_run_selected_first_level", lambda arcade, target: (_ for _ in ()).throw(RuntimeError("boom")))
    errored = exp.run(write=False)
    assert errored["honest_verdict"].startswith("complete: fifteenth_game_no_solve_zz99-abcd1234_offline_run_failed_runtimeerror")
