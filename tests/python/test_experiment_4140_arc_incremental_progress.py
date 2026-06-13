"""Tests for Exp 4140 ARC-AGI-3 scoped incremental progress.

Spec refs: REQ-PHASE4-050, SCENARIO-PHASE4-050.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4140_arc_incremental_progress as exp
from carnot.experiment_4140_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_LEVELS_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    R11L_GAME_ID,
    FrontierOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    load_environment_baselines,
    select_target_from_survey,
    validate_frontier_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _target() -> TargetSelection:
    return TargetSelection(
        game="r11l",
        game_id=R11L_GAME_ID,
        target_level=5,
        prior_level=4,
        baseline_actions=52,
        selection_mode="next_level_after_strict_nonspatial_exhaustion",
        selection_reason="selected r11l L5 after strict non-spatial L1 pool was exhausted",
        strict_nonspatial_exhausted=True,
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    verification = validate_frontier_replay(
        start_level=4,
        final_level=5 if advanced else 4,
        heldout_transition_count=2,
        predicted_level=5,
    )
    return FrontierOutcome(
        target_game=R11L_GAME_ID,
        target_level=5,
        prior_level=4,
        final_level_completed=5 if advanced else 4,
        executed_real_env_actions=6 if advanced else 0,
        exploration_actions_used=24,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[verification],
        action_plan=[{"kind": "safe-composite-path", "moves": [{"group_id": "g", "piece_index": 0}]}]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 4},
            {"phase": "explore", "actions": 24},
            {"phase": "induce", "mechanic": "r11l_safe_composite_path"},
            verification,
            {"phase": "act", "levels_completed": 5 if advanced else 4},
        ],
        induced_mechanic="r11l safe-composite path through collision-forbidden mask",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_050_spec_declares_exp4140_contract() -> None:
    """REQ-PHASE4-050: OpenSpec declares the 4140 scoped incremental artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-050" in spec
    assert "SCENARIO-PHASE4-050" in spec
    assert "experiment_4140_arc_incremental_progress.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "r11l-495a7899" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_levels_solved", "levels_completed", "real_env_confirmed", "target_game"):
        assert field in spec


def test_req_phase4_050_selects_r11l_next_frontier_when_strict_l1_pool_is_exhausted() -> None:
    """REQ-PHASE4-050: selection falls back to the next r11l frontier level."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = {"r11l": (R11L_GAME_ID, 22), "bp35": ("bp35-0a0ad940", 21)}

    selected = select_target_from_survey(
        survey,
        baselines,
        solved_prefixes=exp.SOLVED_PREFIXES_BEFORE_4140,
        frontier_levels={"r11l": 4},
    )

    assert selected.game == "r11l"
    assert selected.game_id == R11L_GAME_ID
    assert selected.prior_level == 4
    assert selected.target_level == 5
    assert selected.selection_mode == "next_level_after_strict_nonspatial_exhaustion"
    assert selected.strict_nonspatial_exhausted is True

    strict_survey = {
        "ranked_targets": [{"game": "zz99"}],
        "per_game_surveys": [{"game": "zz99", "is_spatial_planning": False, "win_difficulty": "easy"}],
    }
    strict = select_target_from_survey(
        strict_survey,
        {"zz99": ("zz99-abcd1234", 7)},
        solved_prefixes=(),
        frontier_levels={},
    )
    assert strict.game == "zz99"
    assert strict.target_level == 1
    assert strict.prior_level == 0
    assert strict.selection_mode == "strict_nonspatial_l1"
    assert strict.strict_nonspatial_exhausted is False

    with pytest.raises(ValueError, match="no selectable ARC-AGI-3 target"):
        select_target_from_survey({"ranked_targets": [], "per_game_surveys": []}, {}, solved_prefixes=())


def test_scenario_phase4_050_artifacts_and_schema_guard_increment_only_real_confirmed_levels() -> None:
    """SCENARIO-PHASE4-050: artifacts preserve the monotonic level counter contract."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4140, duration_s=0.25)

    assert success["honest_verdict"] == "success: incremental_progress_r11l-495a7899_advanced_to_L5_total14"
    assert success["total_games_solved"] == 13
    assert success["total_levels_solved"] == 14
    assert success["prior_total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert success["levels_completed"] == 5
    assert success["new_levels_solved_this_task"] == 1
    assert success["real_env_confirmed"] is True
    assert success["target_game"] == R11L_GAME_ID
    assert success["target_level"] == 5
    assert success["field_principles"]["honest_verdict"] == "Terminal-prefixed. An honest no-solve is a COMPLETE verdict."
    assert success["field_principles"]["total_levels_solved"].startswith("The monotonic progress metric")
    assert success["requirements"] == REQUIREMENTS
    assert artifact_schema_errors(success) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in success

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4140, duration_s=0.25)
    assert no_solve["honest_verdict"] == "complete: incremental_progress_no_solve_r11l-495a7899_L5_no_verifier_validated_level_up_candidate"
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(target_game=R11L_GAME_ID, target_level=5, random_seed=4140, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    bad_success = dict(success)
    bad_success["real_env_confirmed"] = False
    bad_success["total_levels_solved"] = PRIOR_TOTAL_LEVELS_SOLVED
    errors = artifact_schema_errors(bad_success)
    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("total_levels_solved must increment" in err for err in errors)


def test_scenario_phase4_050_schema_rejects_malformed_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-050: schema guards prevent fabricated or malformed increments."""

    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4140}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("total_levels_solved must be a bare int" in err for err in artifact_schema_errors({"total_levels_solved": True}))
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 1}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4140}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))
    assert any("field_principles must be a dict" in err for err in artifact_schema_errors({"field_principles": []}))
    assert any(
        "field_principles missing target_game" in err
        for err in artifact_schema_errors({"field_principles": {"honest_verdict": "x"}})
    )

    bad_success = {
        "honest_verdict": "success: incremental_progress_r11l-495a7899_advanced_to_L5_total14",
        "target_game": R11L_GAME_ID,
        "target_level": 5,
        "total_games_solved": 13,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": 14,
        "new_levels_solved_this_task": 0,
        "levels_completed": 4,
        "real_env_confirmed": True,
        "verifier_validated": False,
        "solve_trace": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "requirements": REQUIREMENTS,
        "field_principles": {
            "honest_verdict": "x",
            "total_levels_solved": "x",
            "levels_completed": "x",
            "real_env_confirmed": "x",
            "target_game": "x",
        },
    }
    errors = artifact_schema_errors(bad_success)
    assert any("verifier_validated must be true" in err for err in errors)
    assert any("new_levels_solved_this_task must be one" in err for err in errors)
    assert any("levels_completed must reach target_level" in err for err in errors)
    assert any("solve_trace must include phase_trace" in err for err in errors)

    bad_complete = {
        **bad_success,
        "honest_verdict": "complete: incremental_progress_no_solve_r11l-495a7899_L5_x",
        "total_levels_solved": 14,
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": False,
    }
    errors = artifact_schema_errors(bad_complete)
    assert any("total_levels_solved must remain" in err for err in errors)
    assert any("new_levels_solved_this_task must be zero" in err for err in errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(advanced=True), _target(), random_seed=4140, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game=R11L_GAME_ID, target_level=5, random_seed=4140, duration_s=0.0)


def test_req_phase4_050_baselines_and_frontier_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-PHASE4-050: local metadata, fixture, and frontier helpers handle edge cases."""

    fixture_dir = tmp_path / "environment_files" / "aa01" / "abcd1234"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text("{bad json", encoding="utf-8")
    bad_id_dir = tmp_path / "environment_files" / "bb02" / "bbbb2222"
    bad_id_dir.mkdir(parents=True)
    bad_id_dir.joinpath("metadata.json").write_text(json.dumps({"game_id": "bb02"}), encoding="utf-8")
    good_dir = tmp_path / "environment_files" / "cc03" / "cccc3333"
    good_dir.mkdir(parents=True)
    good_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "cc03-cccc3333", "baseline_actions": [9]}),
        encoding="utf-8",
    )

    assert load_environment_baselines(tmp_path / "environment_files") == {"cc03": ("cc03-cccc3333", 9)}

    survey = {
        "top_pick": "r11l",
        "ranked_targets": [],
        "per_game_surveys": [{"game": "r11l", "is_spatial_planning": False}],
    }
    selected = select_target_from_survey(
        survey,
        {"r11l": (R11L_GAME_ID, 22)},
        solved_prefixes=("r11l",),
        frontier_levels={"r11l": 4},
    )
    assert selected.target_level == 5

    monkeypatch.setattr(exp, "REPO", tmp_path)
    assert exp._fixture_available("bad") is False
    assert exp._prior_frontier_levels() == {"r11l": 4}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_3992_incremental_levels_verifier_validated.json").write_text(
        json.dumps({"real_env_confirmed": True, "ACCURACY_levels_solved": 3}),
        encoding="utf-8",
    )
    assert exp._prior_frontier_levels() == {"r11l": 3}


def test_scenario_phase4_050_runner_writes_fake_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-050: runner writes a stable artifact from real-env evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4129_fourteenth_game_explore_first.json").write_text(
        json.dumps({"total_games_solved": 13, "levels_completed": 1}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4021_heuristic_search_over_verified_wm.json").write_text(
        json.dumps({"levels_completed_after": 4, "real_env_confirmed": True}),
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "environment_files" / "r11l" / "495a7899"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": R11L_GAME_ID, "baseline_actions": [22, 33, 51, 26, 52, 49]}),
        encoding="utf-8",
    )
    fixture_dir.joinpath("r11l.py").write_text("# synthetic fixture marker\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(
        exp,
        "_run_r11l_next_frontier",
        lambda arc, target: _outcome(advanced=True),
    )

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["total_levels_solved"] == 14
    written = json.loads((tmp_path / "results" / "experiment_4140_arc_incremental_progress.json").read_text())
    assert written == artifact


def test_scenario_phase4_050_runner_blocked_and_no_driver_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-050: runner reports blocked/no-driver paths without inflating progress."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4140_arc_incremental_progress.json").exists()

    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"ranked_targets": [], "per_game_surveys": []}),
        encoding="utf-8",
    )
    no_target = exp.run(write=True)
    assert no_target["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps(
            {
                "ranked_targets": [{"game": "zz99"}],
                "per_game_surveys": [{"game": "zz99", "is_spatial_planning": False}],
            }
        ),
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "environment_files" / "zz99" / "abcd1234"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "zz99-abcd1234", "baseline_actions": [7]}),
        encoding="utf-8",
    )
    missing_fixture_py = exp.run(write=True)
    assert missing_fixture_py["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir.joinpath("zz99.py").write_text("# marker\n", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    no_driver = exp.run(write=False)
    assert no_driver["honest_verdict"].startswith("complete: incremental_progress_no_solve_zz99-abcd1234_L1")
    assert no_driver["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED

    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    r11l_dir = tmp_path / "environment_files" / "r11l" / "495a7899"
    r11l_dir.mkdir(parents=True, exist_ok=True)
    r11l_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": R11L_GAME_ID, "baseline_actions": [22]}),
        encoding="utf-8",
    )
    r11l_dir.joinpath("r11l.py").write_text("# marker\n", encoding="utf-8")
    (tmp_path / "results" / "experiment_4021_heuristic_search_over_verified_wm.json").write_text(
        json.dumps({"levels_completed_after": 4, "real_env_confirmed": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "_run_r11l_next_frontier", lambda arc, target: (_ for _ in ()).throw(RuntimeError("boom")))
    errored = exp.run(write=False)
    assert errored["honest_verdict"].startswith("complete: incremental_progress_no_solve_r11l-495a7899_L5")
    assert "offline_run_failed_runtimeerror" in errored["honest_verdict"]
