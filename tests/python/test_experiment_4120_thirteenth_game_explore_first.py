"""Tests for Exp 4120 ARC-AGI-3 thirteenth-game strict non-spatial attempt.

Spec refs: REQ-PHASE4-048, SCENARIO-PHASE4-048.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_exp4120_thirteenth_game_explore_first as arc4120
from carnot.agentic.arc_exp4070_ninth_game_explore_first import load_environment_baselines
from carnot.agentic.arc_exp4120_thirteenth_game_explore_first import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    NoUnsolvedNonSpatialCandidate,
    artifact_schema_errors,
    blocked_artifact,
    build_no_solve_artifact,
    build_selection_report,
    select_exp4120_candidate_from_survey,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4120_thirteenth_game_explore_first as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
SURVEY_PATH = REPO / "results" / "arc3_win_condition_survey.json"


def _write_metadata(root: Path, game: str, game_id: str, baseline_actions: int) -> None:
    suffix = game_id.split("-", maxsplit=1)[1]
    metadata_dir = root / "environment_files" / game / suffix
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": game_id, "baseline_actions": [baseline_actions]}),
        encoding="utf-8",
    )
    metadata_dir.joinpath(f"{game}.py").write_text("# synthetic offline fixture\n", encoding="utf-8")


def test_req_phase4_048_spec_declares_exp4120_contract() -> None:
    """REQ-PHASE4-048: OpenSpec declares Exp 4120 and required principle fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-048" in spec
    assert "SCENARIO-PHASE4-048" in spec
    assert "experiment_4120_thirteenth_game_explore_first.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "complete: thirteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates" in spec
    assert INFERENCE_SUBSTRATE in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in ("honest_verdict", "total_games_solved", "levels_completed", "real_env_confirmed"):
        assert field in spec


def test_req_phase4_048_reports_strict_nonspatial_exhaustion() -> None:
    """REQ-PHASE4-048: candidate selection refuses a spatial fallback when strict rows are exhausted."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")

    report = build_selection_report(survey, baselines)

    assert report.no_unsolved_strict_nonspatial_candidates is True
    assert [candidate.game for candidate in report.unsolved_strict_nonspatial_candidates] == []
    assert [candidate.game for candidate in report.strict_nonspatial_candidates] == [
        "cd82",
        "dc22",
        "lp85",
        "r11l",
        "sc25",
        "tn36",
    ]
    assert all(candidate.game in report.solved_prefixes for candidate in report.strict_nonspatial_candidates)
    assert [candidate.game for candidate in report.remaining_unsolved_offline_candidates[:3]] == [
        "bp35",
        "ls20",
        "re86",
    ]

    with pytest.raises(NoUnsolvedNonSpatialCandidate) as excinfo:
        select_exp4120_candidate_from_survey(survey, baselines)
    assert excinfo.value.report.no_unsolved_strict_nonspatial_candidates is True

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
    selected = select_exp4120_candidate_from_survey(
        strict_survey,
        {"aa01": ("aa01-game", 8), "bb02": ("bb02-game", 4)},
        solved_prefixes=(),
    )
    assert selected.game == "bb02"
    assert selected.selection_mode == "strict_survey_non_spatial"


def test_scenario_phase4_048_no_solve_artifact_has_required_fields() -> None:
    """SCENARIO-PHASE4-048: strict exhaustion is a complete terminal no-solve verdict."""

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    baselines = load_environment_baselines(REPO / "environment_files")
    report = build_selection_report(survey, baselines)

    artifact = build_no_solve_artifact(
        report,
        random_seed=4120,
        duration_s=0.25,
        offline_driver_available=True,
        arc_env_count=25,
    )

    assert artifact["honest_verdict"] == (
        "complete: thirteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates"
    )
    assert artifact["game_solved"] is False
    assert artifact["target_game"] == "none"
    assert artifact["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert artifact["levels_completed"] == 0
    assert artifact["first_solve_at_action"] == -1
    assert artifact["actions_vs_baseline"] == 0.0
    assert artifact["real_env_confirmed"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["nonspatial_candidates_exhausted"] is True
    assert artifact["solve_trace"]["selection_report"]["no_unsolved_strict_nonspatial_candidates"] is True
    assert artifact["solve_trace"]["actions"] == []
    assert artifact["field_principles"]["honest_verdict"].startswith("Terminal-prefixed")
    assert artifact["requirements"] == ["REQ-PHASE4-048", "SCENARIO-PHASE4-048"]
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_req_phase4_048_blocked_and_schema_validation(monkeypatch) -> None:
    """REQ-PHASE4-048: blocked and malformed artifacts do not inflate solved games."""

    blocked = blocked_artifact(
        random_seed=4120,
        duration_s=0.0,
        reason="offline fixtures unavailable",
    )
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["game_solved"] is False
    assert blocked["target_game"] == "none"
    assert blocked["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert blocked["levels_completed"] == 0
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []

    assert any("missing required field levels_completed" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4120}))
    assert any("honest_verdict must start" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("game_solved must be a bare bool" in err for err in artifact_schema_errors({"game_solved": "no"}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4120}))
    assert any("total_games_solved must be a bare int" in err for err in artifact_schema_errors({"total_games_solved": "12"}))
    assert any("levels_completed must be a bare int" in err for err in artifact_schema_errors({"levels_completed": "0"}))
    assert any(
        "first_solve_at_action must be a bare int" in err
        for err in artifact_schema_errors({"first_solve_at_action": 0.0})
    )
    assert any(
        "actions_vs_baseline must be a bare float" in err
        for err in artifact_schema_errors({"actions_vs_baseline": "0.0"})
    )
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 0}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any(
        "inference_substrate must equal" in err
        for err in artifact_schema_errors({"inference_substrate": "wrong"})
    )
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))

    bad_no_solve = {
        "honest_verdict": "complete: thirteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates",
        "game_solved": True,
        "target_game": "bp35-0a0ad940",
        "total_games_solved": 13,
        "levels_completed": 1,
        "first_solve_at_action": 1,
        "actions_vs_baseline": 1.0,
        "real_env_confirmed": True,
        "solve_trace": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    bad_no_solve_errors = artifact_schema_errors(bad_no_solve)
    assert any("game_solved must be false" in err for err in bad_no_solve_errors)
    assert any("target_game must be none" in err for err in bad_no_solve_errors)
    assert any("total_games_solved must remain at 12" in err for err in bad_no_solve_errors)
    assert any("levels_completed must be zero" in err for err in bad_no_solve_errors)
    assert any("first_solve_at_action must be -1" in err for err in bad_no_solve_errors)
    assert any("actions_vs_baseline must be 0.0" in err for err in bad_no_solve_errors)
    assert any("real_env_confirmed must be false" in err for err in bad_no_solve_errors)

    monkeypatch.setattr(arc4120, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        arc4120.build_no_solve_artifact(
            build_selection_report({}, {}),
            random_seed=4120,
            duration_s=0.0,
            offline_driver_available=True,
            arc_env_count=0,
        )
    with pytest.raises(ValueError, match="forced schema error"):
        arc4120.blocked_artifact(random_seed=4120, duration_s=0.0, reason="x")


def test_scenario_phase4_048_script_writes_no_solve_from_exhausted_strict_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-048: runner writes no-solve when strict non-spatial rows are exhausted."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for game, game_id, baseline in (
        ("cd82", "cd82-fb555c5d", 55),
        ("dc22", "dc22-fdcac232", 59),
        ("lp85", "lp85-305b61c3", 17),
        ("r11l", "r11l-495a7899", 22),
        ("sc25", "sc25-635fd71a", 36),
        ("tn36", "tn36-ef4dde99", 32),
        ("bp35", "bp35-0a0ad940", 21),
    ):
        _write_metadata(tmp_path, game, game_id, baseline)

    fake_arcade = type("Arcade", (), {"get_environments": lambda self: ["a", "b", "c"]})()
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: fake_arcade)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == (
        "complete: thirteenth_game_no_solve_no_unsolved_strict_nonspatial_candidates"
    )
    assert artifact["game_solved"] is False
    assert artifact["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert artifact["real_env_confirmed"] is False
    assert artifact["arc_env_count"] == 3
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["experiment"] == (
        "experiment_4120_thirteenth_game_explore_first"
    )


def test_scenario_phase4_048_script_blocks_when_fixtures_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-048: missing offline fixtures stop with the required blocked verdict."""

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        SURVEY_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert artifact["game_solved"] is False
    assert artifact["real_env_confirmed"] is False
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
