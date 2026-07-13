"""Tests for Exp 4500 ARC submitted value-weight remeasurement.

Spec refs: REQ-REPORT-4500, SCENARIO-REPORT-4500-CONTROL,
SCENARIO-REPORT-4500-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4500_value_weight_remeasure as exp4500
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "fixture-torch",
        "env_game_blocked": True,
        "value_head_v3_model_present": True,
        "parity_test_target": "tests/python/test_arc_submitted_agent_parity.py",
    }


def _runner(game: str, value_weight: float) -> dict[str, object]:
    solved = game == "lp85" and value_weight in {0.0, 0.5, 1.0, 2.0, 5.0}
    return {
        "game": game,
        "value_weight": value_weight,
        "solved": solved,
        "actions_to_first_levelup": 21 if solved else None,
        "actions": 21 if solved else 400,
        "levels_delta": 1 if solved else 0,
        "start_level": 0,
        "reached_level": 1 if solved else 0,
        "wall_seconds": 1.0 + value_weight,
        "timed_out": False,
        "frame_only": True,
        "env_game_access_blocked": True,
    }


def test_req_report_4500_spec_declares_value_weight_sweep_contract() -> None:
    """REQ-REPORT-4500: OpenSpec names the value-weight sweep contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4500" in spec
    assert "SCENARIO-REPORT-4500-CONTROL" in spec
    assert "SCENARIO-REPORT-4500-SCHEMA" in spec
    assert exp4500.RESULT_RELATIVE_PATH in spec
    assert "0.0`, `0.5`, `1.0`, `2.0`, and `5.0" in spec
    assert "env._game" in spec
    assert "390 seconds" in spec
    for field, principle in exp4500.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4500_control_keeps_zero_when_positive_weights_do_not_win(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4500-CONTROL: no solve-rate win keeps value_weight=0."""

    sweep = exp4500.run_weight_sweep(
        ["lp85", "tr87"],
        value_weights=exp4500.VALUE_WEIGHTS,
        game_runner=_runner,
    )
    artifact = exp4500.build_artifact(
        sweep=sweep,
        preconditions_checked=_preconditions(),
    )

    assert artifact["honest_verdict"] == "complete: value_weight_remeasure_null_keep_0_1_of_2"
    assert artifact["inference_substrate"] == exp4500.INFERENCE_SUBSTRATE
    assert artifact["value_weights_tested"] == [0.0, 0.5, 1.0, 2.0, 5.0]
    assert artifact["control_value_weight"] == 0.0
    assert artifact["selected_value_weight"] == 0.0
    # NOTE (2026-07-12): SUBMITTED_VALUE_WEIGHT moved from exactly 0.0 to a tiny
    # bounded-positive 1e-12 in commit 0fad75f38 (PHASE A1, REQ-LEARN-4652 --
    # "the component-labeling cost fix makes a bounded positive value route
    # affordable"), a later, deliberate policy change independent of this null
    # result. "before"/"after" here must track whatever SUBMITTED_VALUE_WEIGHT
    # currently is, not a value frozen at this test's original authoring time.
    assert artifact["submitted_value_weight_before"] == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert artifact["submitted_value_weight_after"] == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert artifact["submitted_agent_config"] == SUBMITTED_AGENT_CONFIG
    assert artifact["selection"]["reason"] == "no_positive_weight_beats_control_within_budget"
    assert artifact["leaderboard_submission"] is False

    control = artifact["per_weight"][0]
    assert control["value_weight"] == 0.0
    assert control["heldout_solve_rate"] == pytest.approx(0.5)
    assert control["median_actions_to_first_levelup"] == 21
    assert control["median_per_game_wall_seconds"] == pytest.approx(1.0)
    assert control["per_game"][0]["env_game_access_blocked"] is True
    assert exp4500.artifact_schema_errors(artifact) == []

    written = exp4500.write_artifact(tmp_path, artifact)
    assert json.loads(written.read_text(encoding="utf-8")) == artifact


def test_scenario_report_4500_selection_requires_solve_rate_win_and_wall_budget() -> None:
    """SCENARIO-REPORT-4500-CONTROL: speed alone or tied solve-rate cannot raise weight."""

    summaries = [
        {
            "value_weight": 0.0,
            "heldout_solve_rate": 0.5,
            "solved_games": 1,
            "attempted_games": 2,
            "median_actions_to_first_levelup": 21,
            "median_per_game_wall_seconds": 1.0,
            "per_game": [],
        },
        {
            "value_weight": 0.5,
            "heldout_solve_rate": 1.0,
            "solved_games": 2,
            "attempted_games": 2,
            "median_actions_to_first_levelup": 20,
            "median_per_game_wall_seconds": 391.0,
            "per_game": [],
        },
        {
            "value_weight": 1.0,
            "heldout_solve_rate": 0.5,
            "solved_games": 1,
            "attempted_games": 2,
            "median_actions_to_first_levelup": 18,
            "median_per_game_wall_seconds": 0.5,
            "per_game": [],
        },
        {
            "value_weight": 2.0,
            "heldout_solve_rate": 1.0,
            "solved_games": 2,
            "attempted_games": 2,
            "median_actions_to_first_levelup": 19,
            "median_per_game_wall_seconds": 6.0,
            "per_game": [],
        },
    ]

    selected = exp4500.select_value_weight(
        summaries,
        control_value_weight=0.0,
        eval_budget_median_wall_s=390.0,
    )

    assert selected == {
        "selected_value_weight": 2.0,
        "control_solve_rate": 0.5,
        "selected_solve_rate": 1.0,
        "beats_control": True,
        "within_wall_budget": True,
        "should_raise_submitted_value_weight": True,
        "reason": "positive_weight_beats_control_within_budget",
    }


def test_req_report_4500_schema_rejects_missing_control_and_drift() -> None:
    """REQ-REPORT-4500: schema rejects missing control and config drift."""

    sweep = exp4500.run_weight_sweep(
        ["lp85", "tr87"],
        value_weights=exp4500.VALUE_WEIGHTS,
        game_runner=_runner,
    )
    artifact = exp4500.build_artifact(sweep=sweep, preconditions_checked=_preconditions())
    bad = {
        **artifact,
        "honest_verdict": "partial: stale",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "preconditions_checked": [],
        "field_principles": {"honest_verdict": {"principle": "wrong"}},
        "value_weights_tested": [0.5, 1.0],
        "control_value_weight": 0.5,
        "selected_value_weight": 0.5,
        "submitted_value_weight_after": 0.5,
        "submitted_agent_config": {**artifact["submitted_agent_config"], "value_weight": 0.5},
        "per_weight": [
            {
                **artifact["per_weight"][0],
                "value_weight": 0.5,
                "heldout_solve_rate": {"value": 0.5},
                "solved_games": "1",
                "attempted_games": "2",
                "median_per_game_wall_seconds": "1.0",
                "per_game": [
                    {
                        **artifact["per_weight"][0]["per_game"][0],
                        "env_game_access_blocked": False,
                    }
                ],
            }
        ],
    }

    errors = exp4500.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must equal verifier_ensemble_against_cached_candidates" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "field_principles must match required principles" in errors
    assert "value_weights_tested must include the zero-weight control" in errors
    assert "control_value_weight must be 0.0" in errors
    assert "submitted_agent_config must match SUBMITTED_AGENT_CONFIG" in errors
    assert "per_weight[0].heldout_solve_rate must be bare float" in errors
    assert "per_weight[0].solved_games must be bare int" in errors
    assert "per_weight[0].attempted_games must be bare int" in errors
    assert "per_weight[0].median_per_game_wall_seconds must be bare float or null" in errors
    assert "per_weight[0].per_game[0] must block env._game" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        exp4500.write_artifact(Path("/tmp"), bad)


def test_req_report_4500_runner_blocks_env_game_and_measures_first_levelup() -> None:
    """REQ-REPORT-4500: frame-only runner blocks env._game and records first level-up."""

    class FakeEnv:
        def __init__(self) -> None:
            self.steps = 0
            self._game = object()

        def reset(self) -> SimpleNamespace:
            return SimpleNamespace(level=0)

        def step(self, _action: object, data: object = None) -> SimpleNamespace:
            self.steps += 1
            return SimpleNamespace(level=1 if self.steps >= 2 else 0, data=data)

    class FakeArcade:
        def open_scorecard(self) -> str:
            return "scorecard"

        def make(self, game: str, scorecard_id: str) -> FakeEnv:
            assert game == "lp85"
            assert scorecard_id == "scorecard"
            return FakeEnv()

    class FakeAction:
        ACTION1 = object()

    class FakePolicy:
        def __init__(self) -> None:
            self.moves = [("RESET", None), (1, None), (1, {"x": 1})]

        def is_done(self, _frames: list[object], _latest: object) -> bool:
            return False

        def next_move(self, _frames: list[object], _latest: object) -> tuple[object, object]:
            return self.moves.pop(0) if self.moves else (None, None)

    blocked = exp4500._BlockedEnvGame(FakeEnv())
    with pytest.raises(AttributeError, match="blocked"):
        _ = blocked._game
    assert blocked.reset().level == 0
    assert exp4500._NoopProposer().induce() == (False, {})

    row = exp4500.run_policy_game(
        "lp85",
        value_weight=0.5,
        arcade=FakeArcade(),
        game_action=FakeAction,
        budget=5,
        wall_budget_s=390.0,
        policy_factory=lambda _game, _weight: FakePolicy(),
        level_getter=lambda frame: frame.level,
        clock=iter([10.0, 10.1, 10.2, 10.3, 10.4, 13.25]).__next__,
    )

    assert row["game"] == "lp85"
    assert row["value_weight"] == 0.5
    assert row["solved"] is True
    assert row["actions_to_first_levelup"] == 2
    assert row["actions"] == 2
    assert row["levels_delta"] == 1
    assert row["wall_seconds"] == pytest.approx(3.25)
    assert row["frame_only"] is True
    assert row["env_game_access_blocked"] is True


def test_req_report_4500_preconditions_record_verified_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-4500: preconditions list the import smoke and Torch resource."""

    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "arc_verifier_cross_game_v3.json").write_text("{}", encoding="utf-8")

    class FakeKit:
        @staticmethod
        def offline_arcade() -> object:
            return object()

    monkeypatch.setattr(exp4500, "_import_arc_solver_kit", lambda: FakeKit)
    monkeypatch.setattr(exp4500, "_import_torch_version", lambda: "fixture-torch")

    checks = exp4500.check_preconditions(tmp_path)

    assert checks["agents_md_read"] is True
    assert checks["codex_md_read"] is True
    assert checks["offline_arcade_import_smoke"] is True
    assert checks["torch_import"] is True
    assert checks["torch_version"] == "fixture-torch"
    assert checks["env_game_blocked"] is True
    assert checks["value_head_v3_model_present"] is True
