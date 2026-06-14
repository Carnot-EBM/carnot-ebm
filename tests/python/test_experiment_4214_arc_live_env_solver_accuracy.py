"""Tests for Exp 4214 ARC-AGI-3 live solver accuracy probe.

Spec refs: REQ-PHASE4-060, SCENARIO-PHASE4-060.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import carnot.experiment_4214_arc_live_env_solver_accuracy as exp
from carnot.agentic.arc_agi3_live_adapter import ArcLivePreconditions, EnvironmentSummary
from carnot.experiment_4202_arc_live_env_solver_vs_floor import FloorBaseline
from carnot.experiment_4214_arc_live_env_solver_accuracy import (
    LP85_GAME_ID,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    completion_action_budget,
    extract_exp4202_floor,
    run_solver_completion,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeActionEnum:
    ACTION6 = SimpleNamespace(value=6, name="ACTION6")

    @staticmethod
    def from_id(action_id: int) -> SimpleNamespace:
        return SimpleNamespace(value=int(action_id), name=f"ACTION{int(action_id)}")


class FakeEnv:
    scorecard_id = "accuracy-open-scorecard"

    def __init__(self, *, advance_after: int = 5) -> None:
        self._advance_after = int(advance_after)
        self._level = 0
        self.actions: list[dict[str, int] | None] = []

    def reset(self) -> SimpleNamespace:
        self._level = 0
        self.actions.clear()
        return self._frame()

    def step(
        self,
        action: object,
        data: dict[str, int] | None = None,
        reasoning: dict | None = None,
    ) -> SimpleNamespace:
        del reasoning
        assert int(getattr(action, "value", action)) == 6
        self.actions.append(data)
        if len(self.actions) >= self._advance_after:
            self._level = 1
        return self._frame()

    def _frame(self) -> SimpleNamespace:
        grid = np.zeros((3, 3), dtype=np.int16)
        grid[1, 1] = len(self.actions)
        return SimpleNamespace(
            frame=grid,
            levels_completed=self._level,
            available_actions=[6],
            state="PLAYING",
            guid="accuracy-fixture-guid",
        )


class FakeScoreProvider:
    score_source = "fixture_open_scorecard"

    def __call__(self, env: FakeEnv) -> SimpleNamespace:
        completed = int(env._level)
        return SimpleNamespace(
            guid="accuracy-fixture-guid",
            score=115.0 if completed else 0.0,
            levels_completed=completed,
            actions=len(env.actions),
            resets=1,
            completed=False,
            level_actions=[len(env.actions)],
            level_baseline_actions=[17],
            message="",
        )


def _solver_plan() -> list[dict[str, object]]:
    return [
        {"action_id": 6, "x": 4, "y": 32, "source": "banked_lp85_L1_replay", "expected_levels_completed": 0},
        {"action_id": 6, "x": 4, "y": 32, "source": "banked_lp85_L1_replay", "expected_levels_completed": 0},
        {"action_id": 6, "x": 4, "y": 32, "source": "banked_lp85_L1_replay", "expected_levels_completed": 0},
        {"action_id": 6, "x": 4, "y": 32, "source": "banked_lp85_L1_replay", "expected_levels_completed": 0},
        {"action_id": 6, "x": 4, "y": 32, "source": "banked_lp85_L1_replay", "expected_levels_completed": 1},
    ]


def _prior_solver_artifact() -> dict[str, object]:
    return {
        "phase_trace": [
            {"source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0},
            {"source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 1},
        ]
    }


def _exp4202_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4202_arc_live_env_solver_vs_floor",
        "random_greedy_floor": {
            "environment": {
                "game_id": LP85_GAME_ID,
                "title": "LP85",
                "tags": ["click"],
                "baseline_actions": [17, 38],
            },
            "actions_taken": 6,
            "baseline_actions": 17,
            "actions_vs_baseline_actions": 6 / 17,
            "score": 0.0,
            "levels_completed": 0,
            "source_path": "results/experiment_4191_arc_live_env_grounding_probe.json",
        },
    }


def _floor() -> FloorBaseline:
    return extract_exp4202_floor(_exp4202_artifact(), source_path="results/experiment_4202_arc_live_env_solver_vs_floor.json")


def test_req_phase4_060_spec_declares_exp4214_contract() -> None:
    """REQ-PHASE4-060: OpenSpec declares the live solver accuracy artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-060" in spec
    assert "SCENARIO-PHASE4-060" in spec
    assert "experiment_4214_arc_live_env_solver_accuracy.json" in spec
    assert "blocked_arc_live_unreachable" in spec
    assert "solver_completes_level" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_060_blocked_artifact_is_terminal_and_non_submitting() -> None:
    """SCENARIO-PHASE4-060: missing SDK/network stops before live contact."""

    artifact = blocked_artifact(
        preconditions=ArcLivePreconditions(
            sdk_importable=False,
            sdk_version="missing",
            network_reachable=True,
            base_url="https://three.arcprize.org",
            error="No module named arc_agi",
        ),
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_arc_live_unreachable"
    assert artifact["solver_completes_level"] is False
    assert artifact["solver_beats_floor"] == {}
    assert artifact["live_env_metrics"] == {}
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["leaderboard_submission_attempted"] is False
    assert artifact["scorecard_closed"] is False
    assert artifact["requirements"] == REQUIREMENTS
    assert artifact_schema_errors(artifact) == []


def test_req_phase4_060_loads_exp4202_same_env_floor_and_completion_budget() -> None:
    """REQ-PHASE4-060: the live probe inherits Exp 4202's same-env floor."""

    floor = _floor()

    assert floor.environment.game_id == LP85_GAME_ID
    assert floor.actions_taken == 6
    assert floor.baseline_actions == 17
    assert completion_action_budget(floor.environment, floor, _solver_plan(), requested_action_budget=2) == 17
    assert completion_action_budget(EnvironmentSummary(LP85_GAME_ID, "LP85", ["click"], []), floor, _solver_plan()) == 17


def test_req_phase4_060_helper_edges_and_floor_defenses(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-060: helper loading and floor validation stay deterministic."""

    floor_path = tmp_path / "floor.json"
    floor_path.write_text(json.dumps(_exp4202_artifact()), encoding="utf-8")
    solver_path = tmp_path / "solver.json"
    solver_path.write_text(json.dumps(_prior_solver_artifact()), encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    loaded_floor = exp.load_exp4202_floor(floor_path)
    assert loaded_floor.source_path == "floor.json"
    assert exp._read_json(floor_path)["experiment"] == "experiment_4202_arc_live_env_solver_vs_floor"
    assert len(exp.load_completion_solver_plan(solver_path)) == 2

    other_path = tmp_path.parent / "outside-floor.json"
    other_path.write_text(json.dumps(_exp4202_artifact()), encoding="utf-8")
    assert exp.load_exp4202_floor(other_path).source_path == str(other_path)

    with pytest.raises(ValueError, match="random_greedy_floor"):
        extract_exp4202_floor({})
    with pytest.raises(ValueError, match="floor missing environment"):
        extract_exp4202_floor({"random_greedy_floor": {}})
    with pytest.raises(ValueError, match="same LP85 environment"):
        extract_exp4202_floor({"random_greedy_floor": {"environment": {"game_id": "other-env"}}})


def test_scenario_phase4_060_solver_completion_is_accuracy_win() -> None:
    """SCENARIO-PHASE4-060: scorecard level completion is the bare accuracy gate."""

    floor = _floor()
    outcome = run_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=2,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    artifact = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )

    assert outcome.action_budget == 17
    assert outcome.actions_taken == 5
    assert artifact["honest_verdict"] == "success: solver_completes_level_live_lp85-305b61c3"
    assert artifact["solver_completes_level"] is True
    assert artifact["live_env_metrics"]["levels_completed"] == 1
    assert artifact["live_env_metrics"]["score"] == 115.0
    assert artifact["live_env_metrics"]["action_budget"] == 17
    assert artifact["solver_beats_floor"]["accuracy"]["beats"] is True
    assert artifact["solver_beats_floor"]["efficiency"]["beats"] is True
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["scorecard_closed"] is False
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_060_zero_level_scorecard_is_complete_verdict() -> None:
    """SCENARIO-PHASE4-060: zero-level live scorecard evidence remains a valid result."""

    floor = _floor()
    outcome = run_solver_completion(
        FakeEnv(advance_after=99),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    artifact = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only"
    assert artifact["solver_completes_level"] is False
    assert artifact["live_env_metrics"]["levels_completed"] == 0
    assert artifact["solver_beats_floor"]["accuracy"]["beats"] is False
    assert artifact["solver_beats_floor"]["efficiency"]["beats"] is True
    assert artifact_schema_errors(artifact) == []

    score_only_floor = FloorBaseline(
        environment=floor.environment,
        actions_taken=5,
        baseline_actions=17,
        actions_vs_baseline_actions=5 / 17,
        score=-1.0,
        levels_completed=0,
        source_path="fixture",
    )
    score_only = build_artifact(
        outcome=outcome,
        floor=score_only_floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )
    assert score_only["honest_verdict"] == "complete: solver_score_beats_floor_without_completion_live_lp85-305b61c3"

    no_beat_floor = FloorBaseline(
        environment=floor.environment,
        actions_taken=5,
        baseline_actions=17,
        actions_vs_baseline_actions=5 / 17,
        score=0.0,
        levels_completed=0,
        source_path="fixture",
    )
    no_beat = build_artifact(
        outcome=outcome,
        floor=no_beat_floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )
    assert no_beat["honest_verdict"] == "complete: solver_completes_0_levels_live_lp85-305b61c3"


def test_scenario_phase4_060_schema_rejects_malformed_accuracy_claims() -> None:
    """SCENARIO-PHASE4-060: malformed completion, score, and submission claims fail."""

    floor = _floor()
    outcome = run_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    good = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
    )
    bad = {
        **good,
        "honest_verdict": "maybe",
        "solver_completes_level": "true",
        "solver_beats_floor": [],
        "live_env_metrics": {"score": "bad", "levels_completed": 1, "actions_taken": 5, "baseline_actions": 17, "action_budget": 6},
        "no_leaderboard_submission": False,
        "leaderboard_submission_attempted": True,
        "scorecard_closed": True,
        "preconditions_checked": [],
        "requirements": [],
        "field_principles": [],
    }

    errors = artifact_schema_errors(bad)
    assert any("honest_verdict must be terminal-prefixed" in err for err in errors)
    assert any("solver_completes_level must be a bare bool" in err for err in errors)
    assert any("solver_beats_floor must be a dict" in err for err in errors)
    assert any("live_env_metrics.score must be numeric" in err for err in errors)
    assert any("live_env_metrics.action_budget must be >= baseline_actions" in err for err in errors)
    assert any("no_leaderboard_submission must be true" in err for err in errors)
    assert any("leaderboard_submission_attempted must be false" in err for err in errors)
    assert any("scorecard_closed must be false" in err for err in errors)
    assert any("preconditions_checked must be a dict" in err for err in errors)
    assert any("requirements must include" in err for err in errors)
    assert any("field_principles must be a dict" in err for err in errors)
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))

    assert any(
        "honest_verdict must be a string" in err
        for err in artifact_schema_errors({**good, "honest_verdict": 4214})
    )
    blocked_bad = blocked_artifact(
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        duration_s=0,
    )
    assert any(
        "blocked artifacts must leave solver_beats_floor empty" in err
        for err in artifact_schema_errors({**blocked_bad, "solver_beats_floor": {"overall": False}})
    )
    assert any(
        "blocked artifacts must leave live_env_metrics empty" in err
        for err in artifact_schema_errors({**blocked_bad, "live_env_metrics": {"score": 0.0}})
    )

    schema_edges = {
        **good,
        "solver_completes_level": False,
        "solver_beats_floor": {"accuracy": {}, "overall": "true"},
        "live_env_metrics": {
            "score": 1.0,
            "levels_completed": "1",
            "actions_taken": "5",
            "baseline_actions": "17",
        },
        "offline_validation": {"passed": False},
        "real_metric_mapping": {},
        "preconditions_checked": {"sdk_importable": "yes", "network_reachable": "yes", "base_url": "url"},
        "field_principles": {"honest_verdict": "present"},
    }
    edge_errors = artifact_schema_errors(schema_edges)
    assert any("preconditions_checked missing sdk_version" in err for err in edge_errors)
    assert any("preconditions_checked.sdk_importable must be a bare bool" in err for err in edge_errors)
    assert any("field_principles missing solver_completes_level" in err for err in edge_errors)
    assert any("solver_beats_floor missing efficiency" in err for err in edge_errors)
    assert any("solver_beats_floor.overall must be a bare bool" in err for err in edge_errors)
    assert any("live_env_metrics missing action_budget" in err for err in edge_errors)
    assert any("live_env_metrics.levels_completed must be a bare int" in err for err in edge_errors)
    assert any("reachable artifacts require passed offline_validation" in err for err in edge_errors)
    assert any("real_metric_mapping must equal" in err for err in edge_errors)
    assert any(
        "solver_completes_level must equal live_env_metrics.levels_completed>=1" in err
        for err in artifact_schema_errors({**good, "solver_completes_level": False})
    )

    assert any(
        "live_env_metrics must be a dict" in err
        for err in artifact_schema_errors({**good, "live_env_metrics": []})
    )
    with pytest.raises(ValueError, match="same environment"):
        build_artifact(
            outcome=outcome,
            floor=FloorBaseline(
                environment=EnvironmentSummary("other-env", "OTHER", [], [1]),
                actions_taken=1,
                baseline_actions=1,
                actions_vs_baseline_actions=1.0,
                score=0.0,
                levels_completed=0,
                source_path="fixture",
            ),
            preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
            offline_validation={"passed": True},
            environment_count=1,
            duration_s=0.0,
        )
    blocked_via_builder = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=ArcLivePreconditions(False, "missing", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=0,
        duration_s=0.0,
    )
    assert blocked_via_builder["honest_verdict"] == "blocked_arc_live_unreachable"


def test_scenario_phase4_060_run_paths_and_write(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-PHASE4-060: run writes blocked, success, and live-error artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    exp._write_artifact({"honest_verdict": "complete: fixture"})
    assert (tmp_path / "results" / exp.RESULT_NAME).exists()

    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=exp.BASE_URL: ArcLivePreconditions(False, "missing", True, base_url),
    )
    blocked = exp.run(write=True)
    assert blocked["honest_verdict"] == "blocked_arc_live_unreachable"

    floor = _floor()
    outcome = run_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=exp.BASE_URL: ArcLivePreconditions(True, "0.9.8", True, base_url),
    )
    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": True})
    monkeypatch.setattr(exp, "load_exp4202_floor", lambda path=None: floor)
    monkeypatch.setattr(exp, "load_completion_solver_plan", lambda path=None: _solver_plan())
    monkeypatch.setattr(exp, "open_online_arcade", lambda base_url=exp.BASE_URL: object())
    monkeypatch.setattr(
        exp,
        "run_live_solver_accuracy",
        lambda arcade, floor, solver_plan, action_budget: (25, outcome),
    )

    success = exp.run(write=True, action_budget=17)
    assert success["solver_completes_level"] is True
    assert success["environment_count"] == 25

    monkeypatch.setattr(
        exp,
        "run_live_solver_accuracy",
        lambda arcade, floor, solver_plan, action_budget: (_ for _ in ()).throw(RuntimeError("live down")),
    )
    failed_live = exp.run(write=False, action_budget=17)
    assert failed_live["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "live_solver_error=RuntimeError" in failed_live["preconditions_checked"]["error"]

    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": False})
    validation_failed = exp.run(write=False, action_budget=17)
    assert validation_failed["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "recorded fixture adapter validation failed" in validation_failed["preconditions_checked"]["error"]
