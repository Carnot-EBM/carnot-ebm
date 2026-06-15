"""Tests for Exp 4237 ARC-AGI-3 live solver accuracy with margin trigger.

Spec refs: REQ-PHASE4-064, SCENARIO-PHASE4-064.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import carnot.experiment_4237_arc_live_env_solver_accuracy as exp
from carnot.agentic.arc_agi3_live_adapter import ArcLivePreconditions
from carnot.experiment_4202_arc_live_env_solver_vs_floor import FloorBaseline
from carnot.experiment_4237_arc_live_env_solver_accuracy import (
    LP85_GAME_ID,
    MARGIN_TRIGGER_THRESHOLD,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    MarginTriggeredOverrideConfig,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    extract_exp4225_floor,
    margin_triggered_override_decision,
    route_plan_through_margin_trigger,
    run_margin_triggered_solver_completion,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeActionEnum:
    ACTION6 = SimpleNamespace(value=6, name="ACTION6")

    @staticmethod
    def from_id(action_id: int) -> SimpleNamespace:
        return SimpleNamespace(value=int(action_id), name=f"ACTION{int(action_id)}")


class FakeEnv:
    scorecard_id = "margin-open-scorecard"

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
        assert reasoning is not None
        assert reasoning["policy"] == exp.MARGIN_TRIGGERED_POLICY_NAME
        assert reasoning["margin_triggered_override"]["commit_induced_rule"] is True
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
            guid="margin-fixture-guid",
        )


class FakeNoResetEnv(FakeEnv):
    def reset(self) -> None:
        return None


class FakeNoneFrameEnv(FakeEnv):
    def step(
        self,
        action: object,
        data: dict[str, int] | None = None,
        reasoning: dict | None = None,
    ) -> None:
        assert reasoning is not None
        assert int(getattr(action, "value", action)) == 6
        self.actions.append(data)
        return None


class FakeScoreProvider:
    score_source = "fixture_open_scorecard"

    def __call__(self, env: FakeEnv) -> SimpleNamespace:
        completed = int(env._level)
        return SimpleNamespace(
            guid="margin-fixture-guid",
            score=125.0 if completed else 0.0,
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


def _exp4225_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4225_arc_live_env_solver_accuracy",
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
            "source_path": "results/experiment_4214_arc_live_env_solver_accuracy.json",
        },
    }


def _floor() -> FloorBaseline:
    return extract_exp4225_floor(
        _exp4225_artifact(),
        source_path="results/experiment_4225_arc_live_env_solver_accuracy.json",
    )


def _preconditions(ok: bool = True) -> ArcLivePreconditions:
    return ArcLivePreconditions(ok, "0.9.8" if ok else "missing", True, "https://three.arcprize.org")


def test_req_phase4_064_spec_declares_exp4237_contract() -> None:
    """REQ-PHASE4-064: OpenSpec declares the margin-triggered live probe."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-064" in spec
    assert "SCENARIO-PHASE4-064" in spec
    assert "experiment_4237_arc_live_env_solver_accuracy.json" in spec
    assert "margin-triggered override keyed to 2606.04323" in spec
    assert "Exp 4225's same-env random/greedy floor" in spec
    assert "blocked_arc_live_unreachable" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_064_margin_trigger_commits_only_above_threshold() -> None:
    """SCENARIO-PHASE4-064: low-margin evidence keeps exploring."""

    committed = margin_triggered_override_decision(
        MarginTriggeredOverrideConfig(learned_margin=0.20, verifier_margin=0.25)
    )
    refused = margin_triggered_override_decision(
        MarginTriggeredOverrideConfig(learned_margin=0.20, verifier_margin=0.05)
    )

    assert committed["commit_induced_rule"] is True
    assert committed["fallback_policy"] == "execute_verified_policy"
    assert committed["margin_threshold"] == pytest.approx(MARGIN_TRIGGER_THRESHOLD)
    assert "2606.04323" in committed["references"]
    assert refused["commit_induced_rule"] is False
    assert refused["fallback_policy"] == "continue_exploring"

    routed, decision = route_plan_through_margin_trigger(_solver_plan(), MarginTriggeredOverrideConfig())
    assert len(routed) == 5
    assert decision["commit_induced_rule"] is True
    assert routed[0]["margin_triggered_override"]["commit_induced_rule"] is True
    assert routed[0]["source"].endswith("_margin_trigger_verified")

    low_margin_plan, low_margin = route_plan_through_margin_trigger(
        _solver_plan(),
        MarginTriggeredOverrideConfig(learned_margin=0.01, verifier_margin=0.20),
    )
    assert low_margin_plan == []
    assert low_margin["commit_induced_rule"] is False


def test_req_phase4_064_loads_exp4225_same_env_floor_and_defends_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-064: the live probe inherits Exp 4225's same-env floor."""

    floor_path = tmp_path / "floor.json"
    floor_path.write_text(json.dumps(_exp4225_artifact()), encoding="utf-8")
    monkeypatch.setattr(exp, "REPO", tmp_path)

    loaded = exp.load_exp4225_floor(floor_path)
    assert loaded.environment.game_id == LP85_GAME_ID
    assert loaded.actions_taken == 6
    assert loaded.baseline_actions == 17
    assert loaded.source_path == "floor.json"

    outside_path = tmp_path.parent / "outside-floor-4237.json"
    outside_path.write_text(json.dumps(_exp4225_artifact()), encoding="utf-8")
    assert exp.load_exp4225_floor(outside_path).source_path == str(outside_path)

    with pytest.raises(ValueError, match="random_greedy_floor"):
        extract_exp4225_floor({})
    with pytest.raises(ValueError, match="floor missing environment"):
        extract_exp4225_floor({"random_greedy_floor": {}})
    with pytest.raises(ValueError, match="same LP85 environment"):
        extract_exp4225_floor({"random_greedy_floor": {"environment": {"game_id": "other-env"}}})


def test_scenario_phase4_064_solver_completion_is_accuracy_win_with_margin_trigger() -> None:
    """SCENARIO-PHASE4-064: scorecard level completion is the bare accuracy gate."""

    floor = _floor()
    outcome, margin = run_margin_triggered_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=2,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
        margin_config=MarginTriggeredOverrideConfig(),
    )
    artifact = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
        margin_triggered_override=margin,
    )

    assert outcome.action_budget == 17
    assert outcome.actions_taken == 5
    assert artifact["honest_verdict"] == "success: solver_completes_level_live_lp85-305b61c3"
    assert artifact["solver_completes_level"] is True
    assert artifact["live_env_metrics"]["levels_completed"] == 1
    assert artifact["live_env_metrics"]["score"] == 125.0
    assert artifact["solver_beats_floor"]["accuracy"]["beats"] is True
    assert artifact["solver_beats_floor"]["efficiency"]["beats"] is True
    assert artifact["margin_triggered_override"]["commit_induced_rule"] is True
    assert artifact["solver_policy"] == exp.MARGIN_TRIGGERED_POLICY_NAME
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["scorecard_closed"] is False
    assert artifact_schema_errors(artifact) == []

    local_score_outcome, local_score_margin = run_margin_triggered_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        margin_config=MarginTriggeredOverrideConfig(),
    )
    assert local_score_outcome.score_source == "local_adapter_fallback"
    assert local_score_outcome.score.levels_completed == 1
    assert local_score_margin["commit_induced_rule"] is True

    none_frame_outcome, _ = run_margin_triggered_solver_completion(
        FakeNoneFrameEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        margin_config=MarginTriggeredOverrideConfig(),
    )
    assert none_frame_outcome.trace[0]["event"] == "step_returned_no_frame"

    with pytest.raises(ValueError, match="refused to commit"):
        run_margin_triggered_solver_completion(
            FakeEnv(),
            floor.environment,
            floor=floor,
            solver_plan=_solver_plan(),
            requested_action_budget=17,
            action_enum=FakeActionEnum,
            margin_config=MarginTriggeredOverrideConfig(learned_margin=0.01, verifier_margin=0.20),
        )
    with pytest.raises(ValueError, match="reset returned no frame"):
        run_margin_triggered_solver_completion(
            FakeNoResetEnv(),
            floor.environment,
            floor=floor,
            solver_plan=_solver_plan(),
            requested_action_budget=17,
            action_enum=FakeActionEnum,
            margin_config=MarginTriggeredOverrideConfig(),
        )


def test_scenario_phase4_064_refusal_and_schema_edges() -> None:
    """SCENARIO-PHASE4-064: schema preserves the no-submission accuracy contract."""

    floor = _floor()
    refused = margin_triggered_override_decision(
        MarginTriggeredOverrideConfig(learned_margin=0.01, verifier_margin=0.20)
    )
    refused_artifact = exp.margin_trigger_refused_artifact(
        environment=floor.environment,
        floor=floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
        margin_triggered_override=refused,
    )
    assert refused_artifact["honest_verdict"] == "complete: margin_triggered_override_kept_exploring_lp85-305b61c3"
    assert refused_artifact["solver_completes_level"] is False
    assert refused_artifact["live_env_metrics"]["actions_taken"] == 0
    assert refused_artifact["solver_beats_floor"]["accuracy"]["beats"] is False
    assert refused_artifact["solver_beats_floor"]["efficiency"]["beats"] is False
    assert artifact_schema_errors(refused_artifact) == []

    blocked = blocked_artifact(preconditions=_preconditions(False), duration_s=0.25)
    assert blocked["honest_verdict"] == "blocked_arc_live_unreachable"
    assert blocked["solver_completes_level"] is False
    assert blocked["solver_beats_floor"] == {}
    assert blocked["live_env_metrics"] == {}
    assert blocked["requirements"] == REQUIREMENTS
    assert artifact_schema_errors(blocked) == []

    outcome, margin = run_margin_triggered_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
        margin_config=MarginTriggeredOverrideConfig(),
    )
    good = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
        margin_triggered_override=margin,
    )
    bad = {
        **good,
        "honest_verdict": 4237,
        "solver_completes_level": "true",
        "solver_beats_floor": [],
        "live_env_metrics": {"score": "bad", "levels_completed": "1", "actions_taken": "5", "baseline_actions": "17", "action_budget": 6},
        "margin_triggered_override": {"commit_induced_rule": "yes"},
        "no_leaderboard_submission": False,
        "leaderboard_submission_attempted": True,
        "scorecard_closed": True,
        "preconditions_checked": {"sdk_importable": "yes", "network_reachable": "yes", "base_url": "url"},
        "requirements": [],
        "field_principles": [],
        "offline_validation": {"passed": False},
        "real_metric_mapping": {},
    }
    errors = artifact_schema_errors(bad)
    assert "honest_verdict must be a string" in errors
    assert "solver_completes_level must be a bare bool" in errors
    assert "solver_beats_floor must be a dict" in errors
    assert "live_env_metrics.score must be numeric" in errors
    assert "live_env_metrics.levels_completed must be a bare int" in errors
    assert "margin_triggered_override.commit_induced_rule must be a bare bool" in errors
    assert "no_leaderboard_submission must be true" in errors
    assert "leaderboard_submission_attempted must be false" in errors
    assert "scorecard_closed must be false" in errors
    assert "preconditions_checked missing sdk_version" in errors
    assert "preconditions_checked.sdk_importable must be a bare bool" in errors
    assert "requirements must include REQ-PHASE4-064 and SCENARIO-PHASE4-064" in errors
    assert "field_principles must be a dict" in errors
    assert "reachable artifacts require passed offline_validation" in errors
    assert "real_metric_mapping must equal the ARC live EnvironmentScore mapping" in errors
    assert "missing required field honest_verdict" in artifact_schema_errors({})
    assert "honest_verdict must be terminal-prefixed" in artifact_schema_errors({**good, "honest_verdict": "maybe"})
    assert "blocked artifacts must leave solver_beats_floor empty" in artifact_schema_errors(
        {**blocked, "solver_beats_floor": {"overall": False}}
    )
    assert "blocked artifacts must leave live_env_metrics empty" in artifact_schema_errors(
        {**blocked, "live_env_metrics": {"score": 0.0}}
    )
    assert "field_principles missing solver_completes_level" in artifact_schema_errors(
        {**good, "field_principles": {"honest_verdict": "present"}}
    )
    assert "solver_beats_floor missing efficiency" in artifact_schema_errors(
        {**good, "solver_beats_floor": {"accuracy": {}, "overall": True}}
    )
    assert "solver_beats_floor.overall must be a bare bool" in artifact_schema_errors(
        {**good, "solver_beats_floor": {"accuracy": {}, "efficiency": {}, "overall": "true"}}
    )
    assert "live_env_metrics missing action_budget" in artifact_schema_errors(
        {**good, "live_env_metrics": {"score": 1.0, "levels_completed": 1, "actions_taken": 5, "baseline_actions": 17}}
    )
    assert "live_env_metrics must be a dict" in artifact_schema_errors({**good, "live_env_metrics": []})
    assert "live_env_metrics.action_budget must be >= baseline_actions" in artifact_schema_errors(
        {**good, "live_env_metrics": {**good["live_env_metrics"], "action_budget": 1}}
    )
    assert "margin_triggered_override must be a dict" in artifact_schema_errors({**good, "margin_triggered_override": []})
    assert "solver_completes_level must equal live_env_metrics.levels_completed>=1" in artifact_schema_errors(
        {**good, "solver_completes_level": False}
    )

    zero_outcome, zero_margin = run_margin_triggered_solver_completion(
        FakeEnv(advance_after=99),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
        margin_config=MarginTriggeredOverrideConfig(),
    )
    zero = build_artifact(
        outcome=zero_outcome,
        floor=floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
        margin_triggered_override=zero_margin,
    )
    assert zero["honest_verdict"] == "complete: solver_completes_0_levels_live_lp85-305b61c3_efficiency_only"

    score_only_floor = FloorBaseline(floor.environment, 5, 17, 5 / 17, -1.0, 0, "fixture")
    score_only = build_artifact(
        outcome=zero_outcome,
        floor=score_only_floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
        margin_triggered_override=zero_margin,
    )
    assert score_only["honest_verdict"] == "complete: solver_score_beats_floor_without_completion_live_lp85-305b61c3"

    no_beat_floor = FloorBaseline(floor.environment, 5, 17, 5 / 17, 0.0, 0, "fixture")
    no_beat = build_artifact(
        outcome=zero_outcome,
        floor=no_beat_floor,
        preconditions=_preconditions(),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
        margin_triggered_override=zero_margin,
    )
    assert no_beat["honest_verdict"] == "complete: solver_completes_0_levels_live_lp85-305b61c3"

    blocked_via_builder = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=_preconditions(False),
        offline_validation={"passed": True},
        environment_count=0,
        duration_s=0.0,
        margin_triggered_override=margin,
    )
    assert blocked_via_builder["honest_verdict"] == "blocked_arc_live_unreachable"
    assert blocked_via_builder["offline_validation"] == {"passed": True}

    with pytest.raises(ValueError, match="same environment"):
        build_artifact(
            outcome=outcome,
            floor=FloorBaseline(floor.environment.__class__("other-env", "OTHER", [], [1]), 1, 1, 1.0, 0.0, 0, "fixture"),
            preconditions=_preconditions(),
            offline_validation={"passed": True},
            environment_count=1,
            duration_s=0.0,
            margin_triggered_override=margin,
        )


def test_scenario_phase4_064_run_paths_and_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-064: run writes blocked, success, refusal, and live-error artifacts."""

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
    outcome, margin = run_margin_triggered_solver_completion(
        FakeEnv(),
        floor.environment,
        floor=floor,
        solver_plan=_solver_plan(),
        requested_action_budget=17,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
        margin_config=MarginTriggeredOverrideConfig(),
    )
    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=exp.BASE_URL: ArcLivePreconditions(True, "0.9.8", True, base_url),
    )
    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": True})
    monkeypatch.setattr(exp, "load_exp4225_floor", lambda path=None: floor)
    monkeypatch.setattr(exp, "load_completion_solver_plan", lambda path=None: _solver_plan())
    monkeypatch.setattr(exp, "open_online_arcade", lambda base_url=exp.BASE_URL: object())
    monkeypatch.setattr(
        exp,
        "run_live_margin_triggered_solver_accuracy",
        lambda arcade, floor, solver_plan, action_budget, margin_config: (25, outcome, margin),
    )

    success = exp.run(write=True, action_budget=17)
    assert success["solver_completes_level"] is True
    assert success["environment_count"] == 25

    low_margin = exp.run(
        write=False,
        margin_config=MarginTriggeredOverrideConfig(learned_margin=0.01, verifier_margin=0.20),
    )
    assert low_margin["honest_verdict"] == "complete: margin_triggered_override_kept_exploring_lp85-305b61c3"

    monkeypatch.setattr(
        exp,
        "run_live_margin_triggered_solver_accuracy",
        lambda arcade, floor, solver_plan, action_budget, margin_config: (_ for _ in ()).throw(RuntimeError("live down")),
    )
    failed_live = exp.run(write=False, action_budget=17)
    assert failed_live["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "live_solver_error=RuntimeError" in failed_live["preconditions_checked"]["error"]

    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": False})
    validation_failed = exp.run(write=False, action_budget=17)
    assert validation_failed["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "recorded fixture adapter validation failed" in validation_failed["preconditions_checked"]["error"]
