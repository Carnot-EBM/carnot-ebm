"""Tests for Exp 4202 ARC-AGI-3 live solver-vs-floor probe.

Spec refs: REQ-PHASE4-058, SCENARIO-PHASE4-058.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import carnot.experiment_4202_arc_live_env_solver_vs_floor as exp
from carnot.agentic.arc_agi3_live_adapter import ArcLivePreconditions, EnvironmentSummary
from carnot.experiment_4202_arc_live_env_solver_vs_floor import (
    LP85_GAME_ID,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    FloorBaseline,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    compare_solver_to_floor,
    extract_banked_lp85_l1_plan,
    extract_floor_baseline,
    load_banked_solver_plan,
    load_floor_baseline,
    run_solver_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeActionEnum:
    ACTION6 = SimpleNamespace(value=6, name="ACTION6")

    @staticmethod
    def from_id(action_id: int) -> SimpleNamespace:
        return SimpleNamespace(value=int(action_id), name=f"ACTION{int(action_id)}")


class FakeEnv:
    scorecard_id = "open-scorecard-fixture"

    def __init__(self, advance_after: int = 5) -> None:
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
            guid="solver-fixture-guid",
        )


class FakeScoreProvider:
    score_source = "fixture_open_scorecard"

    def __call__(self, env: FakeEnv) -> SimpleNamespace:
        return SimpleNamespace(
            guid="solver-fixture-guid",
            score=100.0 if env._level else 0.0,
            levels_completed=env._level,
            actions=len(env.actions),
            resets=1,
            completed=False,
            level_actions=[len(env.actions)],
            level_baseline_actions=[17],
            message="",
        )


def _floor_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4191_arc_live_env_grounding_probe",
        "random_greedy_baseline": {
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
            "environment_score": {"score": 0.0, "levels_completed": 0, "actions": 0},
            "trace": [],
            "scorecard_id": "floor-open-scorecard",
            "leaderboard_submission_attempted": False,
        },
    }


def _prior_solver_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4190_arc_incremental_progress",
        "target_game": LP85_GAME_ID,
        "real_env_confirmed": True,
        "verifier_validated": True,
        "phase_trace": [
            {"phase": "observe", "levels_completed": 0},
            {"phase": "replay", "source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0},
            {"phase": "replay", "source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0},
            {"phase": "replay", "source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0},
            {"phase": "replay", "source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0},
            {"phase": "replay", "source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 1},
            {
                "phase": "hardened-gap4-verify",
                "verifier": "hardened_gap4_heldout_executed_consistency_deeper_level_replay",
                "retained": True,
            },
        ],
    }


def test_req_phase4_058_spec_declares_exp4202_contract() -> None:
    """REQ-PHASE4-058: OpenSpec declares the live solver-vs-floor artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-058" in spec
    assert "SCENARIO-PHASE4-058" in spec
    assert "experiment_4202_arc_live_env_solver_vs_floor.json" in spec
    assert "blocked_arc_live_unreachable" in spec
    assert "solver_beats_floor" in spec
    assert "live_env_metrics" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_058_blocked_artifact_is_terminal_and_non_submitting() -> None:
    """SCENARIO-PHASE4-058: missing SDK/network stops before live contact."""

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
    assert artifact["solver_beats_floor"] == {}
    assert artifact["live_env_metrics"] == {}
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["leaderboard_submission_attempted"] is False
    assert artifact["requirements"] == REQUIREMENTS
    assert artifact_schema_errors(artifact) == []


def test_req_phase4_058_loads_same_floor_and_banked_solver_plan() -> None:
    """REQ-PHASE4-058: Exp 4202 compares against Exp 4191 on the same LP85 env."""

    floor = extract_floor_baseline(_floor_artifact())
    plan = extract_banked_lp85_l1_plan(_prior_solver_artifact())

    assert floor.environment.game_id == LP85_GAME_ID
    assert floor.actions_taken == 6
    assert floor.baseline_actions == 17
    assert floor.score == 0.0
    assert len(plan) == 5
    assert plan[-1]["expected_levels_completed"] == 1
    assert {step["x"] for step in plan} == {4}
    assert {step["y"] for step in plan} == {32}

    with pytest.raises(ValueError, match="random_greedy_baseline"):
        extract_floor_baseline({})
    with pytest.raises(ValueError, match="same LP85 environment"):
        extract_floor_baseline({"random_greedy_baseline": {"environment": {"game_id": "other"}, "actions_taken": 1}})
    with pytest.raises(ValueError, match="banked LP85 L1 replay"):
        extract_banked_lp85_l1_plan({"phase_trace": []})
    with pytest.raises(ValueError, match="does not confirm L1"):
        extract_banked_lp85_l1_plan(
            {
                "phase_trace": [
                    {"source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 0}
                ]
            }
        )


def test_scenario_phase4_058_solver_replay_compares_accuracy_and_efficiency() -> None:
    """SCENARIO-PHASE4-058: live metrics and floor comparison are schema-stable."""

    floor = FloorBaseline(
        environment=EnvironmentSummary(LP85_GAME_ID, "LP85", ["click"], [17, 38]),
        actions_taken=6,
        baseline_actions=17,
        actions_vs_baseline_actions=6 / 17,
        score=0.0,
        levels_completed=0,
        source_path="results/experiment_4191_arc_live_env_grounding_probe.json",
    )
    outcome = run_solver_replay(
        FakeEnv(),
        floor.environment,
        solver_plan=extract_banked_lp85_l1_plan(_prior_solver_artifact()),
        action_budget=6,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    comparison = compare_solver_to_floor(outcome, floor)

    assert outcome.actions_taken == 5
    assert outcome.baseline_actions == 17
    assert outcome.score.score == 100.0
    assert outcome.score.levels_completed == 1
    assert outcome.actions_vs_baseline_actions == 5 / 17
    assert [step["action"]["data"] for step in outcome.trace] == [{"x": 4, "y": 32}] * 5
    assert comparison["accuracy"]["beats"] is True
    assert comparison["efficiency"]["beats"] is True
    assert comparison["overall"] is True

    artifact = build_artifact(
        outcome=outcome,
        floor=floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "success: solver_beats_floor_live_lp85-305b61c3_accuracy_and_efficiency"
    assert artifact["live_env_metrics"]["score"] == 100.0
    assert artifact["live_env_metrics"]["levels_completed"] == 1
    assert artifact["live_env_metrics"]["actions_taken"] == 5
    assert artifact["live_env_metrics"]["baseline_actions"] == 17
    assert artifact["solver_beats_floor"] == comparison
    assert artifact["no_leaderboard_submission"] is True
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_058_no_beat_verdict_and_schema_rejections() -> None:
    """SCENARIO-PHASE4-058: no-win outcomes remain complete and malformed claims fail."""

    floor = FloorBaseline(
        environment=EnvironmentSummary(LP85_GAME_ID, "LP85", ["click"], [17]),
        actions_taken=3,
        baseline_actions=17,
        actions_vs_baseline_actions=3 / 17,
        score=100.0,
        levels_completed=1,
        source_path="fixture",
    )
    outcome = run_solver_replay(
        FakeEnv(advance_after=99),
        floor.environment,
        solver_plan=extract_banked_lp85_l1_plan(_prior_solver_artifact()),
        action_budget=5,
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

    assert artifact["honest_verdict"] == "complete: solver_does_not_beat_floor_live_lp85-305b61c3"
    assert artifact["solver_beats_floor"]["overall"] is False

    bad = {
        "honest_verdict": "maybe",
        "solver_beats_floor": [],
        "live_env_metrics": [],
        "no_leaderboard_submission": False,
        "preconditions_checked": [],
        "leaderboard_submission_attempted": True,
        "requirements": [],
        "field_principles": [],
    }
    errors = artifact_schema_errors(bad)
    assert any("honest_verdict must be terminal-prefixed" in err for err in errors)
    assert any("solver_beats_floor must be a dict" in err for err in errors)
    assert any("live_env_metrics must be a dict" in err for err in errors)
    assert any("no_leaderboard_submission must be true" in err for err in errors)
    assert any("leaderboard_submission_attempted must be false" in err for err in errors)
    assert any("preconditions_checked must be a dict" in err for err in errors)
    assert any("requirements must include" in err for err in errors)
    assert any("field_principles must be a dict" in err for err in errors)

    missing_live = {**artifact, "live_env_metrics": {"score": 0.0}}
    assert any("live_env_metrics missing levels_completed" in err for err in artifact_schema_errors(missing_live))
    blocked_bad = {**blocked_artifact(preconditions=ArcLivePreconditions(True, "0.9.8", True, "url"), duration_s=0), "live_env_metrics": {"score": 1}}
    assert any("blocked artifacts must leave live_env_metrics empty" in err for err in artifact_schema_errors(blocked_bad))


def test_req_phase4_058_helper_edges_and_schema_defenses(tmp_path: Path) -> None:
    """REQ-PHASE4-058: helper edge paths stay deterministic and auditable."""

    floor_path = tmp_path / "floor.json"
    floor_path.write_text(__import__("json").dumps(_floor_artifact()), encoding="utf-8")
    solver_path = tmp_path / "solver.json"
    solver_path.write_text(__import__("json").dumps(_prior_solver_artifact()), encoding="utf-8")

    assert load_floor_baseline(floor_path).source_path == str(floor_path)
    assert len(load_banked_solver_plan(solver_path)) == 5
    assert exp._read_json(floor_path)["experiment"] == "experiment_4191_arc_live_env_grounding_probe"
    assert "Exp 4191 floor missing environment" in str(
        pytest.raises(ValueError, extract_floor_baseline, {"random_greedy_baseline": {}}).value
    )

    fallback_plan = extract_banked_lp85_l1_plan({"solve_trace": {"phase_trace": _prior_solver_artifact()["phase_trace"]}})
    assert len(fallback_plan) == 5
    malformed = {
        "phase_trace": [
            {"source": "banked_lp85_L1_replay", "x": 4, "levels_completed": 1},
            {"source": "banked_lp85_L1_replay", "x": 4, "y": 32, "levels_completed": 1},
        ]
    }
    assert extract_banked_lp85_l1_plan(malformed) == [
        {
            "action_id": 6,
            "x": 4,
            "y": 32,
            "source": "banked_lp85_L1_replay",
            "expected_levels_completed": 1,
        }
    ]

    class ResetNoneEnv:
        def reset(self) -> None:
            return None

    with pytest.raises(ValueError, match="reset returned no frame"):
        run_solver_replay(
            ResetNoneEnv(),
            EnvironmentSummary(LP85_GAME_ID, "LP85", ["click"], [17]),
            solver_plan=fallback_plan,
            action_enum=FakeActionEnum,
        )

    class StepNoneEnv(FakeEnv):
        def step(
            self,
            action: object,
            data: dict[str, int] | None = None,
            reasoning: dict | None = None,
        ) -> None:
            del action, data, reasoning
            self.actions.append(None)
            return None

    step_none = run_solver_replay(
        StepNoneEnv(),
        EnvironmentSummary(LP85_GAME_ID, "LP85", ["click"], [17]),
        solver_plan=fallback_plan,
        action_enum=FakeActionEnum,
    )
    assert step_none.trace[-1]["event"] == "step_returned_no_frame"
    assert step_none.score_source == "local_adapter_fallback"

    assert exp._verdict_from_comparison({"accuracy": {"beats": True}, "efficiency": {"beats": False}}, LP85_GAME_ID).endswith("accuracy_only")
    assert exp._verdict_from_comparison({"accuracy": {"beats": False}, "efficiency": {"beats": True}}, LP85_GAME_ID).endswith("efficiency_only")

    good_floor = extract_floor_baseline(_floor_artifact())
    good_outcome = run_solver_replay(
        FakeEnv(),
        good_floor.environment,
        solver_plan=fallback_plan,
        action_budget=6,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    blocked_via_builder = build_artifact(
        outcome=good_outcome,
        floor=good_floor,
        preconditions=ArcLivePreconditions(False, "missing", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=0,
        duration_s=0.0,
    )
    assert blocked_via_builder["honest_verdict"] == "blocked_arc_live_unreachable"

    with pytest.raises(ValueError, match="same environment"):
        build_artifact(
            outcome=good_outcome,
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

    success = build_artifact(
        outcome=good_outcome,
        floor=good_floor,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.0,
    )
    malformed_artifact = {
        **success,
        "solver_beats_floor": {"accuracy": {}, "efficiency": {}, "overall": "true"},
        "live_env_metrics": {
            "score": "bad",
            "levels_completed": "1",
            "actions_taken": "5",
            "baseline_actions": "17",
        },
        "offline_validation": {"passed": False},
        "real_metric_mapping": {},
        "preconditions_checked": {"sdk_importable": "yes", "network_reachable": "yes", "base_url": "url"},
        "field_principles": {"honest_verdict": "present"},
    }
    errors = artifact_schema_errors(malformed_artifact)
    assert any("solver_beats_floor.overall must be a bare bool" in err for err in errors)
    assert any(
        "solver_beats_floor missing efficiency" in err
        for err in artifact_schema_errors({**success, "solver_beats_floor": {"accuracy": {}, "overall": True}})
    )
    assert any("live_env_metrics.levels_completed must be a bare int" in err for err in errors)
    assert any("live_env_metrics.score must be numeric" in err for err in errors)
    assert any("reachable artifacts require passed offline_validation" in err for err in errors)
    assert any("real_metric_mapping must equal" in err for err in errors)
    assert any("preconditions_checked missing sdk_version" in err for err in errors)
    assert any("preconditions_checked.sdk_importable must be a bare bool" in err for err in errors)
    assert any("field_principles missing solver_beats_floor" in err for err in errors)
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({**success, "honest_verdict": 4202}))
    assert any(
        "blocked artifacts must leave solver_beats_floor empty" in err
        for err in artifact_schema_errors(
            {
                **blocked_artifact(
                    preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
                    duration_s=0.0,
                ),
                "solver_beats_floor": {"overall": False},
            }
        )
    )


def test_scenario_phase4_058_run_paths_and_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-058: run writes blocked, success, and live-error artifacts."""

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

    floor = extract_floor_baseline(_floor_artifact(), source_path="fixture-floor.json")
    outcome = run_solver_replay(
        FakeEnv(),
        floor.environment,
        solver_plan=extract_banked_lp85_l1_plan(_prior_solver_artifact()),
        action_budget=6,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )
    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=exp.BASE_URL: ArcLivePreconditions(True, "0.9.8", True, base_url),
    )
    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": True})
    monkeypatch.setattr(exp, "load_floor_baseline", lambda path=None: floor)
    monkeypatch.setattr(exp, "load_banked_solver_plan", lambda path=None: extract_banked_lp85_l1_plan(_prior_solver_artifact()))
    monkeypatch.setattr(exp, "open_online_arcade", lambda base_url=exp.BASE_URL: object())
    monkeypatch.setattr(
        exp,
        "run_live_solver_vs_floor",
        lambda arcade, floor, solver_plan, action_budget: (25, outcome),
    )
    success = exp.run(write=True, action_budget=6)
    assert success["solver_beats_floor"]["overall"] is True
    assert success["environment_count"] == 25

    monkeypatch.setattr(
        exp,
        "run_live_solver_vs_floor",
        lambda arcade, floor, solver_plan, action_budget: (_ for _ in ()).throw(RuntimeError("live down")),
    )
    failed_live = exp.run(write=False, action_budget=6)
    assert failed_live["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "live_solver_error=RuntimeError" in failed_live["preconditions_checked"]["error"]

    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": False})
    validation_failed = exp.run(write=False, action_budget=6)
    assert validation_failed["honest_verdict"] == "blocked_arc_live_unreachable"
