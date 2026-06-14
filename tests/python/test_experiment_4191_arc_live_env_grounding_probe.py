"""Tests for Exp 4191 ARC-AGI-3 live-env grounding probe.

Spec refs: REQ-PHASE4-056, SCENARIO-PHASE4-056.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

import carnot.agentic.arc_agi3_live_adapter as adapter
import carnot.experiment_4191_arc_live_env_grounding_probe as exp
from carnot.agentic.arc_agi3_live_adapter import (
    ArcAction,
    ArcLivePreconditions,
    EnvironmentSummary,
    LiveProbeOutcome,
    MetricMapping,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    choose_probe_environment,
    run_random_greedy_baseline,
    validate_recorded_fixture,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeActionEnum:
    RESET = SimpleNamespace(value=0, name="RESET")

    @staticmethod
    def from_id(action_id: int) -> SimpleNamespace:
        return SimpleNamespace(value=int(action_id), name=f"ACTION{int(action_id)}")


class FakeEnv:
    def __init__(self) -> None:
        self.scorecard_id = "scorecard-fixture"
        self._level = 0
        self._index = 0
        self.actions: list[tuple[int, dict[str, int] | None]] = []
        self.frames = [
            np.zeros((3, 3), dtype=np.int16),
            np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.int16),
            np.array([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=np.int16),
        ]

    def reset(self) -> SimpleNamespace:
        self._level = 0
        self._index = 0
        self.actions.clear()
        return self._frame()

    def step(self, action: object, data: dict[str, int] | None = None, reasoning: dict | None = None) -> SimpleNamespace:
        del reasoning
        action_id = int(getattr(action, "value", action))
        self.actions.append((action_id, data))
        self._index = min(self._index + 1, 2)
        if action_id == 2:
            self._level = 1
        return self._frame()

    def _frame(self) -> SimpleNamespace:
        return SimpleNamespace(
            frame=self.frames[self._index],
            levels_completed=self._level,
            available_actions=[1, 2],
            state="PLAYING",
            guid="guid-fixture",
        )


class FakeScoreProvider:
    def __call__(self, env: FakeEnv) -> object:
        return SimpleNamespace(
            id="lp85-305b61c3",
            guid="guid-fixture",
            score=12.5,
            levels_completed=env._level,
            actions=len(env.actions),
            resets=1,
            completed=False,
            level_actions=[len(env.actions)],
            level_baseline_actions=[4],
            level_scores=[12.5],
            message=None,
        )


def test_req_phase4_056_spec_declares_exp4191_contract() -> None:
    """REQ-PHASE4-056: OpenSpec declares the live grounding terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-056" in spec
    assert "SCENARIO-PHASE4-056" in spec
    assert "experiment_4191_arc_live_env_grounding_probe.json" in spec
    assert "blocked_arc_live_unreachable" in spec
    assert "EnvironmentScore.score/levels_completed" in spec
    assert "actions_vs_baseline_actions" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_056_blocked_artifact_is_terminal_and_non_submitting() -> None:
    """SCENARIO-PHASE4-056: unreachable SDK/network writes an honest blocked artifact."""

    preconditions = ArcLivePreconditions(
        sdk_importable=False,
        sdk_version="missing",
        network_reachable=True,
        base_url="https://three.arcprize.org",
        error="No module named arc_agi",
    )

    artifact = blocked_artifact(preconditions=preconditions, duration_s=0.25)

    assert artifact["honest_verdict"] == "blocked_arc_live_unreachable"
    assert artifact["live_env_reachable"] is False
    assert artifact["random_greedy_baseline"] == {}
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["preconditions_checked"]["sdk_importable"] is False
    assert artifact["requirements"] == exp.REQUIREMENTS
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_056_recorded_fixture_replays_adapter_path() -> None:
    """SCENARIO-PHASE4-056: the live adapter reproduces recorded offline behavior first."""

    validation = validate_recorded_fixture()

    assert validation["passed"] is True
    assert validation["expected_action_ids"] == [1, 2]
    assert validation["observed_action_ids"] == [1, 2]
    assert validation["score"]["levels_completed"] == 1
    assert validation["score"]["actions"] == 2


def test_scenario_phase4_056_random_greedy_baseline_records_real_mapping_floor() -> None:
    """SCENARIO-PHASE4-056: random/greedy baseline records score and actions-vs-baseline."""

    outcome = run_random_greedy_baseline(
        FakeEnv(),
        EnvironmentSummary(
            game_id="lp85-305b61c3",
            title="LP85",
            tags=["click"],
            baseline_actions=[4],
        ),
        action_budget=2,
        random_seed=4191,
        action_enum=FakeActionEnum,
        score_provider=FakeScoreProvider(),
    )

    assert outcome.environment.game_id == "lp85-305b61c3"
    assert outcome.score.score == 12.5
    assert outcome.score.levels_completed == 1
    assert outcome.actions_taken == 2
    assert outcome.baseline_actions == 4
    assert outcome.actions_vs_baseline_actions == 0.5
    assert [step["action"]["action_id"] for step in outcome.trace] == [1, 2]
    assert outcome.leaderboard_submission_attempted is False


def test_scenario_phase4_056_success_artifact_schema_and_selection() -> None:
    """SCENARIO-PHASE4-056: reachable probe artifacts expose metric mapping and no submission."""

    environments = [
        EnvironmentSummary("wa30-ee6fef47", "WA30", ["keyboard"], [71]),
        EnvironmentSummary("lp85-305b61c3", "LP85", ["click"], [17]),
        EnvironmentSummary("sb26-7fbdac44", "SB26", ["keyboard_click"], [18]),
    ]
    selected = choose_probe_environment(environments)
    assert selected.game_id == "lp85-305b61c3"

    outcome = LiveProbeOutcome(
        environment=selected,
        action_budget=3,
        actions_taken=2,
        baseline_actions=17,
        actions_vs_baseline_actions=2 / 17,
        score=MetricMapping.Score(
            score=0.0,
            levels_completed=0,
            actions=2,
            level_actions=[2],
            level_baseline_actions=[17],
            completed=False,
        ),
        trace=[
            {
                "action_index": 1,
                "action": ArcAction(6, {"x": 4, "y": 32}, "fixture").to_json(),
                "levels_completed_after": 0,
            }
        ],
        scorecard_id="scorecard-open-not-closed",
        score_source="sdk_get_scorecard_open_scorecard",
        anonymous_key_used=True,
        leaderboard_submission_attempted=False,
    )
    artifact = build_artifact(
        outcome=outcome,
        preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=25,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "complete: arc_live_env_reachable_random_greedy_baseline_lp85-305b61c3"
    assert artifact["live_env_reachable"] is True
    assert artifact["real_metric_mapping"] == MetricMapping().to_json()
    assert artifact["random_greedy_baseline"]["score"] == 0.0
    assert artifact["random_greedy_baseline"]["actions_taken"] == 2
    assert artifact["random_greedy_baseline"]["actions_vs_baseline_actions"] == 2 / 17
    assert artifact["no_leaderboard_submission"] is True
    assert artifact_schema_errors(artifact) == []


def test_req_phase4_056_preconditions_and_environment_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PHASE4-056: preconditions and live environment summaries are normalized."""

    class OkResponse:
        status_code = 301

    monkeypatch.setattr(adapter.requests, "get", lambda *args, **kwargs: OkResponse())
    preconditions = adapter.check_live_preconditions(base_url="https://three.arcprize.org")
    assert preconditions.sdk_importable is True
    assert preconditions.network_reachable is True
    assert preconditions.ok is True

    monkeypatch.setattr(
        adapter.importlib.metadata,
        "version",
        lambda package: (_ for _ in ()).throw(adapter.importlib.metadata.PackageNotFoundError),
    )
    unknown_version = adapter.check_live_preconditions(base_url="https://three.arcprize.org")
    assert unknown_version.sdk_version == "version_unknown"

    def raise_network(*args: object, **kwargs: object) -> object:
        raise requests.RequestException("down")

    monkeypatch.setattr(adapter.requests, "get", raise_network)
    blocked = adapter.check_live_preconditions(base_url="https://three.arcprize.org")
    assert blocked.sdk_importable is True
    assert blocked.network_reachable is False
    assert "network_error" in blocked.error

    info = SimpleNamespace(
        game_id="aa00-123",
        title="AA00",
        tags=["keyboard", 7],
        baseline_actions=[3, "5"],
    )
    summary = EnvironmentSummary.from_info(info)
    assert summary.to_json() == {
        "game_id": "aa00-123",
        "title": "AA00",
        "tags": ["keyboard", "7"],
        "baseline_actions": [3, 5],
    }
    assert adapter.enumerate_live_environments(
        SimpleNamespace(get_environments=lambda: [info, SimpleNamespace(game_id="")])
    ) == [summary]
    assert adapter._quiet_logger().propagate is False

    with pytest.raises(ValueError, match="no live ARC-AGI-3 environments"):
        choose_probe_environment([])
    assert choose_probe_environment(
        [
            EnvironmentSummary("zz99-1", "ZZ99", [], [44]),
            EnvironmentSummary("aa00-1", "AA00", [], [9]),
        ]
    ).game_id == "aa00-1"


def test_scenario_phase4_056_action_score_and_scorecard_branches() -> None:
    """SCENARIO-PHASE4-056: action candidates and scorecard extraction cover SDK shapes."""

    assert adapter._available_action_ids(
        SimpleNamespace(available_actions=[SimpleNamespace(value=3), "ACTION4", 0, 3])
    ) == [3, 4]

    click_grid = np.zeros((5, 5), dtype=np.int16)
    click_grid[1, 2] = 4
    click_grid[4, 4] = 5
    click_candidates = adapter._action_candidates(
        SimpleNamespace(frame=click_grid, available_actions=[6])
    )
    assert [candidate.action_id for candidate in click_candidates] == [6, 6]
    assert click_candidates[0].data == {"x": 2, "y": 1}

    center_candidate = adapter._action_candidates(
        SimpleNamespace(frame=np.zeros((4, 6), dtype=np.int16), available_actions=[6])
    )[0]
    assert center_candidate.data == {"x": 3, "y": 2}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(adapter, "objects", lambda grid: [(1, 1), (1, 1)])
    try:
        assert len(adapter._action_candidates(SimpleNamespace(frame=click_grid, available_actions=[6]))) == 1
    finally:
        monkeypatch.undo()

    class AttrAction:
        ACTION7 = "action-seven"

    assert adapter._game_action(AttrAction, 7) == "action-seven"
    assert adapter._game_action(object(), 8) == 8
    assert adapter._normalise_score(MetricMapping.Score(1.0, 1, 2)).score == 1.0
    assert adapter._normalise_score({"score": 2.0, "levels_completed": 1, "actions": 3}).actions == 3

    run_score = SimpleNamespace(score=3.0, levels_completed=1, actions=4)
    scorecard = SimpleNamespace(find_environment=lambda game_id: SimpleNamespace(runs=[run_score]))
    assert adapter._extract_environment_score(scorecard, "lp85-305b61c3") is run_score

    calls: list[str] = []

    def find_by_prefix(game_id: str) -> object | None:
        calls.append(game_id)
        return SimpleNamespace(runs=[run_score]) if game_id == "lp85" else None

    assert adapter._extract_environment_score(
        SimpleNamespace(find_environment=find_by_prefix),
        "lp85-305b61c3",
    ) is run_score
    assert calls == ["lp85-305b61c3", "lp85"]

    env_score = SimpleNamespace(id="lp85-305b61c3", runs=[])
    scorecard_without_finder = SimpleNamespace(environments=[env_score])
    assert adapter._extract_environment_score(scorecard_without_finder, "lp85-305b61c3") is env_score
    with pytest.raises(ValueError, match="no scorecard"):
        adapter._extract_environment_score(None, "lp85-305b61c3")
    with pytest.raises(ValueError, match="did not include"):
        adapter._extract_environment_score(SimpleNamespace(environments=[]), "lp85-305b61c3")

    assert adapter._baseline_reference(EnvironmentSummary("x", "X", [], [9]), MetricMapping.Score(0, 0, 0)) == 9
    assert adapter._baseline_reference(EnvironmentSummary("x", "X", [], []), MetricMapping.Score(0, 0, 0)) == 0


def test_scenario_phase4_056_baseline_edge_paths() -> None:
    """SCENARIO-PHASE4-056: baseline runner handles no-actions, no-frame, and fallback scores."""

    class NoActionEnv:
        scorecard_id = "none"

        def reset(self) -> SimpleNamespace:
            return SimpleNamespace(frame=np.zeros((2, 2), dtype=np.int16), levels_completed=0, available_actions=[])

        def step(self, action: object, data: dict | None = None, reasoning: dict | None = None) -> object:
            raise AssertionError("step should not run")

    no_action = run_random_greedy_baseline(
        NoActionEnv(),
        EnvironmentSummary("no00", "NO00", [], [5]),
        action_budget=1,
        action_enum=FakeActionEnum,
    )
    assert no_action.trace == [{"action_index": 1, "event": "no_available_actions"}]
    assert no_action.score_source == "local_adapter_fallback"
    assert no_action.baseline_actions == 5

    class ResetNoneEnv:
        def reset(self) -> None:
            return None

    with pytest.raises(ValueError, match="reset returned no frame"):
        run_random_greedy_baseline(
            ResetNoneEnv(),
            EnvironmentSummary("bad00", "BAD00", [], [1]),
            action_enum=FakeActionEnum,
        )

    class StepNoneEnv:
        scorecard_id = "step-none"

        def reset(self) -> SimpleNamespace:
            return SimpleNamespace(frame=np.zeros((2, 2), dtype=np.int16), levels_completed=0, available_actions=[1])

        def step(self, action: object, data: dict | None = None, reasoning: dict | None = None) -> None:
            return None

    step_none = run_random_greedy_baseline(
        StepNoneEnv(),
        EnvironmentSummary("sn00", "SN00", [], [7]),
        action_budget=1,
        action_enum=FakeActionEnum,
    )
    assert step_none.trace[-1]["event"] == "step_returned_no_frame"

    class OneActionChangingEnv:
        scorecard_id = "one-action"

        def __init__(self) -> None:
            self.index = 0

        def reset(self) -> SimpleNamespace:
            self.index = 0
            return self._frame(0)

        def step(self, action: object, data: dict | None = None, reasoning: dict | None = None) -> SimpleNamespace:
            self.index += 1
            return self._frame(self.index)

        def _frame(self, value: int) -> SimpleNamespace:
            grid = np.zeros((2, 2), dtype=np.int16)
            grid[0, 0] = value
            return SimpleNamespace(frame=grid, levels_completed=0, available_actions=[1])

    changing = run_random_greedy_baseline(
        OneActionChangingEnv(),
        EnvironmentSummary("ch00", "CH00", [], [7]),
        action_budget=2,
        action_enum=FakeActionEnum,
    )
    assert len(changing.trace) == 2

    class OneActionSameEnv(OneActionChangingEnv):
        def _frame(self, value: int) -> SimpleNamespace:
            del value
            return SimpleNamespace(
                frame=np.zeros((2, 2), dtype=np.int16),
                levels_completed=0,
                available_actions=[1],
            )

    same = run_random_greedy_baseline(
        OneActionSameEnv(),
        EnvironmentSummary("sm00", "SM00", [], [7]),
        action_budget=2,
        action_enum=FakeActionEnum,
    )
    assert len(same.trace) == 2


def test_scenario_phase4_056_schema_rejects_fabricated_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-056: schema errors reject malformed live-result claims."""

    bad = {
        "honest_verdict": 4191,
        "live_env_reachable": "true",
        "real_metric_mapping": {},
        "random_greedy_baseline": [],
        "no_leaderboard_submission": False,
        "preconditions_checked": {"sdk_importable": 1, "network_reachable": 1},
        "leaderboard_submission_attempted": True,
        "field_principles": [],
        "requirements": [],
    }
    errors = artifact_schema_errors(bad)
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in errors)
    assert any("live_env_reachable must be a bare bool" in err for err in errors)
    assert any("no_leaderboard_submission must be true" in err for err in errors)
    assert any("leaderboard_submission_attempted must be false" in err for err in errors)
    assert any("real_metric_mapping must equal" in err for err in errors)
    assert any("preconditions_checked missing sdk_version" in err for err in errors)
    assert any("preconditions_checked.sdk_importable must be a bare bool" in err for err in errors)
    assert any("random_greedy_baseline must be a dict" in err for err in errors)
    assert any("requirements must include" in err for err in errors)
    assert any("field_principles must be a dict" in err for err in errors)
    assert any("honest_verdict must be terminal-prefixed" in err for err in artifact_schema_errors({**bad, "honest_verdict": "maybe"}))
    assert any("preconditions_checked must be a dict" in err for err in artifact_schema_errors({**bad, "preconditions_checked": []}))
    assert any(
        "complete reachable artifacts must set live_env_reachable true" in err
        for err in artifact_schema_errors({**bad, "honest_verdict": "complete: bad", "live_env_reachable": False})
    )

    live_missing = {
        **blocked_artifact(
            preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
            duration_s=0.0,
        ),
        "honest_verdict": "complete: fabricated",
        "live_env_reachable": True,
        "random_greedy_baseline": {"actions_taken": -1, "action_budget": 0},
        "offline_validation": {"passed": False},
        "field_principles": {"honest_verdict": "only one"},
    }
    live_errors = artifact_schema_errors(live_missing)
    assert any("random_greedy_baseline missing environment" in err for err in live_errors)
    assert any("actions_taken must be non-negative" in err for err in live_errors)
    assert any("action_budget must be positive" in err for err in live_errors)
    assert any("field_principles missing live_env_reachable" in err for err in live_errors)
    assert any("complete reachable artifacts require passed offline_validation" in err for err in live_errors)

    blocked_bad = {
        **blocked_artifact(
            preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
            duration_s=0.0,
        ),
        "live_env_reachable": True,
    }
    assert any("blocked artifacts must set live_env_reachable false" in err for err in artifact_schema_errors(blocked_bad))

    blocked_via_builder = build_artifact(
        outcome=LiveProbeOutcome(
            environment=EnvironmentSummary("lp85-305b61c3", "LP85", [], [17]),
            action_budget=1,
            actions_taken=0,
            baseline_actions=17,
            actions_vs_baseline_actions=0.0,
            score=MetricMapping.Score(0.0, 0, 0),
            trace=[],
            scorecard_id="open",
            score_source="fixture",
            anonymous_key_used=True,
            leaderboard_submission_attempted=False,
        ),
        preconditions=ArcLivePreconditions(False, "missing", True, "https://three.arcprize.org"),
        offline_validation={"passed": True},
        environment_count=0,
        duration_s=0.0,
    )
    assert blocked_via_builder["honest_verdict"] == "blocked_arc_live_unreachable"

    monkeypatch.setattr(adapter, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        blocked_artifact(
            preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
            duration_s=0.0,
        )
    with pytest.raises(ValueError, match="forced"):
        build_artifact(
            outcome=LiveProbeOutcome(
                environment=EnvironmentSummary("lp85-305b61c3", "LP85", [], [17]),
                action_budget=1,
                actions_taken=1,
                baseline_actions=17,
                actions_vs_baseline_actions=1 / 17,
                score=MetricMapping.Score(0.0, 0, 1),
                trace=[],
                scorecard_id="open",
                score_source="fixture",
                anonymous_key_used=True,
            ),
            preconditions=ArcLivePreconditions(True, "0.9.8", True, "https://three.arcprize.org"),
            offline_validation={"passed": True},
            environment_count=1,
            duration_s=0.0,
        )


def test_scenario_phase4_056_exp_run_paths_and_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-056: experiment run writes blocked, success, and live-error artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    exp._write_artifact({"honest_verdict": "complete: fixture"})
    assert (tmp_path / "results" / exp.RESULT_NAME).exists()

    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=adapter.BASE_URL: ArcLivePreconditions(False, "missing", True, base_url),
    )
    blocked = exp.run(write=True)
    assert blocked["honest_verdict"] == "blocked_arc_live_unreachable"

    selected = EnvironmentSummary("lp85-305b61c3", "LP85", ["click"], [17])
    outcome = LiveProbeOutcome(
        environment=selected,
        action_budget=1,
        actions_taken=1,
        baseline_actions=17,
        actions_vs_baseline_actions=1 / 17,
        score=MetricMapping.Score(0.0, 0, 1, [1], [17]),
        trace=[],
        scorecard_id="open",
        score_source="fixture",
        anonymous_key_used=True,
        leaderboard_submission_attempted=False,
    )
    monkeypatch.setattr(
        exp,
        "check_live_preconditions",
        lambda base_url=adapter.BASE_URL: ArcLivePreconditions(True, "0.9.8", True, base_url),
    )
    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": True})
    monkeypatch.setattr(exp, "open_online_arcade", lambda base_url=adapter.BASE_URL: object())
    monkeypatch.setattr(exp, "run_live_reachability_probe", lambda arcade, action_budget, random_seed: (25, outcome))
    success = exp.run(write=True, action_budget=1)
    assert success["live_env_reachable"] is True
    assert success["environment_count"] == 25

    monkeypatch.setattr(
        exp,
        "run_live_reachability_probe",
        lambda arcade, action_budget, random_seed: (_ for _ in ()).throw(RuntimeError("live down")),
    )
    failed_live = exp.run(write=False, action_budget=1)
    assert failed_live["honest_verdict"] == "blocked_arc_live_unreachable"
    assert "live_probe_error=RuntimeError" in failed_live["preconditions_checked"]["error"]

    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": False})
    validation_failed = exp.run(write=False, action_budget=1)
    assert validation_failed["honest_verdict"] == "blocked_arc_live_unreachable"

    monkeypatch.setattr(exp, "validate_recorded_fixture", lambda: {"passed": True})
    monkeypatch.setattr(
        exp,
        "run_live_reachability_probe",
        lambda arcade, action_budget, random_seed: (_ for _ in ()).throw(RuntimeError("schema path")),
    )
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        exp.run(write=False, action_budget=1)
