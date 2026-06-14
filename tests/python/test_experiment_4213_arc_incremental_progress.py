"""Tests for Exp 4213 ARC-AGI-3 hardened fallback incremental progress.

Spec refs: REQ-PHASE4-059, SCENARIO-PHASE4-059.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import carnot.experiment_4213_arc_incremental_progress as exp
from carnot.experiment_4213_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_LEVELS_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIREMENTS,
    SC25_GAME_ID,
    FrontierOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    build_sc25_pattern_click_plan,
    execute_plan_until_level,
    gap4_hardening_ready,
    plan_sc25_suffix_bounded,
    select_deeper_level_target,
    target_pattern_cells,
    validate_hardened_gap4_heldout_replay,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class FakeAction:
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION6 = 6


@dataclass
class FakeSprite:
    x: int
    y: int
    scale: int = 1


class FakeSc25Game:
    def __init__(self, level: int = 1) -> None:
        self.zzpoabuniyn = {
            "l1": [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ],
            "tevyeq": [
                [True, True, False],
                [False, True, False],
                [False, False, False],
            ],
        }
        self.set_level(level)

    def set_level(self, level: int) -> None:
        self._current_level_index = int(level)
        self.jlpticwjyvy = ["l1" if level == 0 else "tevyeq"]
        self.ijhfdcamokt = self.jlpticwjyvy[0]
        self.xhhaqjfncnp = [[False for _ in range(3)] for _ in range(3)]
        self.rrinmfkkstu = 0
        self.txyqmqkitgl = None
        self.pattern_ready = False
        self.moves_after_pattern = 0
        self.eycwbtepcvs = False
        self.xelyxfeemol = 0
        self.ufpevlpokkj = 0
        self.wihhwrkolym = 0
        self.plnqvukupu = FakeSprite(10 + level, 10 + level, 1)


class FakeSc25Env:
    def __init__(self) -> None:
        self._game = FakeSc25Game(level=0)

    def reset(self) -> SimpleNamespace:
        self._game.set_level(0)
        return SimpleNamespace(levels_completed=0)

    def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
        game = self._game
        action_id = exp._action_id(action)
        if action_id == 6 and data:
            coord_to_cell = {
                (25, 50): (0, 0),
                (30, 50): (0, 1),
                (35, 50): (0, 2),
                (25, 55): (1, 0),
                (30, 55): (1, 1),
                (35, 55): (1, 2),
                (25, 60): (2, 0),
                (30, 60): (2, 1),
                (35, 60): (2, 2),
            }
            cell = coord_to_cell[(int(data["x"]), int(data["y"]))]
            row, col = cell
            game.xhhaqjfncnp[row][col] = not game.xhhaqjfncnp[row][col]
            game.rrinmfkkstu += 1
            if game.xhhaqjfncnp == game.zzpoabuniyn[game.jlpticwjyvy[0]]:
                game.pattern_ready = True
                game.xhhaqjfncnp = [[False for _ in range(3)] for _ in range(3)]
        elif action_id == 1 and game.pattern_ready:
            game.moves_after_pattern += 1
            game.rrinmfkkstu += 1
            needed = 1 if game._current_level_index == 0 else 2
            if game.moves_after_pattern >= needed:
                game.set_level(game._current_level_index + 1)
                return SimpleNamespace(levels_completed=game._current_level_index)
        else:
            game.rrinmfkkstu += 1
        return SimpleNamespace(levels_completed=game._current_level_index)


class DemoClickSc25Env(FakeSc25Env):
    def __init__(self) -> None:
        super().__init__()
        self.demo_click_pending = True

    def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
        if self.demo_click_pending and exp._action_id(action) == 6:
            self.demo_click_pending = False
            return SimpleNamespace(levels_completed=self._game._current_level_index)
        return super().step(action, data)


class PatternOnlySc25Env(FakeSc25Env):
    def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
        frame = super().step(action, data)
        if exp._action_id(action) == 6 and getattr(self._game, "pattern_ready", False):
            self._game.set_level(self._game._current_level_index + 1)
            return SimpleNamespace(levels_completed=self._game._current_level_index)
        return frame


class FakeArcade:
    def make(self, game_id: str) -> FakeSc25Env:
        assert game_id == SC25_GAME_ID
        return FakeSc25Env()


def _prior_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4201_arc_incremental_progress",
        "honest_verdict": "complete: incremental_progress_no_solve_lp85-305b61c3_L4_no_observed_level_up_candidate",
        "target_game": "lp85-305b61c3",
        "target_level": 4,
        "total_levels_solved": 15,
        "levels_completed": 3,
        "real_env_confirmed": False,
        "new_levels_solved_this_task": 0,
    }


def _hardening_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4187_gap4_graded_execution_gate_hardening",
        "vote_aware_guard_blocked_mispromotion": True,
        "gross_recovery_ledger": {"recovered": 4, "lost": 0},
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=2,
        prior_level=1,
        baseline_actions=6,
        selection_mode="fallback_deeper_level_after_lp85_L4_structural_block",
        selection_reason=(
            "selected sc25 L2 after Exp 4201 structurally blocked on lp85 L4; "
            "sc25 has a local L2 baseline and already-solved L1 prefix"
        ),
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    validation = validate_hardened_gap4_heldout_replay(
        start_level=1,
        final_level=2 if advanced else 1,
        heldout_transition_count=3,
        predicted_level=2,
        gap4_artifact=_hardening_artifact(),
    )
    return FrontierOutcome(
        target_game=SC25_GAME_ID,
        target_level=2,
        prior_level=1,
        final_level_completed=2 if advanced else 1,
        replay_actions_used=20,
        executed_real_env_actions=5 if advanced else 0,
        exploration_actions_used=8,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[validation],
        action_plan=[
            {"action": 6, "x": 25, "y": 50, "kind": "pattern_click"},
            {"action": 1, "kind": "move"},
        ]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 1},
            {"phase": "explore", "observed_transition_count": 8},
            {"phase": "induce", "mechanic": "sc25 pattern toggles plus exit touch"},
            validation,
            {"phase": "act", "levels_completed": 2 if advanced else 1},
        ],
        induced_mechanic="sc25 3x3 pattern-toggle unlock followed by exit-touch movement",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_059_spec_declares_exp4213_contract() -> None:
    """REQ-PHASE4-059: OpenSpec declares the Exp 4213 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-059" in spec
    assert "SCENARIO-PHASE4-059" in spec
    assert "experiment_4213_arc_incremental_progress.json" in spec
    assert SC25_GAME_ID in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_059_selects_sc25_l2_after_lp85_l4_block() -> None:
    """REQ-PHASE4-059: target selection switches to a different solved-game level."""

    survey = {"ranked_targets": [{"game": "lp85"}, {"game": "sc25"}]}
    baselines = {"sc25": (SC25_GAME_ID, [36, 6])}

    assert select_deeper_level_target(survey, baselines, _prior_artifact(), _hardening_artifact()) == _target()

    bad_prior = {**_prior_artifact(), "real_env_confirmed": True}
    with pytest.raises(ValueError, match="Exp 4201 lp85 L4 structural-block evidence unavailable"):
        select_deeper_level_target(survey, baselines, bad_prior, _hardening_artifact())
    with pytest.raises(ValueError, match="hardened GAP-4 verifier evidence unavailable"):
        select_deeper_level_target(survey, baselines, _prior_artifact(), {"gross_recovery_ledger": {"lost": 1}})
    with pytest.raises(ValueError, match="sc25 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {}, _prior_artifact(), _hardening_artifact())
    with pytest.raises(ValueError, match="sc25 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {"sc25": ("sc25-wrong", [36])}, _prior_artifact(), _hardening_artifact())
    with pytest.raises(ValueError, match="sc25 survey evidence unavailable"):
        select_deeper_level_target({"ranked_targets": [{"game": "lp85"}]}, baselines, _prior_artifact(), _hardening_artifact())


def test_scenario_phase4_059_artifacts_validate_success_no_solve_and_blocked() -> None:
    """SCENARIO-PHASE4-059: only hardened-verified real-env evidence increments levels."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4213, duration_s=0.25)

    assert success["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L2_total16"
    assert success["total_games_solved"] == 13
    assert success["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED + 1
    assert success["levels_completed"] == 2
    assert success["new_levels_solved_this_task"] == 1
    assert success["real_env_confirmed"] is True
    assert success["verifier_validated"] is True
    assert success["requirements"] == REQUIREMENTS
    assert success["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(success) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in success

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4213, duration_s=0.25)
    assert no_solve["honest_verdict"] == (
        "complete: incremental_progress_no_solve_sc25-635fd71a_L2_no_verifier_validated_level_up_candidate"
    )
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert no_solve["acceptance_gate_passed"] is True
    assert artifact_schema_errors(no_solve) == []

    blocked = blocked_artifact(target_game=SC25_GAME_ID, target_level=2, random_seed=4213, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert blocked["real_env_confirmed"] is False
    assert artifact_schema_errors(blocked) == []


def test_req_phase4_059_loads_local_baselines_and_fixture_guards(tmp_path: Path) -> None:
    """REQ-PHASE4-059: malformed local metadata cannot fabricate fixture readiness."""

    valid = tmp_path / "sc25" / "635fd71a"
    valid.mkdir(parents=True)
    valid.joinpath("metadata.json").write_text(
        json.dumps({"game_id": SC25_GAME_ID, "baseline_actions": [36, 6]}),
        encoding="utf-8",
    )
    malformed = tmp_path / "bad" / "json"
    malformed.mkdir(parents=True)
    malformed.joinpath("metadata.json").write_text("{not json", encoding="utf-8")
    no_dash = tmp_path / "bad" / "id"
    no_dash.mkdir(parents=True)
    no_dash.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "sc25", "baseline_actions": [1]}),
        encoding="utf-8",
    )

    assert exp.load_environment_baselines(tmp_path) == {"sc25": (SC25_GAME_ID, [36, 6])}
    assert exp._fixture_available("malformed") is False


def test_scenario_phase4_059_hardened_gap4_and_sc25_planning() -> None:
    """SCENARIO-PHASE4-059: held-out validation precedes SC25 L2 execution."""

    assert gap4_hardening_ready(_hardening_artifact()) is True
    assert gap4_hardening_ready({"gross_recovery_ledger": {"lost": 1}}) is False

    retained = validate_hardened_gap4_heldout_replay(1, 2, 3, 2, gap4_artifact=_hardening_artifact())
    rejected = validate_hardened_gap4_heldout_replay(1, 1, 3, 2, gap4_artifact=_hardening_artifact())
    unready = validate_hardened_gap4_heldout_replay(1, 2, 3, 2, gap4_artifact={})
    assert retained["retained"] is True
    assert retained["energy"] == 0.0
    assert retained["verifier"] == exp.HARDENED_VERIFIER
    assert rejected["retained"] is False
    assert unready["retained"] is False
    assert unready["hardened_gap4_ready"] is False

    game = FakeSc25Game(level=1)
    game.xhhaqjfncnp[0][0] = True
    assert target_pattern_cells(game) == [(0, 0), (0, 1), (1, 1)]
    assert build_sc25_pattern_click_plan(game) == [
        {"action": 6, "kind": "pattern_click", "row": 0, "col": 1, "x": 30, "y": 50},
        {"action": 6, "kind": "pattern_click", "row": 1, "col": 1, "x": 30, "y": 55},
    ]

    env = FakeSc25Env()
    env._game.set_level(1)
    plan, planner_trace = plan_sc25_suffix_bounded(env, FakeAction, target_level=2, max_depth=8)
    assert [step["kind"] for step in plan] == ["pattern_click", "pattern_click", "pattern_click", "move", "move"]
    assert planner_trace["found"] is True
    assert planner_trace["predicted_level"] == 2
    assert planner_trace["observed_transition_count"] > 0
    assert env._game._current_level_index == 1

    final_level, executed, trace = execute_plan_until_level(
        FakeSc25Env(),
        FakeAction,
        [{"action": 1, "kind": "move"}, {"action": 1, "kind": "move"}],
        prior_level=1,
        target_level=2,
    )
    assert (final_level, executed) == (1, 2)
    assert [row["phase"] for row in trace] == ["act", "act"]


def test_scenario_phase4_059_planner_retries_demo_click_and_stops_bounded() -> None:
    """SCENARIO-PHASE4-059: copied exploration handles demo clicks and bounded exhaustion."""

    class ValueAction:
        value = 7

    class StringValueAction:
        value = "ACTION4"

    class IntLikeAction:
        def __int__(self) -> int:
            return 2

    assert exp._action_id(ValueAction()) == 7
    assert exp._action_id(StringValueAction()) == 4
    assert exp._action_id(IntLikeAction()) == 2

    demo_env = DemoClickSc25Env()
    demo_plan, demo_trace = plan_sc25_suffix_bounded(demo_env, FakeAction, target_level=1, max_depth=4)
    assert demo_trace["found"] is True
    assert demo_trace["pattern_click_count"] == 5
    assert [step["kind"] for step in demo_plan].count("pattern_click") == 5

    class RetryPatternOnlyEnv(DemoClickSc25Env):
        def step(self, action: object, data: dict[str, int] | None = None) -> SimpleNamespace:
            frame = super().step(action, data)
            if exp._action_id(action) == 6 and getattr(self._game, "pattern_ready", False):
                self._game.set_level(self._game._current_level_index + 1)
                return SimpleNamespace(levels_completed=self._game._current_level_index)
            return frame

    retry_pattern_only = RetryPatternOnlyEnv()
    retry_plan, retry_trace = plan_sc25_suffix_bounded(
        retry_pattern_only,
        FakeAction,
        target_level=1,
        max_depth=0,
    )
    assert retry_trace["stopped_reason"] == "pattern_click_level_increment_found"
    assert [step["kind"] for step in retry_plan].count("pattern_click") == 5

    pattern_only = PatternOnlySc25Env()
    pattern_only._game.set_level(1)
    pattern_plan, pattern_trace = plan_sc25_suffix_bounded(pattern_only, FakeAction, target_level=2, max_depth=0)
    assert pattern_trace["stopped_reason"] == "pattern_click_level_increment_found"
    assert [step["kind"] for step in pattern_plan] == ["pattern_click", "pattern_click", "pattern_click"]

    prefilled_env = FakeSc25Env()
    prefilled_env._game.set_level(1)
    prefilled_env._game.xhhaqjfncnp = deepcopy(prefilled_env._game.zzpoabuniyn["tevyeq"])
    prefilled_plan, prefilled_trace = plan_sc25_suffix_bounded(
        prefilled_env,
        FakeAction,
        target_level=99,
        max_depth=0,
    )
    assert prefilled_plan == []
    assert prefilled_trace["stopped_reason"] == "frontier_exhausted"

    stuck_env = FakeSc25Env()
    stuck_env._game.set_level(1)
    stuck_plan, stuck_trace = plan_sc25_suffix_bounded(stuck_env, FakeAction, target_level=99, max_depth=0)
    assert stuck_plan == []
    assert stuck_trace["stopped_reason"] == "frontier_exhausted"

    broken_game = SimpleNamespace(jlpticwjyvy=["missing"], zzpoabuniyn={})
    with pytest.raises(ValueError, match="sc25 target pattern unavailable"):
        target_pattern_cells(broken_game)


def test_scenario_phase4_059_fake_sc25_frontier_run_advances() -> None:
    """SCENARIO-PHASE4-059: fake offline SC25 run follows replay-explore-verify-act."""

    outcome = exp._run_sc25_l2_frontier(FakeArcade(), _target(), _hardening_artifact())

    assert outcome.advanced is True
    assert outcome.final_level_completed == 2
    assert outcome.replay_actions_used > 0
    assert outcome.executed_real_env_actions == 5
    assert outcome.exploration_actions_used > 0
    assert [row["phase"] for row in outcome.phase_trace if row["phase"] in {"observe", "explore", "induce"}] == [
        "observe",
        "explore",
        "induce",
    ]


def test_scenario_phase4_059_frontier_run_reports_honest_no_solve_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-059: failed frontiers stop before unverified acting."""

    with monkeypatch.context() as patch:
        patch.setattr(exp, "plan_sc25_suffix_bounded", lambda *args, **kwargs: ([], {"observed_transition_count": 3}))
        no_l1 = exp._run_sc25_l2_frontier(FakeArcade(), _target(), _hardening_artifact())
    assert no_l1.failure_reason == "could_not_reestablish_prior_frontier"
    assert no_l1.action_plan == []

    with monkeypatch.context() as patch:
        patch.setattr(
            exp,
            "plan_sc25_suffix_bounded",
            lambda *args, **kwargs: ([{"action": 1, "kind": "move"}], {"observed_transition_count": 4}),
        )
        patch.setattr(
            exp,
            "execute_plan_until_level",
            lambda *args, **kwargs: (0, 1, [{"phase": "replay", "levels_completed": 0}]),
        )
        weak_replay = exp._run_sc25_l2_frontier(FakeArcade(), _target(), _hardening_artifact())
    assert weak_replay.failure_reason == "could_not_reestablish_prior_frontier"
    assert weak_replay.final_level_completed == 0

    with monkeypatch.context() as patch:
        calls: list[int] = []

        def planner(*args: object, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, int]]:
            calls.append(1)
            if len(calls) == 1:
                return [{"action": 1, "kind": "move"}], {"observed_transition_count": 5}
            return [], {"observed_transition_count": 6}

        patch.setattr(exp, "plan_sc25_suffix_bounded", planner)
        patch.setattr(
            exp,
            "execute_plan_until_level",
            lambda *args, **kwargs: (1, 1, [{"phase": "replay", "levels_completed": 1}]),
        )
        no_candidate = exp._run_sc25_l2_frontier(FakeArcade(), _target(), _hardening_artifact())
    assert no_candidate.failure_reason == "no_observed_level_up_candidate"
    assert no_candidate.action_plan == []

    with monkeypatch.context() as patch:
        calls = []

        def validated_planner(*args: object, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, int]]:
            calls.append(1)
            return [{"action": 1, "kind": "move"}], {"observed_transition_count": len(calls)}

        patch.setattr(exp, "plan_sc25_suffix_bounded", validated_planner)
        patch.setattr(
            exp,
            "execute_plan_until_level",
            lambda *args, **kwargs: (1, 1, [{"phase": "replay", "levels_completed": 1}]),
        )
        patch.setattr(
            exp,
            "_validate_suffix_on_copy",
            lambda *args, **kwargs: {"retained": False, "verifier": exp.HARDENED_VERIFIER},
        )
        rejected = exp._run_sc25_l2_frontier(FakeArcade(), _target(), _hardening_artifact())
    assert rejected.failure_reason == "no_verifier_validated_level_up_candidate"
    assert rejected.verification_decisions == [{"retained": False, "verifier": exp.HARDENED_VERIFIER}]


def test_scenario_phase4_059_schema_rejects_fabricated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-059: schema guards reject malformed or fabricated increments."""

    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors({}))
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4213}))
    assert any("honest_verdict must be terminal-prefixed" in err for err in artifact_schema_errors({"honest_verdict": "maybe"}))
    assert any("total_levels_solved must be a bare int" in err for err in artifact_schema_errors({"total_levels_solved": True}))
    assert any("real_env_confirmed must be a bare bool" in err for err in artifact_schema_errors({"real_env_confirmed": 1}))
    assert any("target_game must be a string" in err for err in artifact_schema_errors({"target_game": 4213}))
    assert any("solve_trace must be a dict" in err for err in artifact_schema_errors({"solve_trace": []}))
    assert any("inference_substrate must equal" in err for err in artifact_schema_errors({"inference_substrate": "wrong"}))
    assert any("requirements must include" in err for err in artifact_schema_errors({"requirements": []}))
    assert any("field_principles must be a dict" in err for err in artifact_schema_errors({"field_principles": []}))
    assert any(
        "field_principles missing honest_verdict" in err
        for err in artifact_schema_errors({"field_principles": {"total_levels_solved": "x"}})
    )
    assert any(
        "total_levels_solved must be monotonic" in err
        for err in artifact_schema_errors({"honest_verdict": "blocked_x", "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED - 1})
    )

    bad_success = {
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L2_total16",
        "target_game": SC25_GAME_ID,
        "target_level": 2,
        "prior_total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "total_levels_solved": PRIOR_TOTAL_LEVELS_SOLVED,
        "new_levels_solved_this_task": 0,
        "levels_completed": 1,
        "real_env_confirmed": False,
        "verifier_validated": False,
        "verification_decisions": [{"retained": False}],
        "action_plan": [],
        "solve_trace": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "requirements": REQUIREMENTS,
        "field_principles": dict(exp.REQUIRED_FIELD_PRINCIPLES),
    }
    errors = artifact_schema_errors(bad_success)
    assert any("real_env_confirmed must be true for success" in err for err in errors)
    assert any("verifier_validated must be true for success" in err for err in errors)
    assert any("total_levels_solved must increment" in err for err in errors)
    assert any("levels_completed must reach target_level" in err for err in errors)
    assert any("success requires a retained hardened GAP-4 verifier decision" in err for err in errors)
    assert any("success requires a validated action_plan" in err for err in errors)

    bad_complete = {**deepcopy(bad_success), "honest_verdict": "complete: incremental_progress_no_solve_sc25-635fd71a_L2_x"}
    bad_complete["real_env_confirmed"] = True
    bad_complete["total_levels_solved"] = PRIOR_TOTAL_LEVELS_SOLVED + 1
    bad_complete["new_levels_solved_this_task"] = 1
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_levels_solved must remain" in err for err in complete_errors)
    assert any("new_levels_solved_this_task must be zero" in err for err in complete_errors)
    assert any("real_env_confirmed must be false" in err for err in complete_errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_outcome(advanced=True), _target(), random_seed=4213, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(target_game=SC25_GAME_ID, target_level=2, random_seed=4213, duration_s=0.0)


def test_scenario_phase4_059_runner_writes_terminal_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-059: runner writes blocked and real-env-confirmed artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4213_arc_incremental_progress.json").exists()

    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"ranked_targets": [{"game": "lp85"}, {"game": "sc25"}]}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4201_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json").write_text(
        json.dumps(_hardening_artifact()),
        encoding="utf-8",
    )
    no_fixture = exp.run(write=True)
    assert no_fixture["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir = tmp_path / "environment_files" / "sc25" / "635fd71a"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": SC25_GAME_ID, "baseline_actions": [36, 6]}),
        encoding="utf-8",
    )
    missing_py = exp.run(write=True)
    assert missing_py["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    fixture_dir.joinpath("sc25.py").write_text("# marker\n", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_sc25_l2_frontier", lambda arcade, target, gap4: _outcome(advanced=True))
    success = exp.run(write=True)
    assert success["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L2_total16"
    written = json.loads((tmp_path / "results" / "experiment_4213_arc_incremental_progress.json").read_text())
    assert written == success

    monkeypatch.setattr(
        exp,
        "_run_sc25_l2_frontier",
        lambda arcade, target, gap4: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    errored = exp.run(write=False)
    assert errored["honest_verdict"].startswith("complete: incremental_progress_no_solve_sc25-635fd71a_L2")
    assert "offline_run_failed_runtimeerror" in errored["honest_verdict"]


def test_results_entrypoint_exists() -> None:
    """REQ-PHASE4-059: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4213_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4213_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
