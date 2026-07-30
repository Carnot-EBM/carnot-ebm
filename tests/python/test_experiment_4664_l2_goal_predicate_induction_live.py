"""Tests for Exp 4664 L2 goal-predicate induction.

Spec refs: REQ-ARC-WMTE-4664,
SCENARIO-ARC-WMTE-4664-WIN-STATE-EXEMPLAR,
SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY,
SCENARIO-ARC-WMTE-4664-METRIC-HARNESS.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
pytestmark = pytest.mark.memory_watchdog_skip


def test_req_arc_wmte_4664_spec_declares_l2_goal_artifact_contract() -> None:
    """REQ-ARC-WMTE-4664: OpenSpec anchors the L2 goal-induction artifact."""

    from carnot import experiment_4664_l2_goal_predicate_induction_live as exp4664

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4664" in spec
    assert "SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY" in spec
    assert exp4664.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4664.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4664_previous_level_win_grid_enters_prompt() -> None:
    """SCENARIO-ARC-WMTE-4664-WIN-STATE-EXEMPLAR: L1 win grid is shown without L2 positives."""

    from carnot.agentic import arc_executable_world_model as e3

    transition = e3.Transition(
        grid=np.array([[0, 0], [0, 1]], dtype=np.int16),
        action=1,
        data=None,
        next_grid=np.array([[0, 1], [0, 1]], dtype=np.int16),
        level_before=1,
        level_after=1,
    )
    previous_level_complete = np.array([[7, 7], [0, 7]], dtype=np.int16)

    block = e3._transitions_block(
        [transition],
        previous_level_complete_grid=previous_level_complete,
    )

    # CORRECTED 2026-07-29. This test used to assert the block said "WIN STATE ... COMPLETED the
    # previous level ... next level's completion likely looks structurally similar". Those claims
    # were measured FALSE and the test was locking the defect in: `previous_level_complete_grid` is
    # captured from the frame AFTER the level counter incremented, so it is the CURRENT level's
    # OPENING BOARD, not a state that completed anything. (Measured on ka59's canonical L1 solve: the
    # completing action rewrites 3527 of 4096 cells because it re-lays out the playfield, and a
    # concept-correct predicate is False on that frame.) The block now describes the grid truthfully
    # and states the predicate's polarity on it, which is the whole point of the fix.
    assert "WIN STATE" not in block, "the false win-state labelling must not come back"
    assert "COMPLETED the previous level" not in block
    assert "BOARD AT THE START OF THE CURRENT LEVEL" in block
    assert "is_level_complete must return False here" in block
    # REQ-ARC-WMTE-5593-2 (2026-07-14): full-grid renders now use _rle_grid's row-wise,
    # implicit-column run-length encoding instead of raw to_ascii -- "r0:7x2\nr1:0x1,7x1"
    # is the lossless encoding of [[7,7],[0,7]], replacing the old raw "77\n07" form.
    assert "r0:7x2\nr1:0x1,7x1" in block


def test_scenario_arc_wmte_4664_e3_policy_captures_level_complete_grid() -> None:
    """SCENARIO-ARC-WMTE-4664-WIN-STATE-EXEMPLAR: live policy stores the boundary grid."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    completed_grid = np.array([[3, 3], [0, 3]], dtype=np.int16)

    event = policy._begin_level_goal_episode(
        1,
        frames_seen=4,
        completed_grid=completed_grid,
    )

    assert event["win_state_exemplar_captured"] is True
    assert np.array_equal(policy._previous_level_complete_grid, completed_grid)


def test_scenario_arc_wmte_4664_post_boundary_stall_uses_l2_reinduction(
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4664-WIN-STATE-EXEMPLAR: stall timing cannot bypass L2 induction."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    captured: dict[str, Any] = {}

    def fake_reinduction(**kwargs: Any) -> LlmReinductionResult:
        captured.update(kwargs)
        return LlmReinductionResult(
            planned=False,
            model_specs="Qwen3.5-9B-MTP",
            goal_predicate_satisfiable=False,
            counterexamples=[{"kind": "degenerate_goal_predicate"}],
            skipped="degenerate_goal_predicate",
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", fake_reinduction)

    policy = agent.E3AgentPolicy(
        "lp85", proposer=SimpleNamespace(model_specs="Qwen"), target_levels=2
    )
    policy._pending_induction_reason = "stall"
    policy._start_level = 0
    policy._current_goal_level = 2
    policy._previous_level_complete_grid = np.array([[8]], dtype=np.int16)
    policy.root_grid = np.array([[0]], dtype=np.int16)
    policy.transitions = [
        SimpleNamespace(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["reason"] == "level_up_reinduction"
    assert attempt["win_state_exemplar_injected"] is True
    assert attempt["goal_predicate_satisfiable"] is False
    assert np.array_equal(captured["previous_level_complete_grid"], np.array([[8]], dtype=np.int16))


def test_scenario_arc_wmte_4664_degenerate_goal_rejected_before_planning() -> None:
    """SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY: constant-false goals refine before plan."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    class FakeProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def __init__(self) -> None:
            self.refactor_kinds: list[str] = []

        def induce(self, _game: str, _transitions: list[Any], _cell: int) -> tuple[bool, str]:
            return True, "constant-false goal"

        def refactor(self, _game: str, counterexample: Any) -> tuple[bool, str]:
            self.refactor_kinds.append(counterexample.mismatches[0]["kind"])
            return True, "satisfiable goal"

    transitions = [
        Transition(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
        Transition(
            grid=np.array([[1]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[2]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
    ]

    def increment(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        return np.asarray(grid) + 1

    goals = iter(
        [
            lambda _grid: False,
            lambda grid: bool(np.asarray(grid)[0, 0] >= 2),
        ]
    )
    plan_calls = 0

    def plan_in_model(
        engine: Any, goal: Any, start_grid: np.ndarray
    ) -> list[dict[str, Any]] | None:
        nonlocal plan_calls
        plan_calls += 1
        grid = np.asarray(start_grid)
        path: list[dict[str, Any]] = []
        for _ in range(3):
            if bool(goal(grid)):
                return path
            grid = np.asarray(engine(grid.copy(), 1, None))
            path.append({"action": 1, "data": None})
        return path if bool(goal(grid)) else None

    proposer = FakeProposer()
    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (increment, next(goals)),
        plan_in_model=plan_in_model,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )

    assert result.planned is True
    assert result.goal_predicate_satisfiable is True
    assert result.counterexamples[0]["kind"] == "degenerate_goal_predicate"
    assert result.rounds[0]["goal_predicate_satisfiable"] is False
    assert result.rounds[0]["skipped"] == "degenerate_goal_predicate"
    assert result.rounds[1]["goal_predicate_satisfiable"] is True
    assert proposer.refactor_kinds == ["degenerate_goal_predicate"]
    assert plan_calls == 1


def test_req_arc_wmte_4664_goal_satisfiability_helper_defensive_branches() -> None:
    """REQ-ARC-WMTE-4664: goal satisfiability handles bad goals and bad engines."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic import arc_llm_reinduction as reinduction

    class KwProposer:
        def __init__(self) -> None:
            self.previous = None

        def induce(
            self,
            _game: str,
            _transitions: list[Any],
            _cell: int,
            *,
            previous_level_complete_grid: np.ndarray | None = None,
        ) -> tuple[bool, str]:
            self.previous = previous_level_complete_grid
            return True, "ok"

    proposer = KwProposer()
    assert reinduction._call_induce(
        proposer,
        "fixture",
        [],
        1,
        np.array([[5]], dtype=np.int16),
    ) == (True, "ok")
    assert np.array_equal(proposer.previous, np.array([[5]], dtype=np.int16))

    def raises_signature(_fn: Any) -> None:
        raise ValueError("no signature")

    original_signature = reinduction.inspect.signature
    try:
        reinduction.inspect.signature = raises_signature
        assert reinduction._supports_kwarg(lambda: None, "anything") is False
    finally:
        reinduction.inspect.signature = original_signature

    grid = np.array([[0, 1]], dtype=np.int16)
    assert (
        reinduction._plan_reaches_goal(
            engine=lambda current, _action, _data: np.asarray(current),
            goal=lambda _grid: False,
            start_grid=grid,
            plan=None,
        )["counterexample"]["kind"]
        == "no_reachable_plan"
    )

    assert (
        reinduction._goal_satisfiability_check(
            engine=lambda current, _action, _data: np.asarray(current),
            goal=None,
            start_grid=grid,
        )["counterexample"]["kind"]
        == "missing_goal_predicate"
    )

    def bad_goal(_grid: np.ndarray) -> bool:
        raise RuntimeError("goal boom")

    assert (
        reinduction._goal_satisfiability_check(
            engine=lambda current, _action, _data: np.asarray(current),
            goal=bad_goal,
            start_grid=grid,
        )["counterexample"]["kind"]
        == "goal_predicate_error"
    )

    def bad_or_misshapen_engine(current: np.ndarray, action: int, _data: Any) -> np.ndarray:
        if action == 1:
            raise RuntimeError("engine boom")
        if action == 2:
            return np.zeros((2, 2), dtype=np.int16)
        return np.asarray(current)

    depth_limited = reinduction._goal_satisfiability_check(
        engine=bad_or_misshapen_engine,
        goal=lambda _grid: False,
        start_grid=grid,
        max_depth=0,
    )
    assert depth_limited["counterexample"]["kind"] == "degenerate_goal_predicate"

    degenerate = reinduction._goal_satisfiability_check(
        engine=bad_or_misshapen_engine,
        goal=lambda _grid: False,
        start_grid=grid,
        max_depth=1,
    )
    assert degenerate["counterexample"]["kind"] == "degenerate_goal_predicate"

    reached_limit = reinduction._goal_satisfiability_check(
        engine=lambda current, _action, _data: np.asarray(current) + 1,
        goal=lambda _grid: False,
        start_grid=np.array([[0]], dtype=np.int16),
        max_nodes=2,
    )
    assert reached_limit["counterexample"]["max_nodes"] == 2

    class OneRoundProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def induce(self, _game: str, _transitions: list[Any], _cell: int) -> tuple[bool, str]:
            return True, "candidate"

    transition = Transition(
        grid=np.array([[0]], dtype=np.int16),
        action=1,
        data=None,
        next_grid=np.array([[1]], dtype=np.int16),
        level_before=1,
        level_after=1,
    )

    def action_sensitive_engine(current: np.ndarray, action: int, _data: Any) -> np.ndarray:
        return np.asarray(current) + 1 if action == 1 else np.asarray(current)

    missed_plan = reinduction.execute_bounded_llm_reinduction(
        game="fixture",
        transitions=[transition],
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=OneRoundProposer(),
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (
            action_sensitive_engine,
            lambda current: bool(np.asarray(current)[0, 0] >= 1),
        ),
        plan_in_model=lambda _engine, _goal, _start: [{"action": 2, "data": None}],
        max_rounds=1,
    )
    assert missed_plan.counterexamples[0]["reason"] == "plan_finished_before_goal"


def test_scenario_arc_wmte_4664_metric_artifact_contract(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4664: artifact records fixed harness and honest residuals."""

    from carnot import experiment_4664_l2_goal_predicate_induction_live as exp4664

    artifact = exp4664.build_artifact(
        preconditions_checked={
            "qwen3_5_9b_mtp_gguf_cached": True,
            "offline_arcade": True,
            "live_modules_importable": True,
            "qwen_proposer_port_verified": True,
        },
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        per_game={
            "lp85": {
                "goal_predicate_satisfiable": True,
                "l2_plan_len": 2,
                "l2_plan_reaches_goal": True,
                "generic_agent_reached_level": 2,
                "offline_reproduced": True,
                "reproduced_levels": 2,
            },
            "sc25": {
                "goal_predicate_satisfiable": False,
                "l2_plan_len": 0,
                "l2_plan_reaches_goal": False,
                "generic_agent_reached_level": 1,
                "offline_reproduced": False,
                "reproduced_levels": 1,
            },
        },
        duration_s=60.0,
    )

    assert artifact["honest_verdict"] == "success: l2_goal_induction_generic_agent_reached_L2_lp85"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["win_state_exemplar_injected"] is True
    assert artifact["goal_predicate_satisfiable"]["lp85"] is True
    assert artifact["l2_plan_len"]["lp85"] == 2
    assert artifact["l2_plan_reaches_goal"]["lp85"] is True
    assert artifact["metric_harness_fixed"]["target_levels"] >= 2
    assert artifact["proposer_served_model"] == "Qwen3.5-9B-MTP"
    assert artifact["offline_reproduced"]["lp85"] is True
    assert artifact["residual_cause_hypothesis"] == "none"
    assert exp4664.artifact_schema_errors(artifact) == []

    out = exp4664.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4664_null_and_schema_branches_are_auditable() -> None:
    """REQ-ARC-WMTE-4664: null artifacts name residuals and schema defects."""

    from carnot import experiment_4664_l2_goal_predicate_induction_live as exp4664

    assert (
        exp4664._residual_cause(
            {"lp85": {"goal_predicate_satisfiable": True, "generic_agent_reached_level": 1}}
        )
        == "l2_dynamics_wrong"
    )

    artifact = exp4664.build_artifact(
        preconditions_checked={"qwen_proposer_port_verified": True, "qwen_proposer_port": 8920},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        parity_test_green=True,
        per_game={
            "lp85": {
                "goal_predicate_satisfiable": False,
                "l2_plan_len": 0,
                "l2_plan_reaches_goal": False,
                "generic_agent_reached_level": 1,
                "offline_reproduced": False,
                "reproduced_levels": 1,
                "registry_l2_reachable": True,
            }
        },
        duration_s=60.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient"
    )
    assert artifact["residual_cause_hypothesis"] == "single_exemplar_goal_insufficient"
    assert "null_methodology_note" in artifact
    assert exp4664.artifact_schema_errors(artifact) == []

    broken = dict(artifact)
    broken["honest_verdict"] = "bad"
    broken["verifier_is_oracle"] = True
    broken["solve_provenance"] = "development_proxy"
    broken["proposer_served_model"] = "gemma-4-12B-it"
    broken["metric_harness_fixed"] = {"target_levels": 1}
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_methodology_note")

    errors = exp4664.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "solve_provenance" in errors
    assert "proposer_served_model" in errors
    assert "metric_harness_fixed" in errors
    assert "reproducibility_checksum" in errors

    missing_note = dict(artifact)
    missing_note.pop("null_methodology_note")
    assert "null_methodology_note" in exp4664.artifact_schema_errors(missing_note)


def test_scenario_arc_wmte_4664_run_variant_attempt_continues_past_first_win(
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4664-METRIC-HARNESS: rollout can measure depth >=2."""

    from carnot import experiment_4628_dense_curiosity_progress_loop as exp4628
    from carnot.agentic import arc_competition_agent, arc_solver_kit, arc_variant_generator

    class FakeExplorer:
        graph = {"root": {}, "l1": {}, "l2": {}}

        def curiosity_diagnostics(self) -> dict[str, Any]:
            return {"enabled": False}

    class FakePolicy:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            self.target_levels = int(kwargs["target_levels"])
            self.explorer = FakeExplorer()
            self.calls = 0

        def is_done(self, _frames: list[Any], latest: Any) -> bool:
            return latest is not None and int(latest.levels_completed) >= self.target_levels

        def next_move(self, _frames: list[Any], _latest: Any) -> tuple[int | str, None]:
            self.calls += 1
            if self.calls == 1:
                return "RESET", None
            return 1, None

    class FakeEnv:
        def __init__(self) -> None:
            self.level = 0

        def reset(self) -> SimpleNamespace:
            self.level = 0
            return SimpleNamespace(levels_completed=0)

        def step(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            self.level += 1
            return SimpleNamespace(levels_completed=self.level)

    class FakeArc:
        def open_scorecard(self) -> str:
            return "scorecard"

        def make(self, _game: str, scorecard_id: str | None = None) -> FakeEnv:
            assert scorecard_id == "scorecard"
            return FakeEnv()

    monkeypatch.setattr(arc_competition_agent, "E3AgentPolicy", FakePolicy)
    monkeypatch.setattr(
        arc_competition_agent, "_level_of", lambda frame: int(frame.levels_completed)
    )
    monkeypatch.setattr(arc_solver_kit, "offline_arcade", lambda: FakeArc())
    monkeypatch.setattr(
        arc_solver_kit,
        "reproduce",
        lambda *_args, **kwargs: {
            "game": "aa00",
            "claimed_level": kwargs["claimed_level"],
            "reached_level": kwargs["claimed_level"],
            "reproduced": True,
        },
    )
    monkeypatch.setattr(arc_variant_generator, "VariantEnv", lambda env, *_args, **_kwargs: env)

    attempt = exp4628.run_variant_attempt(
        "loop",
        "aa00",
        {"variant": 1, "kind": "color", "variant_signature": "aa00~color01"},
        budget=5,
    )

    assert attempt["reached_level"] == 2
    assert attempt["solved"] is True
    assert attempt["actions_to_first_levelup"] == 1
    assert len(attempt["solution_labels"]) == 2


def test_scenario_arc_wmte_4664_policy_for_mode_targets_multiple_levels(
    monkeypatch: Any,
) -> None:
    """SCENARIO-ARC-WMTE-4664-METRIC-HARNESS: harness policies request depth >=2."""

    from carnot import experiment_4628_dense_curiosity_progress_loop as exp4628
    from carnot.agentic import arc_competition_agent

    captured: list[dict[str, Any]] = []

    class FakePolicy:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured.append(dict(kwargs))

    monkeypatch.setattr(arc_competition_agent, "E3AgentPolicy", FakePolicy)

    exp4628._policy_for_mode("loop", "aa00")
    exp4628._policy_for_mode("bare", "aa00")

    assert [row["target_levels"] for row in captured] == [2, 2]
