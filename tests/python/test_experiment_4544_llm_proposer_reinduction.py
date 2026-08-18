"""Tests for Exp 4544 live LLM proposer re-induction.

Spec refs: REQ-ARC-WMTE-4544, SCENARIO-ARC-WMTE-4544.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
pytestmark = pytest.mark.memory_watchdog_skip


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _measurement(
    label: str,
    *,
    levels: dict[str, int] | None = None,
    efficiency: float = 2.0074,
    planned: bool = False,
) -> dict[str, object]:
    levels = dict(levels or _levels())
    return {
        "measurement": label,
        "core_efficiency": efficiency,
        "deepest_level_by_game": levels,
        "per_game": [
            {
                "game": game,
                "best_level": level,
                "efficiency": efficiency / 4.0,
                "diagnostics": {
                    "induction_attempts": [
                        {
                            "reason": "level_up_reinduction",
                            "planned": planned,
                            "skipped": "" if planned else "proposer_failed_or_missing_root",
                        }
                    ]
                },
            }
            for game, level in levels.items()
        ],
    }


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "qwen3_5_9b_mtp_gguf_cached": True,
        "qwen3_5_9b_mtp_gguf_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
        "llama_cpp_import": True,
        "llama_cpp_version": "0.3.29",
        "spec_has_req_4544": True,
        "ok": True,
    }


def test_req_arc_wmte_4544_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4544: OpenSpec anchors the live-LLM artifact contract."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4544" in spec
    assert "SCENARIO-ARC-WMTE-4544" in spec
    assert exp4544.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4544.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4544_bounded_refinement_stops_on_reachable_plan(monkeypatch) -> None:
    """REQ-ARC-WMTE-4544: refinement retries are bounded and stop after a reachable plan.

    EARLY-STOP needs at least two rounds to be observable: round 1 returns a stuck engine, round 2
    returns a progressing one, and the property is that the loop halts there instead of spending
    its remaining budget. The shipped cap became 1 on 2026-08-17 (rounds past the first measured
    pooled-negative on held-out), which makes early-stop untestable at the default.

    So this raises the cap through CARNOT_ARC_MAX_REFINEMENT_ROUNDS, which exists for exactly this
    -- testing a multi-round mechanism without editing source. The default stays 1 and is pinned
    separately by tests/python/test_arc_refinement_rounds_cap.py.
    """

    import importlib

    monkeypatch.setenv("CARNOT_ARC_MAX_REFINEMENT_ROUNDS", "3")
    import carnot.agentic.arc_llm_reinduction as _reinduction

    importlib.reload(_reinduction)

    from carnot.agentic.arc_executable_world_model import Transition

    execute_bounded_llm_reinduction = _reinduction.execute_bounded_llm_reinduction

    class FakeProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def __init__(self) -> None:
            self.induce_calls = 0
            self.refactor_calls = 0

        def induce(self, _game, _transitions, _cell):
            self.induce_calls += 1
            return True, "first candidate"

        def refactor(self, _game, _counterexample):
            self.refactor_calls += 1
            return True, "refined candidate"

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

    def stuck_engine(grid, _action, _data):
        return np.asarray(grid)

    def progress_engine(grid, _action, _data):
        return np.asarray(grid) + 1

    engines = iter(
        [
            (stuck_engine, lambda grid: bool(np.asarray(grid)[0, 0] >= 2)),
            (progress_engine, lambda grid: bool(np.asarray(grid)[0, 0] >= 2)),
        ]
    )

    def load_engine(_game):
        return next(engines)

    def plan_in_model(engine, is_done, start_grid):
        grid = np.asarray(engine(np.asarray(start_grid), 1, None))
        return [{"action": 1, "data": None}] if bool(is_done(grid)) else None

    proposer = FakeProposer()
    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[1]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=load_engine,
        plan_in_model=plan_in_model,
        max_rounds=3,
    )

    assert result.planned is True
    assert result.plan == [{"action": 1, "data": None}]
    assert result.refinement_rounds_used == 2
    assert proposer.induce_calls == 1
    assert proposer.refactor_calls == 1
    assert result.verifier_is_oracle is False
    assert result.goal_candidate_names == ["loaded"]
    assert result.dynamics_candidate_names == ["loaded"]
    assert result.counterexamples[0]["kind"] in {
        "degenerate_goal_predicate",
        "no_reachable_plan",
        "plan_execution",
    }


def test_req_arc_wmte_4544_bounded_refinement_caps_at_the_round_bound() -> None:
    """SCENARIO-ARC-WMTE-4544: an unreachable plan cannot exceed the loop bound.

    The property under test is that refinement is BOUNDED, not that the bound is any particular
    number. It was written when the bound was 3 and asserted 3 literally; the bound became 1 on
    2026-08-17 (operator-approved) because rounds past the first measured pooled-negative on
    held-out. Derived from MAX_REFINEMENT_ROUNDS so it tests the property and moves with the
    constant, instead of going red every time the operator retunes the cap.

    Note `execute_bounded_llm_reinduction` clamps with min(max_rounds, MAX_REFINEMENT_ROUNDS), so
    the module constant is a hard ceiling even against an explicit caller argument. That is
    deliberate: it guarantees the cap holds no matter who calls.
    """

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import (
        MAX_REFINEMENT_ROUNDS,
        execute_bounded_llm_reinduction,
    )

    class NeverProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def __init__(self) -> None:
            self.calls = []

        def induce(self, _game, _transitions, _cell):
            self.calls.append("induce")
            return True, "candidate"

        def refactor(self, _game, _counterexample):
            self.calls.append("refactor")
            return True, "refined"

    transition = Transition(
        grid=np.array([[0]], dtype=np.int16),
        action=1,
        data=None,
        next_grid=np.array([[0]], dtype=np.int16),
        level_before=1,
        level_after=1,
    )

    def noop(grid, _action, _data):
        return np.asarray(grid)

    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=[transition],
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=NeverProposer(),
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: (noop, lambda grid: bool(np.asarray(grid)[0, 0] >= 9)),
        plan_in_model=lambda _engine, _goal, _start: None,
        max_rounds=3,
    )

    assert result.planned is False
    # Asked for 3; the ceiling decides. Both assertions derive from the constant so the bound is
    # what is tested, not the number that happened to be shipped when this was written.
    assert result.refinement_rounds_used == MAX_REFINEMENT_ROUNDS
    assert [row["round"] for row in result.rounds] == list(range(1, MAX_REFINEMENT_ROUNDS + 1))
    assert result.skipped == "degenerate_goal_predicate"


def test_req_arc_wmte_4544_honest_null_records_proposer_value(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4544: no L2 still reports measured proposer value and null delta."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    artifact = exp4544.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline"),
        llm_proposer=_measurement("llm_proposer", planned=True),
        llm_proposer_value={"count": 1, "opportunities": 2, "rate": 0.5, "events": ["lp85:L2"]},
        positive_control={"passed": True, "reachable_plan": True, "dsl_reachable_plan": False},
        offline_reproduction={},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [2], "m0r0": [3]},
        barrier_refinement="llm_plan_reaches_model_goal_but_real_execution_still_diverges_at_click",
        random_seed=4544,
        duration_s=61.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: llm_proposer_no_deeper_level_proposer_value_characterized_honest_null"
    )
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["efficiency_delta"] == 0.0
    assert "null_delta_methodology_note" in artifact
    assert artifact["llm_proposer_value"]["count"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert exp4544.artifact_schema_errors(artifact) == []

    out = exp4544.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4544_success_requires_l2_efficiency_preservation_and_replay() -> None:
    """REQ-ARC-WMTE-4544: submitted config changes only on strict CORE improvement."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    artifact = exp4544.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline", levels=_levels(lp85=1)),
        llm_proposer=_measurement(
            "llm_proposer", levels=_levels(lp85=2), efficiency=3.125, planned=True
        ),
        llm_proposer_value={"count": 1, "opportunities": 1, "rate": 1.0, "events": ["lp85:L2"]},
        positive_control={"passed": True, "reachable_plan": True, "dsl_reachable_plan": False},
        offline_reproduction={"reproduced": True, "game": "lp85", "reached_level": 2},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [2]},
        barrier_refinement="resolved: llm proposer reached L2",
        random_seed=4544,
        duration_s=61.0,
    )

    assert artifact["honest_verdict"] == (
        "success: llm_proposer_lp85_reached_L2_core_efficiency_3.1250_above_2.0074"
    )
    assert artifact["core_efficiency_best"] == 3.125
    assert artifact["efficiency_delta"] == 1.1176
    assert artifact["core_solves_preserved"] is True
    assert artifact["chosen_submitted_config"]["llm_proposer_reinduction"] is True
    assert exp4544.artifact_schema_errors(artifact) == []

    dropped = exp4544.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline", levels=_levels(lp85=1, m0r0=1)),
        llm_proposer=_measurement(
            "llm_proposer",
            levels=_levels(lp85=2, m0r0=0),
            efficiency=3.125,
            planned=True,
        ),
        llm_proposer_value={"count": 1, "opportunities": 1, "rate": 1.0, "events": ["lp85:L2"]},
        positive_control={"passed": True, "reachable_plan": True, "dsl_reachable_plan": False},
        offline_reproduction={"reproduced": True, "game": "lp85", "reached_level": 2},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [2]},
        barrier_refinement="lp85_l2_reached_but_m0r0_core_solve_dropped",
        random_seed=4544,
        duration_s=61.0,
    )
    assert dropped["honest_verdict"] == (
        "complete: llm_proposer_no_deeper_level_proposer_value_characterized_honest_null"
    )
    assert dropped["chosen_submitted_config"] == "unchanged"


def test_scenario_arc_wmte_4544_run_writes_injected_measurements(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4544: injected matched measurements write stable JSON."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    artifact = exp4544.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        measurement_runner=lambda: (
            _measurement("offline_dsl_baseline"),
            _measurement("llm_proposer", planned=True),
        ),
        positive_control_runner=lambda: {
            "passed": True,
            "reachable_plan": True,
            "dsl_reachable_plan": False,
        },
        offline_reproduction_runner=lambda _best: {},
        live_invocation_runner=lambda _proposer, _model_path: {"invoked": True, "duration_s": 60.0},
        now=lambda: 1.0,
    )

    assert artifact["result_path"] == exp4544.RESULT_RELATIVE_PATH
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert (
        json.loads((tmp_path / exp4544.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        == artifact
    )


def test_scenario_arc_wmte_4544_e3_policy_records_llm_refinement(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4544: level-up induction records GOAL/DYNAMICS and bounded rounds."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    result = LlmReinductionResult(
        planned=True,
        plan=[{"action": 1, "data": None}],
        goal_predicate=lambda grid: bool(np.asarray(grid).sum()),
        engine=lambda grid, _action, _data: np.asarray(grid),
        selected_candidate_name="candidate-a",
        goal_candidate_names=["goal-a"],
        dynamics_candidate_names=["dynamics-a"],
        refinement_rounds_used=2,
        verifier_is_oracle=False,
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        rounds=[{"round": 1}, {"round": 2}],
        counterexamples=[{"kind": "no_reachable_plan"}],
        skipped="",
    )

    monkeypatch.setattr(
        agent,
        "execute_bounded_llm_reinduction",
        lambda **_kwargs: result,
        raising=False,
    )

    policy = agent.E3AgentPolicy(
        "lp85",
        proposer=SimpleNamespace(model_specs=result.model_specs),
        target_levels=2,
        value_head=None,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "level_up_reinduction"
    policy._current_goal_level = 2

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert policy.plan == [{"action": 1, "data": None}]
    assert attempt["planned"] is True
    assert attempt["refinement_rounds_used"] == 2
    assert attempt["goal_candidate_names"] == ["goal-a"]
    assert attempt["dynamics_candidate_names"] == ["dynamics-a"]
    assert attempt["verifier_is_oracle"] is False


def test_req_arc_fcp_5699_35_stall_refactor_loop_enabled_by_default(monkeypatch) -> None:
    """REQ-ARC-FCP-5699-35 graduated this to default-on: CARNOT_ARC_STALL_REFACTOR_LOOP unset
    (production default) now DOES route a stall-triggered (first-contact) induction attempt
    through execute_bounded_llm_reinduction, superseding the REQ-ARC-FCP-5699-24 dev-only
    "unset == disabled" contract."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    monkeypatch.delenv("CARNOT_ARC_STALL_REFACTOR_LOOP", raising=False)

    called = {"n": 0}

    def _capture(**_kwargs):
        called["n"] += 1
        return LlmReinductionResult(
            planned=False,
            plan=[],
            goal_predicate=None,
            engine=None,
            skipped="proposer_failed",
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _capture, raising=False)

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    policy._induce_and_plan()

    assert called["n"] == 1
    attempt = policy.induction_attempts[-1]
    assert attempt["reason"] == "stall"
    assert attempt["stall_refactor_loop_used"] is True


def test_req_arc_fcp_5699_35_stall_refactor_loop_explicit_opt_out(monkeypatch) -> None:
    """The explicit opt-out (CARNOT_ARC_STALL_REFACTOR_LOOP=0) still reproduces the exact
    pre-REQ-ARC-FCP-5699-24 behavior: execute_bounded_llm_reinduction is never called on a
    stall, and the attempt falls through to the pre-existing plain single-shot path -- the
    escape hatch this graduation preserves."""

    from carnot.agentic import arc_competition_agent as agent

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")

    def _fail_if_called(**_kwargs):
        raise AssertionError(
            "execute_bounded_llm_reinduction must not fire when explicitly disabled"
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fail_if_called, raising=False)

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    policy._induce_and_plan()  # must not raise the AssertionError above

    attempt = policy.induction_attempts[-1]
    assert attempt["reason"] == "stall"
    assert "stall_refactor_loop_used" not in attempt
    assert attempt["skipped"] == "proposer_failed_or_missing_root"  # the plain path's own outcome


def test_req_arc_fcp_5699_24_stall_refactor_loop_records_outcome_when_enabled(monkeypatch) -> None:
    """REQ-ARC-FCP-5699-24: CARNOT_ARC_STALL_REFACTOR_LOOP=1 routes a stall-triggered
    (first-contact) induction attempt through execute_bounded_llm_reinduction, with
    previous_level_complete_grid=None (no exemplar exists yet) and structural_goal_provider=None,
    and records the outcome onto the attempt exactly as the level_up_reinduction branch does.
    Since the mocked outcome is planned=False, this also exercises REQ-ARC-FCP-5699-35's
    fallthrough to the plain single-shot path (the proposer stub below needs a real .induce()
    so that fallthrough completes instead of raising AttributeError on the minimal stub)."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1")

    result = LlmReinductionResult(
        planned=False,
        plan=[],
        goal_predicate=None,
        engine=None,
        selected_candidate_name="candidate-b",
        goal_candidate_names=["goal-b"],
        dynamics_candidate_names=["dynamics-b"],
        refinement_rounds_used=3,
        verifier_is_oracle=False,
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        rounds=[{"round": 1}, {"round": 2}, {"round": 3}],
        counterexamples=[{"kind": "heldout_transition_verification_failed"}],
        skipped="no_reachable_plan_after_refinement",
    )

    captured_kwargs: dict = {}

    def _capture(**kwargs):
        captured_kwargs.update(kwargs)
        return result

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _capture, raising=False)

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs=result.model_specs,
            induce=lambda *_a, **_k: (False, "test_stub_declines"),
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    policy._induce_and_plan()

    assert captured_kwargs["previous_level_complete_grid"] is None  # no exemplar exists yet
    assert captured_kwargs["structural_goal_provider"] is None

    attempt = policy.induction_attempts[-1]
    # the stall loop's own fields survive the fallthrough (they're not touched again below)
    assert attempt["stall_refactor_loop_used"] is True
    assert attempt["refinement_rounds_used"] == 3
    # REQ-ARC-FCP-5699-35's fallthrough then tries the plain single-shot path (since
    # planned=False), whose own outcome legitimately supersedes "skipped"/"planned" as the
    # attempt's FINAL overall result -- both sub-outcomes are honestly represented: the stall
    # loop genuinely ran 3 rounds (recorded above), AND neither mechanism produced a plan.
    assert attempt["skipped"] == "proposer_failed_or_missing_root"
    assert attempt["planned"] is False


def test_req_arc_fcp_5699_35_stall_refactor_loop_planned_true_does_not_fall_through(
    monkeypatch,
) -> None:
    """When execute_bounded_llm_reinduction DOES reach a plan (planned=True), the stall path
    returns immediately with that plan installed -- it must NOT fall through to the plain
    single-shot path and overwrite the already-successful outcome."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1")

    plan = [{"action": 1}]
    result = LlmReinductionResult(
        planned=True,
        plan=plan,
        goal_predicate=None,
        engine=None,
        refinement_rounds_used=1,
        model_specs="stub",
        skipped="",
    )

    monkeypatch.setattr(
        agent, "execute_bounded_llm_reinduction", lambda **_k: result, raising=False
    )

    def _fail_if_called(*_a, **_k):
        raise AssertionError("plain single-shot path must not fire when already planned=True")

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(model_specs="stub", induce=_fail_if_called),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    policy._induce_and_plan()  # must not raise the AssertionError above

    attempt = policy.induction_attempts[-1]
    assert attempt["planned"] is True
    assert policy.plan == plan


def test_req_arc_fcp_5699_38_unsatisfiable_goal_predicate_not_installed(monkeypatch) -> None:
    """REQ-ARC-FCP-5699-38: a real post-submission regression investigation found the stall
    path installing a goal bias from an induced-but-UNSATISFIABLE goal_predicate
    (goal_predicate_satisfiable=False) even though the refinement never reached a plan
    (planned=False) -- persisting into the rest of the episode's exploration and biasing it
    toward a goal the agent's own diagnostics say is unachievable. A goal_predicate that is
    not satisfiable must never be installed, regardless of whether the loop reached a plan."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1")

    def _unsatisfiable_predicate(_grid):
        return False

    result = LlmReinductionResult(
        planned=False,
        plan=[],
        goal_predicate=_unsatisfiable_predicate,
        goal_predicate_satisfiable=False,
        engine=None,
        refinement_rounds_used=3,
        model_specs="stub",
        skipped="hidden_state_trust_below_threshold",
    )

    monkeypatch.setattr(
        agent, "execute_bounded_llm_reinduction", lambda **_k: result, raising=False
    )

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    # E3AgentPolicy always installs a DEFAULT goal_bias at construction time
    # (SUBMITTED_AGENT_CONFIG's goal_energy_enabled=True) -- goal_bias is never None, so the
    # correct check is object identity (did _install_goal_bias REPLACE it?), not a None-check.
    goal_bias_before = policy.explorer.goal_bias
    assert goal_bias_before is not None  # sanity: confirms the default-bias premise holds

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["goal_predicate_satisfiable"] is False
    assert attempt["planned"] is False
    assert policy.explorer.goal_bias is goal_bias_before  # unchanged: NOT replaced/installed


def test_req_arc_fcp_5699_39_tier1_degenerate_goal_energy_not_installed(monkeypatch) -> None:
    """REQ-ARC-FCP-5699-39: a real sc25 repro found the tier-1 warm-started-engine call site
    (gated_engine_from_transitions) installing a goal bias whose energy NEVER improved across a
    genuine 20009-node search (initial_goal_energy == min_goal_energy_observed == 1.0). The fix
    moves the install AFTER plan_in_model and gates it on that search's own diagnostics -- a flat
    (non-improving) energy must not install a bias."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_live_ttt

    def _degenerate_predicate(_grid):
        return False

    def _fake_gated_engine(_game, _transitions):
        return (lambda grid, action, data: grid, _degenerate_predicate, {"gate": "PASS"})

    monkeypatch.setattr(
        arc_live_ttt, "gated_engine_from_transitions", _fake_gated_engine, raising=False
    )

    def _fake_plan_in_model(engine, is_done, start_grid, *, diagnostics=None, **_k):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "used_goal_energy_search": True,
                    "initial_goal_energy": 1.0,
                    "min_goal_energy_observed": 1.0,
                    "nodes_expanded": 20009,
                    "termination_reason": "max_nodes_reached",
                }
            )
        return []  # no plan found

    import carnot.agentic.arc_executable_world_model as e3

    monkeypatch.setattr(e3, "plan_in_model", _fake_plan_in_model, raising=False)

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    goal_bias_before = policy.explorer.goal_bias
    assert goal_bias_before is not None  # the default bias, per E3AgentPolicy construction

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt.get("ttt_prior_goal_bias_installed") is False
    assert policy.explorer.goal_bias is goal_bias_before  # unchanged: NOT replaced/installed


def test_req_arc_fcp_5699_39_tier1_improving_goal_energy_still_installed(monkeypatch) -> None:
    """The fix is a gate, not a blanket removal: when plan_in_model's own search shows the goal
    energy genuinely improved (even without finding a full plan), the bias is still installed."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_live_ttt

    def _promising_predicate(_grid):
        return False

    def _fake_gated_engine(_game, _transitions):
        return (lambda grid, action, data: grid, _promising_predicate, {"gate": "PASS"})

    monkeypatch.setattr(
        arc_live_ttt, "gated_engine_from_transitions", _fake_gated_engine, raising=False
    )

    def _fake_plan_in_model(engine, is_done, start_grid, *, diagnostics=None, **_k):
        if diagnostics is not None:
            diagnostics.update(
                {
                    "used_goal_energy_search": True,
                    "initial_goal_energy": 1.0,
                    "min_goal_energy_observed": 0.2,  # real progress toward the goal
                    "nodes_expanded": 500,
                    "termination_reason": "max_nodes_reached",
                }
            )
        return []

    import carnot.agentic.arc_executable_world_model as e3

    monkeypatch.setattr(e3, "plan_in_model", _fake_plan_in_model, raising=False)

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    goal_bias_before = policy.explorer.goal_bias
    assert goal_bias_before is not None

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt.get("ttt_prior_goal_bias_installed") is True
    assert policy.explorer.goal_bias is not goal_bias_before  # replaced by the induced predicate


def test_req_arc_fcp_5699_38_satisfiable_goal_predicate_still_installed(monkeypatch) -> None:
    """The fix is a gate, not a blanket removal: a genuinely satisfiable induced goal_predicate
    (goal_predicate_satisfiable=True) still installs a goal bias exactly as before."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1")

    def _satisfiable_predicate(_grid):
        return False

    result = LlmReinductionResult(
        planned=False,
        plan=[],
        goal_predicate=_satisfiable_predicate,
        goal_predicate_satisfiable=True,
        engine=None,
        refinement_rounds_used=3,
        model_specs="stub",
        skipped="no_reachable_plan_after_refinement",
    )

    monkeypatch.setattr(
        agent, "execute_bounded_llm_reinduction", lambda **_k: result, raising=False
    )

    policy = agent.E3AgentPolicy(
        "paritytest",
        proposer=SimpleNamespace(
            model_specs="stub", induce=lambda *_a, **_k: (False, "test_stub_declines")
        ),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = "stall"

    goal_bias_before = policy.explorer.goal_bias
    assert goal_bias_before is not None  # the default bias, per E3AgentPolicy construction

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["goal_predicate_satisfiable"] is True
    assert policy.explorer.goal_bias is not goal_bias_before  # replaced by the induced predicate


def test_req_arc_wmte_4544_helper_defensive_branches() -> None:
    """REQ-ARC-WMTE-4544: helper branches emit compact counterexamples."""

    from carnot.agentic.arc_llm_reinduction import (
        WorldModelCandidate,
        _model_specs,
        _normalise_candidates,
        _plan_reaches_goal,
        execute_bounded_llm_reinduction,
    )

    def noop(grid, _action, _data):
        return np.asarray(grid)

    assert _model_specs(SimpleNamespace(repo_substr="Qwen3.5-9B-MTP", model_path="/m.gguf")) == (
        "Qwen3.5-9B-MTP GGUF (/m.gguf)"
    )
    assert _model_specs(SimpleNamespace()).endswith("SimpleNamespace")

    direct = WorldModelCandidate("direct", noop, lambda _grid: False)
    rows = _normalise_candidates(
        [
            direct,
            {"name": "mapping", "engine": noop, "goal_predicate": lambda _grid: False},
            ("tuple", noop),
        ],
        noop,
        lambda _grid: False,
    )
    assert [row.name for row in rows] == ["direct", "mapping", "tuple"]

    grid = np.array([[0]], dtype=np.int16)
    assert (
        _plan_reaches_goal(engine=noop, goal=None, start_grid=grid, plan=[])["counterexample"][
            "kind"
        ]
        == "missing_goal_predicate"
    )
    assert (
        _plan_reaches_goal(
            engine=noop,
            goal=lambda current: bool(np.asarray(current)[0, 0] == 0),
            start_grid=grid,
            plan=[],
        )["reaches_goal"]
        is True
    )

    def bad_goal(_grid):
        raise RuntimeError("goal boom")

    assert (
        _plan_reaches_goal(engine=noop, goal=bad_goal, start_grid=grid, plan=[])["counterexample"][
            "kind"
        ]
        == "goal_predicate_error"
    )

    def bad_engine(_grid, _action, _data):
        raise RuntimeError("engine boom")

    assert (
        _plan_reaches_goal(
            engine=bad_engine,
            goal=lambda _grid: False,
            start_grid=grid,
            plan=[{"action": 1, "data": None}],
        )["counterexample"]["kind"]
        == "plan_execution"
    )
    assert (
        _plan_reaches_goal(
            engine=noop,
            goal=bad_goal,
            start_grid=grid,
            plan=[{"action": 1, "data": None}],
        )["counterexample"]["kind"]
        == "goal_predicate_error"
    )
    assert (
        _plan_reaches_goal(
            engine=noop,
            goal=lambda _grid: False,
            start_grid=grid,
            plan=[{"action": 1, "data": None}],
        )["counterexample"]["reason"]
        == "plan_finished_before_goal"
    )

    proposer = SimpleNamespace(model_specs="Qwen3.5-9B-MTP GGUF (/m.gguf)")
    assert (
        execute_bounded_llm_reinduction(
            game="fixture",
            transitions=[SimpleNamespace()],
            cell=1,
            root_grid=None,
            proposer=proposer,
            candidate_provider=lambda _engine, _goal: [],
            load_engine=lambda _game: (noop, lambda _grid: False),
            plan_in_model=lambda _engine, _goal, _grid: None,
        ).skipped
        == "missing_root_grid"
    )
    assert (
        execute_bounded_llm_reinduction(
            game="fixture",
            transitions=[],
            cell=1,
            root_grid=grid,
            proposer=proposer,
            candidate_provider=lambda _engine, _goal: [],
            load_engine=lambda _game: (noop, lambda _grid: False),
            plan_in_model=lambda _engine, _goal, _grid: None,
        ).skipped
        == "no_active_transitions"
    )


def test_req_arc_wmte_4544_helper_failure_paths() -> None:
    """REQ-ARC-WMTE-4544: proposer and selection failures stay bounded."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    transition = Transition(
        grid=np.array([[0]], dtype=np.int16),
        action=1,
        data=None,
        next_grid=np.array([[0]], dtype=np.int16),
        level_before=0,
        level_after=0,
    )

    class Fails:
        model_specs = "Qwen3.5-9B-MTP GGUF (/m.gguf)"

        def induce(self, _game, _transitions, _cell):
            return False, "no code"

    failed = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=[transition],
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=Fails(),
        candidate_provider=lambda _engine, _goal: [],
        load_engine=lambda _game: (
            lambda grid, _action, _data: np.asarray(grid),
            lambda _grid: False,
        ),
        plan_in_model=lambda _engine, _goal, _grid: None,
    )
    assert failed.skipped == "proposer_failed"
    assert failed.rounds[0]["skipped"] == "proposer_failed"

    class Raises:
        model_specs = "Qwen3.5-9B-MTP GGUF (/m.gguf)"

        def induce(self, _game, _transitions, _cell):
            return True, "candidate"

        def refactor(self, _game, _counterexample):
            return True, "refined"

    errored = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=[transition],
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=Raises(),
        candidate_provider=lambda _engine, _goal: [],
        load_engine=lambda _game: (_ for _ in ()).throw(RuntimeError("load boom")),
        plan_in_model=lambda _engine, _goal, _grid: None,
        max_rounds=1,
    )
    assert errored.skipped == "no_reachable_plan_after_refinement"
    assert errored.counterexamples[0]["kind"] == "selection_or_planning_exception"


def test_req_arc_wmte_4544_experiment_helper_branches() -> None:
    """REQ-ARC-WMTE-4544: measurement helpers handle sparse and failed inputs."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    sparse = {
        "measurement": "sparse",
        "best_level_by_game": {"lp85": 2},
        "efficiency_by_game": {"lp85": 1.0},
        "per_game": [
            "bad-row",
            {"game": "m0r0", "levels": 1, "per_level_efficiency": 0.25},
            {
                "game": "sp80",
                "best_level": 1,
                "diagnostics": {
                    "induction_attempts": [
                        {
                            "reason": "level_up_reinduction",
                            "planned": False,
                            "refinement_rounds_used": 3,
                            "counterexamples": [{"kind": "no_reachable_plan"}],
                        }
                    ]
                },
            },
        ],
    }
    normal = exp4544._normalise_measurement(sparse, label="fallback")
    assert normal["deepest_level_by_game"]["lp85"] == 2
    assert normal["deepest_level_by_game"]["m0r0"] == 1
    assert normal["core_efficiency"] == 1.25
    assert (
        exp4544.characterize_llm_proposer_value({}, {"per_game": ["bad", {"game": "lp85"}]})["rate"]
        == 0.0
    )
    assert exp4544.refinement_rounds_from_measurement(sparse)["sp80"] == [3]
    assert (
        exp4544.positive_control_from_live(
            {"invoked": False, "reachable_plan": False, "dsl_reachable_plan": False}
        )["passed"]
        is False
    )
    assert exp4544._default_barrier(
        llm_proposer=normal,
        positive_control_passed=True,
        success=True,
    ).startswith("resolved:")
    assert exp4544._default_barrier(
        llm_proposer=normal,
        positive_control_passed=False,
        success=False,
    ).startswith("positive_control_failed")
    assert exp4544._default_barrier(
        llm_proposer={"per_game": []},
        positive_control_passed=True,
        success=False,
    ).startswith("no_post_level")
    assert "counterexamples" in exp4544._default_barrier(
        llm_proposer=normal,
        positive_control_passed=True,
        success=False,
    )


def test_req_arc_wmte_4544_schema_error_branches_and_blocked_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4544: schema rejects malformed artifacts and blocked runs write."""

    from carnot import experiment_4544_llm_proposer_reinduction as exp4544

    artifact = exp4544.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline"),
        llm_proposer=_measurement("llm_proposer", planned=True),
        llm_proposer_value={"count": 0, "opportunities": 1, "rate": 0.0, "events": []},
        positive_control={"passed": True, "reachable_plan": True, "dsl_reachable_plan": False},
        offline_reproduction={},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": []},
        barrier_refinement=None,
        random_seed=4544,
        duration_s=61.0,
    )

    mutations = []
    missing = dict(artifact)
    missing.pop("model_specs")
    mutations.append((missing, "missing required field model_specs"))
    for key, value, expected in [
        ("honest_verdict", "oops", "honest_verdict must start"),
        ("inference_substrate", "cached", "inference_substrate"),
        ("model_specs", "Qwen missing path", "model_specs must name"),
        ("field_principles", {}, "field_principles"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("core_efficiency_baseline", 0.0, "core_efficiency_baseline"),
        ("efficiency_delta", 999.0, "efficiency_delta"),
        ("false_negative_risk_checked", False, "false_negative_risk_checked"),
        ("preconditions_checked", [], "preconditions_checked must be a mapping"),
        ("preconditions_checked", {}, "preconditions_checked must record"),
        ("llm_proposer_value", [], "llm_proposer_value must be a mapping"),
        ("deepest_level_reached_per_core_game", [], "deepest_level"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("chosen_submitted_config", {"bad": True}, "non-success"),
    ]:
        changed = dict(artifact)
        changed[key] = value
        mutations.append((changed, expected))
    no_note = dict(artifact)
    no_note.pop("null_delta_methodology_note")
    mutations.append((no_note, "null_delta_methodology_note"))
    bad_value = dict(artifact)
    bad_value["llm_proposer_value"] = {"count": 2, "opportunities": 1}
    mutations.append((bad_value, "count cannot exceed"))

    for changed, expected in mutations:
        assert any(expected in error for error in exp4544.artifact_schema_errors(changed))

    success = exp4544.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline", levels=_levels(lp85=1)),
        llm_proposer=_measurement(
            "llm_proposer", levels=_levels(lp85=2), efficiency=3.2, planned=True
        ),
        llm_proposer_value={"count": 1, "opportunities": 1, "rate": 1.0, "events": ["lp85:L2"]},
        positive_control={"passed": True, "reachable_plan": True, "dsl_reachable_plan": False},
        offline_reproduction={"reproduced": True, "reached_level": 2},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [1]},
        barrier_refinement=None,
        random_seed=4544,
        duration_s=61.0,
    )
    for key, value, expected in [
        ("core_solves_preserved", False, "core_solves_preserved"),
        ("positive_control_passed", False, "positive_control_passed"),
        ("offline_reproduced", False, "offline_reproduced"),
        ("core_efficiency_best", 2.0074, "core_efficiency_best"),
        ("chosen_submitted_config", "unchanged", "chosen submitted config"),
    ]:
        changed = dict(success)
        changed[key] = value
        assert any(expected in error for error in exp4544.artifact_schema_errors(changed))

    with pytest.raises(ValueError):
        exp4544.write_artifact({**artifact, "reproducibility_checksum": "bad"}, root=tmp_path)

    blocked = exp4544.run(
        root=tmp_path,
        preconditions_checked={
            "offline_arcade_import_smoke": False,
            "qwen3_5_9b_mtp_gguf_cached": False,
            "llama_cpp_import": False,
            "spec_has_req_4544": True,
            "qwen3_5_9b_mtp_gguf_path": None,
            "ok": False,
        },
        measurement_runner=lambda: (_measurement("offline"), _measurement("llm")),
        live_invocation_runner=lambda _proposer, _model_path: {"invoked": False},
        now=lambda: 1.0,
    )
    assert blocked["honest_verdict"] == "blocked_llm_proposer_reinduction_precondition"
    assert (
        json.loads((tmp_path / exp4544.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == blocked
    )

    monkeypatch.setattr(exp4544, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError):
        exp4544.run(
            root=tmp_path,
            preconditions_checked=_preconditions(),
            measurement_runner=lambda: (_measurement("offline"), _measurement("llm")),
            positive_control_runner=lambda: {
                "passed": True,
                "reachable_plan": True,
                "dsl_reachable_plan": False,
            },
            offline_reproduction_runner=lambda _best: {},
            live_invocation_runner=lambda _proposer, _model_path: {"duration_s": 60.0},
            now=lambda: 1.0,
        )
