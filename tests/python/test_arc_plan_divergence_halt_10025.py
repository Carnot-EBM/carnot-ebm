"""CPU regressions for REQ-ARC-WMTE-10025 live plan verification."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_llm_reinduction import LlmReinductionResult
from carnot.experiment_10025_induced_plan_divergence import _plan_acceptance_path, _promotion


class Frame:
    """Minimal real-frame fixture for SCENARIO-ARC-WMTE-10025-DIVERGENCE."""

    def __init__(self, value: int, level: int = 0) -> None:
        self.frame = [(np.arange(16).reshape(4, 4) + value).tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _policy(monkeypatch, *, halt: bool) -> agent.E3AgentPolicy:
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    if halt:
        monkeypatch.setenv("CARNOT_ARC_PLAN_DIVERGENCE_HALT", "1")
    else:
        monkeypatch.delenv("CARNOT_ARC_PLAN_DIVERGENCE_HALT", raising=False)
    policy = agent.E3AgentPolicy("xx11", proposer=None, explore_budget=100)
    policy.phase = "execute"
    policy.plan = [{"action": action, "data": None} for action in (1, 2, 3)]
    policy.pi = 0
    policy._plan_divergence_engine = lambda grid, action, data: grid + 1
    policy.explorer = SimpleNamespace(
        next_move=lambda frames, latest: (6, None),
        explored_out=False,
        pending=[],
        set_goal_bias=lambda *args, **kwargs: None,
        root=None,
    )
    monkeypatch.setattr(policy, "_maybe_route_from_frame", lambda latest: None)
    monkeypatch.setattr(policy, "_maybe_route_from_transitions", lambda: None)
    monkeypatch.setattr(policy, "_maybe_supervise_trajectory", lambda latest: None)
    monkeypatch.setattr(policy, "_observe_level_boundary", lambda *args, **kwargs: [])
    monkeypatch.setattr(policy, "_current_goal_reached", lambda: False)
    monkeypatch.setattr(policy, "_should_enter_induction", lambda **kwargs: (False, ""))
    return policy


def test_halt_exactly_at_second_mismatching_step(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-DIVERGENCE abandons only after observed step two."""

    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    policy = _policy(monkeypatch, halt=True)
    assert policy.next_move([], Frame(0)) == (1, None)
    assert policy.next_move([Frame(0)], Frame(1)) == (2, None)
    assert policy.pi == 2
    assert policy.phase == "execute"
    assert policy.next_move([Frame(0), Frame(1)], Frame(99)) == (6, None)
    assert policy.phase == "explore"
    assert policy.plan == []
    assert policy.verified_divergence_events[-1]["step_index"] == 2
    assert policy.plans_abandoned_by_verified_divergence == 1
    summary = policy.action_provenance().summary()
    assert summary["plans_abandoned"] == 1
    assert summary["plans_abandoned_by_verified_divergence"] == 1
    assert summary["plans_abandoned_other_reason"] == 0
    assert policy.action_provenance().rows[-1]["verified_divergence_step_index"] == 2


def test_matching_plan_runs_fully(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-MATCH consumes all three matching steps."""

    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    policy = _policy(monkeypatch, halt=True)
    moves = [
        policy.next_move([], Frame(0)),
        policy.next_move([Frame(0)], Frame(1)),
        policy.next_move([Frame(0), Frame(1)], Frame(2)),
    ]
    assert moves == [(1, None), (2, None), (3, None)]
    assert policy.pi == len(policy.plan) == 3
    assert policy.verified_divergence_events == []
    assert policy.plans_abandoned_by_verified_divergence == 0
    assert policy.next_move([], Frame(3)) == (6, None)
    summary = policy.action_provenance().summary()
    assert summary["plans_consumed_fully"] == 1
    assert summary["plans_abandoned"] == 0
    assert summary["plans_abandoned_by_verified_divergence"] == 0


def test_other_plan_abandonment_keeps_separate_count(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-DIVERGENCE separates external abandonment."""

    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    policy = _policy(monkeypatch, halt=True)
    assert policy.next_move([], Frame(0)) == (1, None)

    def external_stop(latest):
        policy.plan = []
        policy.pi = 0
        policy.phase = "explore"

    monkeypatch.setattr(policy, "_maybe_supervise_trajectory", external_stop)
    assert policy.next_move([Frame(0)], Frame(1)) == (6, None)
    summary = policy.action_provenance().summary()
    assert summary["plans_abandoned"] == 1
    assert summary["plans_abandoned_by_verified_divergence"] == 0
    assert summary["plans_abandoned_other_reason"] == 1
    assert policy.verified_divergence_events == []


def test_level_up_precedes_mismatch(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-DIVERGENCE does not halt a level-up frame."""

    policy = _policy(monkeypatch, halt=True)
    assert policy.next_move([], Frame(0)) == (1, None)
    assert policy.next_move([Frame(0)], Frame(99, level=1)) == (2, None)
    assert policy.verified_divergence_events == []
    assert policy.plans_abandoned_by_verified_divergence == 0


def test_final_step_mismatch_is_not_plan_abandonment(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-MATCH distinguishes exhaustion from halt."""

    policy = _policy(monkeypatch, halt=True)
    for index in range(3):
        assert policy.next_move([], Frame(index)) == (index + 1, None)
    assert policy.next_move([], Frame(99)) == (6, None)
    assert policy.verified_divergence_events[-1]["step_index"] == 3
    assert policy.verified_divergence_events[-1]["stopped_by_divergence_halt"] is False
    assert policy.plans_abandoned_by_verified_divergence == 0


def test_both_live_call_sites_use_threshold_override(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-THRESHOLDS covers both live gate arguments."""

    source = open(agent.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_execute_bounded_llm_reinduction_with_arm_fallback"
    ]
    assert len(calls) == 2
    for call in calls:
        gates = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
            if keyword.arg in {"min_heldout_accuracy", "min_goal_predicate_consistency"}
        }
        assert gates == {
            "min_heldout_accuracy": "_induction_accept_threshold()",
            "min_goal_predicate_consistency": "_induction_accept_threshold()",
        }
    monkeypatch.delenv("CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD", raising=False)
    assert agent._induction_accept_threshold() == 1.0
    for value in (1.0, 0.9, 0.75):
        monkeypatch.setenv("CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD", str(value))
        assert agent._induction_accept_threshold() == value


@pytest.mark.parametrize("reason", ["stall", "level_up_reinduction"])
@pytest.mark.parametrize("threshold", [None, 1.0, 0.9, 0.75])
def test_live_induction_arguments_at_both_sites(monkeypatch, reason, threshold):
    """SCENARIO-ARC-WMTE-10025-THRESHOLDS observes the actual live call arguments."""

    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
    if threshold is None:
        monkeypatch.delenv("CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD", str(threshold))
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return LlmReinductionResult(
            planned=False,
            model_specs="CPU fixture",
            heldout_accuracy=0.0,
            skipped="proposer_failed",
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", capture)
    policy = agent.E3AgentPolicy(
        "xx11",
        proposer=SimpleNamespace(
            model_specs="CPU fixture", induce=lambda *args, **kwargs: (False, "fixture_declines")
        ),
        value_head=lambda frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = np.array([[1]], dtype=np.int16)
    policy._pending_induction_reason = reason
    policy._induce_and_plan()
    assert len(calls) == 1
    assert calls[0]["min_heldout_accuracy"] == (1.0 if threshold is None else threshold)
    assert calls[0]["min_goal_predicate_consistency"] == (1.0 if threshold is None else threshold)


def _main_routed_function():
    source = subprocess.check_output(
        ["git", "show", "main:python/carnot/agentic/arc_competition_agent.py"], text=True
    )
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "E3AgentPolicy"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_next_move_routed"
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)], type_ignores=[]))
    namespace = vars(agent).copy()
    exec(compile(isolated, "main:arc_competition_agent.py", "exec"), namespace)
    return namespace["_next_move_routed"]


def test_flags_off_identical_to_main_on_100_fixtures(monkeypatch):
    """SCENARIO-ARC-WMTE-10025-IDENTITY compares 100 deterministic execute fixtures."""

    monkeypatch.delenv("CARNOT_ARC_PLAN_DIVERGENCE_HALT", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD", raising=False)
    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    baseline = _main_routed_function()
    current = _policy(monkeypatch, halt=False)
    old = _policy(monkeypatch, halt=False)
    old._next_move_routed = MethodType(baseline, old)
    actual_trace = []
    expected_trace = []
    for fixture in range(100):
        for policy in (current, old):
            policy.plan = [{"action": 1 + fixture % 5, "data": None}]
            policy.pi = 0
            policy.phase = "execute"
        frame = Frame(fixture % 7)
        actual = current.next_move([], frame)
        expected = old.next_move([], frame)
        assert (actual, current.phase, current.pi, current.plan) == (
            expected,
            old.phase,
            old.pi,
            old.plan,
        )
        assert current.action_provenance().rows[-1] == old.action_provenance().rows[-1]
        for policy, action, trace in (
            (current, actual, actual_trace),
            (old, expected, expected_trace),
        ):
            trace.append(
                {
                    "action": action,
                    "phase": policy.phase,
                    "pi": policy.pi,
                    "plan": policy.plan,
                    "provenance": policy.action_provenance().rows[-1],
                }
            )
    assert (
        json.dumps(actual_trace, sort_keys=True).encode()
        == json.dumps(expected_trace, sort_keys=True).encode()
    )


def test_real_score_promotion_requires_no_seed_drop():
    """REQ-ARC-WMTE-10025 applies both scorecard mean and paired-seed guards."""

    rows = []
    for seed, baseline, treatment in ((7491001, 1.0, 0.5), (7491002, 1.0, 3.0)):
        for arm, score in (("1.0_halt", baseline), ("0.9_halt", treatment)):
            rows.append(
                {
                    "game": "su15",
                    "seed": seed,
                    "arm": arm,
                    "status": "complete",
                    "real_scorecard_score": score,
                }
            )
    verdict = _promotion(rows, ("su15",))["0.9_halt"]
    assert verdict["arm_mean"] > verdict["control_mean"]
    assert verdict["score_drops"] == ["su15:7491001"]
    assert verdict["promotion_candidate"] is False
    rows[1]["real_scorecard_score"] = 1.5
    assert _promotion(rows, ("su15",))["0.9_halt"]["promotion_candidate"] is True


def test_bounded_rejection_with_later_plan_is_not_bounded_acceptance():
    """REQ-ARC-WMTE-10025 distinguishes the live post-rejection planning path."""

    assert _plan_acceptance_path([{"planned": True, "skipped": ""}]) == "bounded_verified_plan"
    assert (
        _plan_acceptance_path(
            [{"planned": True, "skipped": "heldout_transition_verification_failed"}]
        )
        == "post_rejection_plan"
    )
    assert _plan_acceptance_path([{"planned": False, "skipped": "proposer_failed"}]) == "no_plan"
