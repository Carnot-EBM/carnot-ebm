"""REQ-ARC-PROBE-7680 and REQ-REPORT-7680: scored probe protocol."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import carnot.agentic.arc_probe_protocol as probe_module
from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_probe_protocol import ArcProbeProtocol
from carnot.experiment_7680_v669_arc_probe_protocol import cold_reduce_rows, fixture_frame


def frame(seed: int, *, level: int = 0, state: str = "NOT_FINISHED") -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=int)
    grid[2:6, 2:6] = seed % 4
    grid[0, :] = seed % 8  # A moving HUD is visible, but is not a win signal.
    return SimpleNamespace(
        frame=[grid.tolist()],
        levels_completed=level,
        state=state,
        available_actions=[1, 2, 3, 4, 5],
        score=0,
    )


def test_negative_history_never_confirms_goal() -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: no-win effects remain hypotheses."""
    probe = ArcProbeProtocol("guided")
    for seed in range(4):
        before = frame(seed)
        probe.record_action(before, (seed % 2 + 1, None))
        probe.observe(frame(seed + 1))
    diagnostics = probe.diagnostics()
    assert 0 < len(diagnostics["goal_hypotheses"]) <= 16
    assert diagnostics["goal_confirmations"] == 0
    assert all(row["state"] != "confirmed" for row in diagnostics["goal_hypotheses"])
    assert diagnostics["virtual_engine_calls"] == 0


def test_rejected_probe_falls_back_to_legal_policy() -> None:
    """SCENARIO-ARC-PROBE-7680-ROUTE: no support keeps the legal base action."""
    probe = ArcProbeProtocol("guided")
    assert probe.select(frame(0), (1, 2, 3), (2, None)) == (2, None)
    assert probe.diagnostics()["decisions"][-1]["admitted"] is False
    assert probe.select(frame(0), (1, 2), (5, None)) in {(1, None), (2, None)}


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "case", ["ordinary", "no_engine", "all_refuted", "irreversible", "hud", "stale"]
)
def test_three_arms_execute_scored_wrapper(monkeypatch, seed: int, case: str) -> None:
    """SCENARIO-REPORT-7680-ROWS: 48 groups, three actual wrapper arms each."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")

    class Base:
        game_id = "unseen"

        def __init__(self) -> None:
            pass

    for mode in ("current", "novelty", "guided"):
        agent = make_carnot_agent(Base, arc_probe_protocol=mode)()
        policy = agent._policy
        assert isinstance(policy, E3AgentPolicy)
        first = frame(seed)
        action0 = agent.choose_action([], first)
        assert action0 is not None
        second = first if case == "stale" else frame(seed + 1)
        action1 = agent.choose_action([first], second)
        assert action1 is not None
        third = frame(seed + 2, level=int(case == "irreversible"))
        action2 = agent.choose_action([first, second], third)
        assert action2 is not None
        diagnostics = policy.arc_probe_diagnostics()
        if mode == "current":
            assert diagnostics["enabled"] is False
        else:
            assert diagnostics["enabled"] is True
            assert diagnostics["goal_confirmations"] == 0
            assert diagnostics["virtual_engine_calls"] == 0


def test_checkpoint_roundtrip_preserves_pending_probe() -> None:
    """SCENARIO-ARC-PROBE-7680-ROUTE: next observation joins restored action."""
    probe = ArcProbeProtocol("guided")
    probe.record_action(frame(0), (1, None))
    restored = ArcProbeProtocol.from_checkpoint(probe.checkpoint())
    restored.observe(frame(1))
    assert restored.diagnostics()["observed_actions"] == 1


def test_admitted_probe_changes_scored_action_and_reloads(monkeypatch) -> None:
    """SCENARIO-ARC-PROBE-7680-ROUTE: action 3 uses the observed split."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")

    class Base:
        game_id = "unseen"

        def __init__(self) -> None:
            pass

    agents = {
        mode: make_carnot_agent(Base, arc_probe_protocol=mode)() for mode in ("current", "guided")
    }
    actions = {}
    for mode, agent in agents.items():
        actions[mode] = [
            agent.choose_action([frame(j) for j in range(i)], frame(i)) for i in range(4)
        ]
    probe = agents["guided"]._policy._arc_probe_protocol
    assert any(row["admitted"] for row in probe.diagnostics()["decisions"])
    assert actions["guided"][2] != actions["current"][2]
    restored = ArcProbeProtocol.from_checkpoint(probe.checkpoint())
    assert restored.diagnostics()["observed_actions"] == probe.observed_actions


def test_cold_reducer_requires_three_distinct_arms() -> None:
    """SCENARIO-REPORT-7680-ROWS: views cannot enlarge the group count."""
    rows = [
        {
            "group_id": "case-0",
            "arm": arm,
            "raw_metrics": {"admitted": int(arm == "guided"), "false_confirmation": 0},
        }
        for arm in ("current", "novelty", "guided")
    ]
    assert cold_reduce_rows(rows)["independent_groups"] == 1
    assert cold_reduce_rows(rows)["admitted_probes"] == 1
    with pytest.raises(ValueError, match="arms"):
        cold_reduce_rows(rows[:2])


def test_malformed_and_stale_observations_abstain() -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: bad SDK data adds no false evidence."""
    with pytest.raises(ValueError, match="mode"):
        ArcProbeProtocol("oracle")
    probe = ArcProbeProtocol()
    invalid = SimpleNamespace(frame=[], levels_completed=0)
    assert probe_module._visible_grid(invalid) is None
    assert probe_module._visible_grid(SimpleNamespace(frame=[[[0] * 65] * 65])) is None
    assert probe_module._level(SimpleNamespace(levels_completed="bad")) is None
    probe.record_action(invalid, (1, None))
    assert probe.pending is None
    probe.record_action(frame(0), ("RESET", None))
    assert probe.pending is None
    before = frame(0)
    probe.record_action(before, (1, None))
    probe.observe(SimpleNamespace(frame=[], levels_completed=0))
    assert probe.observed_actions == 0
    probe.record_action(before, (1, None))
    probe.observe(frame(1, level=-1))
    assert probe.observed_actions == 1
    assert probe.effects == []


def test_all_refuted_goal_hypotheses_fall_back() -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: observed non-wins can refute, not prove."""
    probe = ArcProbeProtocol()
    frames = []
    for count in range(3):
        grid = np.zeros((8, 8), dtype=int)
        grid.flat[:count] = 1
        frames.append(SimpleNamespace(frame=[grid.tolist()], levels_completed=0))
    probe.record_action(frames[0], (1, None))
    probe.observe(frames[1])
    probe.record_action(frames[1], (1, None))
    probe.observe(frames[2])
    assert probe.goal_hypotheses
    assert all(row["state"] == "refuted" for row in probe.goal_hypotheses)
    assert probe.select(frames[2], (1, 2), (2, None)) == (2, None)
    assert probe.diagnostics()["goal_confirmations"] == 0


def test_legal_resource_and_time_bounds(monkeypatch) -> None:
    """SCENARIO-ARC-PROBE-7680-ROUTE: exhausted resources keep a legal fallback."""
    probe = ArcProbeProtocol()
    assert probe.select(frame(0), (), ("RESET", None)) == ("RESET", None)
    assert probe.decisions[-1]["reason"] == "no_legal_candidates"
    assert probe.select(SimpleNamespace(frame=[], levels_completed=0), (1,), (1, None)) == (1, None)
    assert probe.decisions[-1]["reason"] == "unreadable_observation"
    assert probe.select(frame(0), (1,), (1, {"x": 1})) == (1, {"x": 1})
    assert probe.decisions[-1]["reason"] == "structured_action_fallback"
    probe.decisions = [{"admitted": False}] * 8
    assert probe.select(frame(0), (1,), (1, None)) == (1, None)
    assert probe.decisions[-1]["reason"] == "probe_budget_exhausted"
    probe.decisions.clear()
    ticks = iter((0.0, 0.02))
    monkeypatch.setattr(probe_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert probe.select(frame(0), (1,), (1, None)) == (1, None)
    assert probe.decisions[-1]["reason"] == "decision_time_exceeded"


def test_level_up_starts_new_episode_without_goal_exemplar() -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: returned frame opens the next level."""
    probe = ArcProbeProtocol()
    probe.record_action(frame(0), (1, None))
    probe.observe(frame(1))
    assert probe.goal_hypotheses
    probe.record_action(frame(1), (2, None))
    probe.observe(frame(2, level=1))
    assert probe.sdk_progress == 1
    assert probe.goal_hypotheses == []
    assert probe.goal_confirmations == 0


def test_false_terminal_is_guarded_on_scored_policy(monkeypatch) -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: predicted goal cannot outrank SDK."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    policy = E3AgentPolicy("unseen", goal_confirmation=True, arc_probe_protocol="guided")
    first = frame(0)
    policy.next_move([], first)
    policy._goal_confirmation.arm(first, frames_seen=0, level=0, predicted_goal=True, plan_length=1)
    policy.next_move([first], frame(1))
    assert policy.goal_confirmation_receipts()[-1]["status"] == "contradiction"
    assert policy.arc_probe_diagnostics()["goal_confirmations"] == 0


def test_all_refuted_fixture_really_exhausts_bank() -> None:
    """SCENARIO-ARC-PROBE-7680-UNKNOWN: the named fixture has no live rule."""
    probe = ArcProbeProtocol()
    frames = [fixture_frame(3, step, "all_refuted") for step in range(4)]
    for step in range(3):
        probe.record_action(frames[step], (1, None))
        probe.observe(frames[step + 1])
    assert probe.goal_hypotheses
    assert all(row["state"] == "refuted" for row in probe.goal_hypotheses)
