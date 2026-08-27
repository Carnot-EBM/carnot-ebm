"""Regression tests for the make_carnot_agent ADAPTER (the Kaggle submission shape).

Why this file exists: the offline validators (scripts/arc_competition_validate.py,
scripts/arc_leaderboard_eval.py) drive the POLICY directly via `policy.next_move(...)`
and then call `env.step(GameAction.ACTIONk, data=data)`. They NEVER exercise the
`make_carnot_agent` adapter's `choose_action`, which is what the REAL ARC-AGI-3-Agents
framework calls. That left two adapter bugs invisible to the whole offline suite until a
pre-submission diff against the live framework (agents/agent.py) surfaced them:

  1. choose_action returned `act.set_data(data)` — but GameAction.set_data() returns the
     inner ComplexAction, NOT the enum member. The framework's do_action_request reads
     `action.action_data.model_dump()` off whatever choose_action returns, so returning a
     ComplexAction crashes EVERY coordinate/click action on the real harness.
  2. The framework's Agent.MAX_ACTIONS default is 80 — too low even for our deepest banked
     replay (lp85 -> L5), so an unmodified submission truncated our best known game.

These tests pin both fixes using the REAL arcengine.GameAction enum, mirroring exactly how
agents/agent.py:do_action_request consumes the returned action.
"""

import pytest

from carnot.agentic.arc_competition_agent import make_carnot_agent

GameAction = pytest.importorskip("arcengine").GameAction


class _FakeBase:
    """Stand-in for the framework's Agent base (we don't need the real env/harness here)."""

    def __init__(self, *a, **k) -> None:
        self.game_id = "lp85"

    def do_action_request(self, action):
        return ("framework", action)


class _StubPolicy:
    def __init__(self, moves):
        self._it = iter(moves)

    def next_move(self, frames, latest):
        return next(self._it)

    def is_done(self, frames, latest):
        return False


def _agent_with_moves(moves):
    """Build a CarnotAgent and inject a stub policy, skipping the real (disk/jax) __init__
    so the test isolates the adapter's choose_action transform."""
    cls = make_carnot_agent(_FakeBase)
    agent = object.__new__(cls)  # bypass __init__: no policy/disk/model load
    agent.game_id = "lp85"
    agent._policy = _StubPolicy(moves)
    return agent


def _harness_consume(action):
    """Exactly what agents/agent.py:do_action_request does with choose_action's return."""
    return action.action_data.model_dump()


def test_click_action_returns_enum_not_complexaction():
    # REQ-ARC-WMTE-6681: optional instrumentation must stay inactive when the
    # policy does not expose its outcome-transport hook, while choose_action
    # still returns the GameAction ENUM so the harness can read
    # action.action_data — returning set_data()'s ComplexAction crashes the harness.
    agent = _agent_with_moves([(6, {"x": 3, "y": 4})])
    act = agent.choose_action([], None)
    assert isinstance(act, GameAction)
    assert act is GameAction.ACTION6
    payload = _harness_consume(act)  # must NOT raise (the old bug raised AttributeError)
    assert payload["x"] == 3 and payload["y"] == 4
    assert payload["game_id"] == "lp85"  # required ComplexAction field, carried through


def test_reset_and_bare_actions_consumable():
    agent = _agent_with_moves([("RESET", None), (1, None)])
    assert agent.choose_action([], None) is GameAction.RESET
    a1 = agent.choose_action([], None)
    assert a1 is GameAction.ACTION1
    assert _harness_consume(a1) == {"game_id": ""}  # bare SimpleAction, harness-consumable


def test_none_kind_falls_back_to_reset():
    agent = _agent_with_moves([(None, None)])
    assert agent.choose_action([], None) is GameAction.RESET


def test_missing_optional_transport_hook_uses_framework_step():
    """REQ-ARC-WMTE-6681 keeps an uninstrumented adapter on the base seam."""

    agent = _agent_with_moves([])
    action = object()
    assert agent.do_action_request(action) == ("framework", action)


def test_max_actions_overrides_framework_default():
    # REQ: the framework's Agent.MAX_ACTIONS default (80) truncates lp85->L5; the adapter
    # must raise the per-game cap so multi-level replays + explore have room.
    cls = make_carnot_agent(_FakeBase)
    assert cls.MAX_ACTIONS >= 200, "per-game cap must exceed the framework's 80 default"
