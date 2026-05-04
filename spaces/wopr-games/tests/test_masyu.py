"""Tests for the Masyu WOPR cartridge.

Spec traces: REQ-MASYU-001, SCENARIO-MASYU-001, and SCENARIO-MASYU-002.
"""

from __future__ import annotations

import os
import sys

_WOPR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, _WOPR_DIR)

from games import ALL_GAMES  # noqa: E402
from games.masyu import (  # noqa: E402
    CANONICAL_MASYU_SOLUTION,
    MasyuGame,
    canonical_edge,
)


VIOLATING_CLOSED_LOOP = frozenset(
    {
        canonical_edge((0, 0), (0, 1)),
        canonical_edge((0, 1), (1, 1)),
        canonical_edge((1, 1), (1, 0)),
        canonical_edge((1, 0), (0, 0)),
    }
)


def test_valid_loop_energy_zero():
    """SCENARIO-MASYU-001 / REQ-MASYU-001: bundled loop has E=0."""
    game = MasyuGame()
    state = frozenset(CANONICAL_MASYU_SOLUTION)

    assert game.energy(state) == 0.0
    assert game.is_solved(state)
    assert game.black_violations(state) == 0
    assert game.white_violations(state) == 0
    assert game.connectivity_violations(state) == 0
    assert "MASYU LOOP CLOSED" in game.visualize(state, game.energy(state))
    assert any(isinstance(candidate, MasyuGame) for candidate in ALL_GAMES)


def test_violated_circle_energy_positive():
    """SCENARIO-MASYU-002 / REQ-MASYU-001: closed bad loop has E>0."""
    game = MasyuGame()

    assert game.connectivity_violations(VIOLATING_CLOSED_LOOP) == 0
    assert game.black_violations(VIOLATING_CLOSED_LOOP) > 0
    assert game.white_violations(VIOLATING_CLOSED_LOOP) > 0
    assert game.energy(VIOLATING_CLOSED_LOOP) > 0.0
    assert not game.is_solved(VIOLATING_CLOSED_LOOP)


def test_is_solved_iff_energy_zero():
    """REQ-MASYU-001: solving is equivalent to zero-energy closed loop."""
    game = MasyuGame()
    initial = game.initial_state()
    target = frozenset(CANONICAL_MASYU_SOLUTION)

    for state in (initial, VIOLATING_CLOSED_LOOP, target):
        assert game.is_solved(state) is (game.energy(state) == 0.0)

    action = game.available_actions(initial)[0]
    toggled = game.apply_action(initial, action)
    restored = game.apply_action(toggled, action)
    step = game.carnot_step(initial, 0)
    solved_step = game.carnot_step(target, 1)

    assert len(toggled) == 1
    assert restored == initial
    assert len(step.state.symmetric_difference(initial)) == 1
    assert step.energy == game.energy(step.state)
    assert not step.is_solved
    assert solved_step.state == target
    assert solved_step.energy == 0.0
    assert solved_step.is_solved
