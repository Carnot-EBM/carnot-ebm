"""Tests for the Kakuro WOPR cartridge.

Spec traces: REQ-KAKURO-001, REQ-KAKURO-002, SCENARIO-KAKURO-001,
SCENARIO-KAKURO-002, SCENARIO-KAKURO-003.
"""

from __future__ import annotations

import os
import sys

_WOPR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, _WOPR_DIR)

from games.kakuro import KakuroGame  # noqa: E402


VALID_SOLUTION = [
    [0, 1, 3, 0],
    [0, 4, 2, 6],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]


def _state_with_cells(game: KakuroGame, cells: list[list[int]]) -> dict[str, object]:
    state = game.initial_state()
    state["cells"] = [row[:] for row in cells]
    return state


def _run_digits(state: dict[str, object], clue: dict[str, object]) -> list[int]:
    cells = state["cells"]
    return [cells[row][col] for row, col in clue["cells"]]


def test_valid_solution_energy_zero():
    """SCENARIO-KAKURO-001 / REQ-KAKURO-001: valid sums have E=0."""
    game = KakuroGame()
    state = _state_with_cells(game, VALID_SOLUTION)

    assert game.energy(state) == 0.0
    assert game.is_solved(state)
    assert "R4" in game.visualize(state, game.energy(state))


def test_invalid_energy_positive():
    """SCENARIO-KAKURO-002 / REQ-KAKURO-001: mismatched sums have E>0."""
    game = KakuroGame()
    state = game.initial_state()

    assert game.energy(state) > 0.0
    assert not game.is_solved(state)


def test_carnot_step_valid_digits():
    """SCENARIO-KAKURO-003 / REQ-KAKURO-002: proposals stay in domain."""
    game = KakuroGame()
    state = game.initial_state()

    for iteration in range(20):
        result = game.carnot_step(state, iteration)
        state = result.state

        for clue in state["clues"]:
            digits = _run_digits(state, clue)
            assert all(1 <= digit <= 9 for digit in digits)
            assert len(digits) == len(set(digits))
        assert result.energy == game.energy(state)
