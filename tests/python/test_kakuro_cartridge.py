"""Tests for the minimal Kakuro WOPR cartridge.

Spec traces: REQ-KAKURO-001, REQ-KAKURO-002,
SCENARIO-KAKURO-001, SCENARIO-KAKURO-002, SCENARIO-KAKURO-003.
"""

from __future__ import annotations

import os
import sys

_WOPR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spaces", "wopr-games")
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_WOPR_DIR))

from games.kakuro import CANONICAL_KAKURO_SOLUTION, KakuroGame  # noqa: E402


def _state_with_cells(game: KakuroGame, cells: list[list[int]]) -> dict[str, object]:
    state = game.initial_state()
    state["cells"] = [row[:] for row in cells]
    return state


def _run_digits(state: dict[str, object], clue: dict[str, object]) -> list[int]:
    cells = state["cells"]
    return [cells[row][col] for row, col in clue["cells"]]


def test_known_solution_energy_zero() -> None:
    """SCENARIO-KAKURO-001 / REQ-KAKURO-001: valid sums have E=0."""
    game = KakuroGame()
    state = _state_with_cells(game, CANONICAL_KAKURO_SOLUTION)

    assert game.energy(state) == 0.0
    assert game.is_solved(state)


def test_invalid_solution_energy_positive() -> None:
    """SCENARIO-KAKURO-002 / REQ-KAKURO-001: invalid sums have E>0."""
    game = KakuroGame()
    state = game.initial_state()

    assert game.energy(state) > 0.0
    assert not game.is_solved(state)


def test_step_stays_in_digit_domain_without_run_duplicates() -> None:
    """SCENARIO-KAKURO-003 / REQ-KAKURO-002: proposals stay run-valid."""
    game = KakuroGame()
    state = game.initial_state()

    for iteration in range(24):
        result = game.carnot_step(state, iteration)
        state = result.state

        for clue in state["clues"]:
            digits = _run_digits(state, clue)
            assert all(1 <= digit <= 9 for digit in digits)
            assert len(digits) == len(set(digits))
        assert result.energy == game.energy(state)


def test_visualization_contains_clues_and_digits() -> None:
    """REQ-KAKURO-002: visualization includes clue targets and current values."""
    game = KakuroGame()
    state = _state_with_cells(game, CANONICAL_KAKURO_SOLUTION)
    html = game.visualize(state, game.energy(state))

    assert "R4" in html
    assert "C5" in html
    assert "R12" in html
    assert "6" in html
