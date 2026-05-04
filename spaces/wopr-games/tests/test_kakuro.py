"""Tests for the Kakuro WOPR cartridge.

Spec traces: REQ-KAKURO-001, REQ-KAKURO-002, SCENARIO-KAKURO-001,
SCENARIO-KAKURO-002, and SCENARIO-KAKURO-003.
"""

from __future__ import annotations

import os
import sys

import pytest

_WOPR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, _WOPR_DIR)

from games import ALL_GAMES  # noqa: E402
from games.kakuro import (  # noqa: E402
    CANONICAL_KAKURO_INITIAL,
    CANONICAL_KAKURO_SOLUTION,
    KakuroGame,
)


def _values_in_domain(state) -> bool:
    return all(1 <= value <= 9 for value in state.values.values())


@pytest.mark.parametrize(
    ("entries", "expected_energy"),
    [
        (CANONICAL_KAKURO_SOLUTION, 0.0),
        (((1, 1, 2), (1, 2, 1), (2, 2, 5), (2, 3, 3)), 3.0),
        (((1, 1, 9), (1, 2, 3), (2, 2, 4), (2, 3, 5)), 64.0),
    ],
)
def test_parametrized_sum_run_energy(entries, expected_energy):
    """REQ-KAKURO-001: sum-run energy matches known valid and invalid boards."""
    game = KakuroGame()
    state = game.state_from_entries(entries)

    assert game.energy(state) == expected_energy


def test_valid_solution_energy_zero():
    """SCENARIO-KAKURO-001: the known valid solution has E=0."""
    game = KakuroGame()
    solution = game.solved_state()

    assert game.energy(solution) == 0.0
    assert game.is_solved(solution)
    assert any(isinstance(candidate, KakuroGame) for candidate in ALL_GAMES)


def test_invalid_solution_energy_positive():
    """SCENARIO-KAKURO-002: a deterministic invalid board has positive energy."""
    game = KakuroGame()
    invalid = game.state_from_entries(CANONICAL_KAKURO_INITIAL)

    assert game.energy(invalid) > 0.0
    assert not game.is_solved(invalid)


def test_carnot_step_returns_valid_state():
    """SCENARIO-KAKURO-003: Metropolis proposals stay in 1..9 with no repeats."""
    game = KakuroGame()
    state = game.initial_state()

    for iteration in range(25):
        result = game.carnot_step(state, iteration)
        state = result.state
        assert _values_in_domain(state)
        assert game.runs_have_unique_digits(state)
        assert result.energy == game.energy(state)


def test_is_solved_matches_energy():
    """REQ-KAKURO-001: solved status is exactly equivalent to E=0."""
    game = KakuroGame()

    for entries in (
        CANONICAL_KAKURO_SOLUTION,
        CANONICAL_KAKURO_INITIAL,
        ((1, 1, 7), (1, 2, 1), (2, 2, 6), (2, 3, 2)),
    ):
        state = game.state_from_entries(entries)
        assert game.is_solved(state) is (game.energy(state) == 0.0)


def test_visualize_includes_clues_and_values():
    """REQ-KAKURO-002: WOPR rendering includes clue targets and live digits."""
    game = KakuroGame()
    state = game.solved_state()
    html = game.visualize(state, game.energy(state))

    assert "R4" in html
    assert "D7" in html
    assert "1" in html
    assert "5" in html
