"""Tests for the Slitherlink WOPR cartridge.

Spec traces: REQ-SLITHERLINK-001 and REQ-SAMPLE-003. The cartridge must
encode every 3x3 dot-grid edge as an Ising spin, penalize clue/degree
violations, reject the empty loop, and converge to E=0 on the canonical
diamond puzzle.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_WOPR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, _WOPR_DIR)

from games import ALL_GAMES  # noqa: E402
from games.slitherlink import (  # noqa: E402
    CANONICAL_SLITHERLINK_PUZZLE,
    CANONICAL_SLITHERLINK_SOLUTION,
    SlitherinkCartridge,
    SlitherlinkGame,
)


def test_energy_at_known_solution_is_zero():
    """REQ-SLITHERLINK-001: a valid canonical loop has zero energy."""
    cartridge = SlitherinkCartridge(CANONICAL_SLITHERLINK_PUZZLE, (3, 3))
    spins = np.asarray(CANONICAL_SLITHERLINK_SOLUTION, dtype=np.int8)

    assert cartridge.n_spins == 24
    assert cartridge.energy(spins) == 0.0
    assert cartridge.is_single_loop(spins)
    assert "SLITHERLINK" in cartridge.to_display(spins)


def test_energy_at_empty_is_nonzero():
    """REQ-SLITHERLINK-001: the all-off spin vector receives the empty penalty."""
    cartridge = SlitherinkCartridge(CANONICAL_SLITHERLINK_PUZZLE, (3, 3))
    empty = np.full(cartridge.n_spins, -1, dtype=np.int8)

    assert cartridge.energy(empty) >= 1000.0
    assert not cartridge.is_single_loop(empty)
    with pytest.raises(ValueError, match="Expected 24 spins"):
        cartridge.energy([-1])


def test_degree_two_dot_has_zero_penalty():
    """REQ-SLITHERLINK-001: a closed 1x1 square gives every dot degree 2."""
    cartridge = SlitherinkCartridge([[None]], (1, 1))
    square = np.ones(cartridge.n_spins, dtype=np.int8)
    searched = cartridge.sample(n_steps=20)

    assert cartridge.n_spins == 4
    assert cartridge.energy(square) == 0.0
    assert cartridge.energy(searched) == 0.0


def test_degree_one_dot_adds_energy():
    """REQ-SLITHERLINK-001: a dangling edge adds degree-violation energy."""
    cartridge = SlitherinkCartridge([[None]], (1, 1))
    impossible = SlitherinkCartridge([[0]], (1, 1))
    dangling = np.full(cartridge.n_spins, -1, dtype=np.int8)
    dangling[0] = 1
    best_limited = impossible.sample(n_steps=4)

    assert cartridge.energy(dangling) == 10.0
    assert not cartridge.is_single_loop(dangling)
    assert impossible.energy(best_limited) > 0.0


def test_sample_converges():
    """REQ-SAMPLE-003: the WOPR cartridge samples an E=0 canonical loop."""
    cartridge = SlitherinkCartridge(CANONICAL_SLITHERLINK_PUZZLE, (3, 3))
    spins = cartridge.sample(n_steps=5000)
    game = SlitherlinkGame()
    steps = game.carnot_solve(max_iterations=5000)
    repeated = game.carnot_step(steps[-1].state, steps[-1].iteration + 1)
    idle_state, moved_edge = game._move_one_edge_toward_target(steps[-1].state)
    html = game.visualize(steps[-1].state, steps[-1].energy)

    assert cartridge.energy(spins) == 0.0
    assert cartridge.last_iterations_to_convergence <= 5000
    assert any(isinstance(candidate, SlitherlinkGame) for candidate in ALL_GAMES)
    assert steps[-1].energy == 0.0
    assert steps[-1].is_solved
    assert repeated.is_solved
    assert idle_state is steps[-1].state
    assert moved_edge is None
    assert "SLITHERLINK LOOP CLOSED" in html
