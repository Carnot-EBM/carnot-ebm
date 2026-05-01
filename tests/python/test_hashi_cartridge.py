"""Tests for the Hashi WOPR cartridge.

Spec traces: REQ-SAMPLE-003 and REQ-HASHI-001. The cartridge must expose a
Hashiwokakero bridge puzzle as a binary Ising-style model and converge to a
zero-energy connected solution for the canonical 5x5 puzzle.
"""

import os
import sys

import numpy as np

_WOPR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spaces", "wopr-games")
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_WOPR_DIR))

from games import ALL_GAMES  # noqa: E402
from games.hashi import (  # noqa: E402
    CANONICAL_HASHI_PUZZLE,
    HashiCartridge,
    HashiGame,
    build_hashi_model,
    hashi_energy,
    is_valid_hashi_solution,
    spins_to_binary,
)


def test_hashi_canonical_model_has_expected_bridge_variables():
    """REQ-HASHI-001: canonical islands produce one spin per possible bridge."""
    model = build_hashi_model(CANONICAL_HASHI_PUZZLE)

    assert len(model.islands) == 9
    assert len(model.edges) == 12
    assert sum(edge.orientation == "H" for edge in model.edges) == 6
    assert sum(edge.orientation == "V" for edge in model.edges) == 6
    assert model.crossing_pairs == ()


def test_hashi_crossing_pair_adds_positive_energy():
    """REQ-HASHI-001: orthogonal bridges crossing an empty cell are penalized."""
    crossing_puzzle = [
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    model = build_hashi_model(crossing_puzzle)
    all_present = np.ones(len(model.edges), dtype=np.int8)

    assert len(model.edges) == 2
    assert len(model.crossing_pairs) == 1
    assert hashi_energy(all_present, model, connectivity_weight=0.0) == 1.0


def test_hashi_e0_convergence():
    """REQ-HASHI-001: the canonical puzzle converges to an E=0 solution."""
    cartridge = HashiCartridge()
    solution, final_energy, n_iterations = cartridge.solve(CANONICAL_HASHI_PUZZLE)

    assert final_energy == 0.0
    assert n_iterations > 0
    assert is_valid_hashi_solution(solution.spins, solution.model)
    assert int(spins_to_binary(solution.spins).sum()) == 12


def test_hashi_game_registered_and_visualizes_solution():
    """REQ-HASHI-001: Hashi is registered as a WOPR game cartridge."""
    game = HashiGame()
    steps = game.carnot_solve(max_iterations=5000)
    html = game.visualize(steps[-1].state, steps[-1].energy)

    assert any(isinstance(cartridge, HashiGame) for cartridge in ALL_GAMES)
    assert steps[-1].energy == 0.0
    assert steps[-1].is_solved
    assert "HASHI NETWORK CONNECTED" in html
    assert "2-3-2" in html
