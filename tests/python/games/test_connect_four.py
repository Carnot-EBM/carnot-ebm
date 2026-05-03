"""Tests for the Connect Four WOPR Ising cartridge.

Spec traces: REQ-CONNECT4-001, REQ-CONNECT4-002, REQ-CONNECT4-003,
SCENARIO-CONNECT4-001, SCENARIO-CONNECT4-002, and SCENARIO-CONNECT4-003.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.games.connect_four import ConnectFourIsingCartridge


def test_empty_board_energy_zero():
    """SCENARIO-CONNECT4-001: the empty spin state is a ground state."""
    cartridge = ConnectFourIsingCartridge(initial_pieces=0)
    empty_spins = np.full(cartridge.n_spins, -1, dtype=np.int8)

    sampled = cartridge.sample()

    assert cartridge.n_spins == 42
    assert cartridge.energy(empty_spins) == 0.0
    assert sampled.shape == (cartridge.BOARD_ROWS, cartridge.BOARD_COLS)
    assert int(sampled.sum()) == 0
    assert cartridge.is_valid(sampled)


def test_valid_board_energy_zero():
    """REQ-CONNECT4-001: a compact board with conserved count has E=0."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:3] = [1, 2, 1]
    board[4, 0] = 2
    cartridge = ConnectFourIsingCartridge(initial_board=board)

    sampled = cartridge.sample()

    assert cartridge.energy(board) == 0.0
    assert cartridge.is_valid(board)
    assert np.array_equal(sampled, board)


def test_gravity_violation_nonzero():
    """SCENARIO-CONNECT4-002: a floating piece is penalized and repaired."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[2, 3] = 1
    cartridge = ConnectFourIsingCartridge(initial_board=board)

    repaired = cartridge.sample(n_steps=1000, beta=2.0)

    assert cartridge.energy(board) > 0.0
    assert not cartridge.is_valid(board)
    assert cartridge.is_valid(repaired)
    assert cartridge.energy(repaired) == 0.0
    assert int((repaired != 0).sum()) == 1
    assert repaired[5, 3] == 1


def test_winner_detection_horizontal():
    """SCENARIO-CONNECT4-003: four adjacent RED cells win horizontally."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:4] = 1
    cartridge = ConnectFourIsingCartridge(initial_board=board)

    assert cartridge.check_winner(board) == "RED"


def test_winner_detection_vertical():
    """SCENARIO-CONNECT4-003: four stacked YELLOW cells win vertically."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[2:6, 4] = 2
    cartridge = ConnectFourIsingCartridge(initial_board=board)

    assert cartridge.check_winner(board) == "YELLOW"


def test_winner_detection_diagonal():
    """SCENARIO-CONNECT4-003: diagonal four-in-a-row is detected."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[4, 1] = 1
    board[3, 2] = 1
    board[2, 3] = 1
    cartridge = ConnectFourIsingCartridge(initial_board=board)

    assert cartridge.check_winner(board) == "RED"


def test_piece_count_mismatch_nonzero():
    """REQ-CONNECT4-001: violating piece conservation adds energy."""
    cartridge = ConnectFourIsingCartridge(initial_pieces=2)
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1

    assert cartridge.energy(board) > 0.0
    assert not cartridge.is_valid(board)


def test_draw_and_ongoing_detection():
    """REQ-CONNECT4-002: full no-winner boards draw; partial boards continue."""
    cartridge = ConnectFourIsingCartridge(initial_pieces=42)
    draw_board = np.asarray(
        [
            [2, 2, 2, 1, 1, 1, 2],
            [2, 1, 2, 2, 2, 1, 2],
            [2, 2, 1, 2, 1, 1, 1],
            [1, 1, 1, 2, 1, 2, 2],
            [2, 2, 1, 1, 1, 2, 2],
            [2, 1, 2, 2, 2, 1, 2],
        ],
        dtype=np.int8,
    )
    ongoing_board = np.zeros((6, 7), dtype=np.int8)
    ongoing_board[5, 0:2] = [1, 2]

    assert cartridge.check_winner(draw_board) == "DRAW"
    assert cartridge.check_winner(ongoing_board) == "ONGOING"


def test_invalid_shapes_and_piece_counts_are_rejected():
    """REQ-CONNECT4-001: malformed inputs fail before scoring."""
    with pytest.raises(ValueError, match="initial_pieces"):
        ConnectFourIsingCartridge(initial_pieces=43)

    cartridge = ConnectFourIsingCartridge(initial_pieces=0)
    with pytest.raises(ValueError, match="Expected"):
        cartridge.energy(np.zeros((3, 3), dtype=np.int8))


def test_spin_winner_and_generated_full_board():
    """REQ-CONNECT4-002: spin boards map +1 to RED and generated boards fill."""
    spin_board = np.full((6, 7), -1, dtype=np.int8)
    spin_board[5, 0:4] = 1
    cartridge = ConnectFourIsingCartridge(initial_pieces=4)
    full_cartridge = ConnectFourIsingCartridge(initial_pieces=42)

    full_board = full_cartridge.sample()

    assert cartridge.check_winner(spin_board) == "RED"
    assert full_board.shape == (6, 7)
    assert int((full_board != 0).sum()) == 42
    assert full_cartridge.energy(full_board) == 0.0
