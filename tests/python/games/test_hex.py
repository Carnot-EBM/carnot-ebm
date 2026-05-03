"""Tests for the WOPR Hex cartridge.

Spec traces: REQ-HEX-001, REQ-HEX-002, REQ-HEX-003,
SCENARIO-HEX-001, SCENARIO-HEX-002, and SCENARIO-HEX-003.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.games.hex import (
    GibbsEnergyPlayer,
    GreedyEnergyPlayer,
    HexBoard,
    HexGame,
    RandomPlayer,
)


def test_hex_legal_moves():
    """SCENARIO-HEX-001: legal actions are exactly the empty cells."""
    game = HexGame(n=3)
    board = game.reset()
    board[0, 0] = game.BLACK
    board[1, 1] = game.WHITE

    actions = game.legal_actions(board)

    assert len(actions) == 7
    assert actions == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_hex_winner_detection():
    """SCENARIO-HEX-002: edge-to-edge connected chains win."""
    game = HexGame(n=3)
    black_win = np.asarray([[1, 0, 0], [1, 2, 0], [1, 2, 0]], dtype=np.int8)
    white_win = np.asarray([[2, 2, 2], [1, 1, 0], [0, 0, 0]], dtype=np.int8)

    assert game.check_winner(black_win) == game.BLACK
    assert game.check_winner(white_win) == game.WHITE


def test_hex_no_draws():
    """SCENARIO-HEX-003: complete legal Hex games always end with a winner."""
    game = HexGame(n=5)

    for seed in range(5):
        player = RandomPlayer(seed=seed)
        board = game.reset()
        done = False
        winner = None
        turn = game.BLACK

        while not done:
            action = player.select_action(game, board, turn)
            board, done, winner = game.step(board, action, turn)
            turn = game.WHITE if turn == game.BLACK else game.BLACK

        assert winner in (game.BLACK, game.WHITE)
        assert game.legal_actions(board) or game.check_winner(board) in (game.BLACK, game.WHITE)


def test_hex_step_validation_and_board_wrapper():
    """REQ-HEX-001: boards copy cleanly and invalid moves are rejected."""
    with pytest.raises(ValueError, match="positive"):
        HexBoard(0)
    with pytest.raises(ValueError, match="positive"):
        HexGame(0)
    with pytest.raises(ValueError, match="Expected"):
        HexBoard(2, np.zeros((3, 3), dtype=np.int8))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        HexBoard(2, [[0, 3], [1, 2]])

    wrapper = HexBoard(2, [[0, 1], [2, 0]])
    wrapper_copy = wrapper.copy()
    assert np.asarray(wrapper, dtype=np.int16).dtype == np.int16
    assert np.array_equal(wrapper.cells, wrapper_copy.cells)

    game = HexGame(n=2)
    board = game.reset()
    moved, done, winner = game.step(board, (0, 0), game.BLACK)

    assert board[0, 0] == 0
    assert moved[0, 0] == game.BLACK
    assert done is False
    assert winner is None

    with pytest.raises(ValueError, match="occupied"):
        game.step(moved, (0, 0), game.WHITE)
    with pytest.raises(ValueError, match="player"):
        game.step(board, (0, 1), 3)
    with pytest.raises(ValueError, match="outside"):
        game.step(board, (2, 0), game.BLACK)
    with pytest.raises(ValueError, match="Expected"):
        game.legal_actions(np.zeros((3, 3), dtype=np.int8))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        game.legal_actions(np.asarray([[0, 1], [2, 9]], dtype=np.int8))


def test_hex_energy_and_energy_players():
    """REQ-HEX-003: greedy and Gibbs players choose legal energy-minimizing moves."""
    game = HexGame(n=3)
    board = np.asarray([[1, 2, 0], [1, 2, 0], [0, 0, 0]], dtype=np.int8)
    greedy = GreedyEnergyPlayer()
    gibbs = GibbsEnergyPlayer(seed=7, n_steps=32)

    greedy_action = greedy.select_action(game, board, game.BLACK)
    gibbs_action = gibbs.select_action(game, board, game.BLACK)
    greedy_board, greedy_done, greedy_winner = game.step(board, greedy_action, game.BLACK)

    assert game.energy(board, game.BLACK) == -2.0
    assert game.path_strength(board, game.WHITE) == 1
    assert greedy_action == (2, 0)
    assert gibbs_action in game.legal_actions(board)
    assert game.energy_after_action(board, greedy_action, game.BLACK) == -3.0
    assert greedy_done is True
    assert greedy_winner == game.BLACK
    assert game.check_winner(greedy_board) == game.BLACK


def test_hex_player_edge_cases():
    """REQ-HEX-003: players report terminal-board errors and one-action Gibbs moves."""
    game = HexGame(n=1)
    empty = game.reset()
    full = np.asarray([[game.BLACK]], dtype=np.int8)
    gibbs = GibbsEnergyPlayer(seed=3)

    assert RandomPlayer(seed=1).select_action(game, empty, game.BLACK) == (0, 0)
    assert gibbs.select_action(game, empty, game.BLACK) == (0, 0)
    assert gibbs.last_diagnostics == {"n_candidates": 1.0, "best_free_energy": None}

    with pytest.raises(ValueError, match="RandomPlayer"):
        RandomPlayer(seed=1).select_action(game, full, game.BLACK)
    with pytest.raises(ValueError, match="GreedyEnergyPlayer"):
        GreedyEnergyPlayer().select_action(game, full, game.BLACK)
    with pytest.raises(ValueError, match="GibbsEnergyPlayer"):
        GibbsEnergyPlayer(seed=1).select_action(game, full, game.BLACK)
