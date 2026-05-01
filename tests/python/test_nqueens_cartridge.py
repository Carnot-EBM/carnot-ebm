"""Tests for the N-Queens WOPR cartridge.

Spec traces: REQ-SAMPLE-003 and REQ-NQUEENS-001. The cartridge must expose
the 8x8 CSP as a 64-spin Ising coupling matrix and reach a zero-conflict
8-queen placement within 50000 WOPR iterations.
"""

import os
import sys

import numpy as np

_WOPR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spaces", "wopr-games")
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_WOPR_DIR))

from games.nqueens import (  # noqa: E402
    BOARD_SIZE,
    N_SPINS,
    NQueensGame,
    build_nqueens_coupling_matrix,
    columns_to_spins,
    nqueens_energy,
)


def test_nqueens_coupling_matrix_has_correct_row_constraints():
    """REQ-NQUEENS-001: each same-row queen pair has antiferromagnetic coupling."""
    J = build_nqueens_coupling_matrix()

    assert J.shape == (N_SPINS, N_SPINS)
    assert np.allclose(np.diag(J), 0.0)
    for row in range(BOARD_SIZE):
        for col_a in range(BOARD_SIZE):
            for col_b in range(col_a + 1, BOARD_SIZE):
                a = row * BOARD_SIZE + col_a
                b = row * BOARD_SIZE + col_b
                assert J[a, b] == -1.0
                assert J[b, a] == -1.0


def test_nqueens_coupling_matrix_has_correct_column_constraints():
    """REQ-NQUEENS-001: each same-column queen pair has antiferromagnetic coupling."""
    J = build_nqueens_coupling_matrix()

    for col in range(BOARD_SIZE):
        for row_a in range(BOARD_SIZE):
            for row_b in range(row_a + 1, BOARD_SIZE):
                a = row_a * BOARD_SIZE + col
                b = row_b * BOARD_SIZE + col
                assert J[a, b] == -1.0
                assert J[b, a] == -1.0


def test_nqueens_coupling_matrix_has_correct_diagonal_constraints():
    """REQ-NQUEENS-001: diagonal attacks are encoded and non-attacks stay uncoupled."""
    J = build_nqueens_coupling_matrix()

    assert J[0 * BOARD_SIZE + 0, 1 * BOARD_SIZE + 1] == -1.0
    assert J[0 * BOARD_SIZE + 7, 1 * BOARD_SIZE + 6] == -1.0
    assert J[0 * BOARD_SIZE + 0, 1 * BOARD_SIZE + 2] == 0.0


def test_nqueens_energy_zero_for_known_valid_8queens_solution():
    """REQ-NQUEENS-001: a known valid 8-queen placement has zero Ising energy."""
    J = build_nqueens_coupling_matrix()
    valid_columns = [0, 4, 7, 5, 2, 6, 1, 3]
    spins = columns_to_spins(valid_columns)

    assert int(spins.sum()) == BOARD_SIZE
    assert nqueens_energy(spins, J) == 0.0


def test_nqueens_energy_positive_for_invalid_placement():
    """REQ-NQUEENS-001: an attacking placement has positive conflict energy."""
    J = build_nqueens_coupling_matrix()
    invalid_columns = [0, 0, 7, 5, 2, 6, 1, 3]
    spins = columns_to_spins(invalid_columns)

    assert nqueens_energy(spins, J) > 0.0


def test_nqueens_ising_reaches_ground_state_within_50000_iters():
    """REQ-SAMPLE-003: the WOPR cartridge reaches E=0 within 50000 iterations."""
    game = NQueensGame(seed=17)
    steps = game.carnot_solve(max_iterations=50000)

    assert steps, "carnot_solve returned no steps"
    assert game.ising_solver_used, "ParallelIsingSampler was not invoked"
    assert steps[-1].energy == 0.0
    assert steps[-1].is_solved
    assert steps[-1].iteration < 50000
    assert int(np.asarray(steps[-1].state.spins).sum()) == BOARD_SIZE
