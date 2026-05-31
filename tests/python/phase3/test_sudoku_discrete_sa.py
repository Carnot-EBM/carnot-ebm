"""Tests for the discrete simulated annealing Sudoku solver (Exp 3505).

Traces to REQ-KONA-3505, SCENARIO-KONA-3505.

Design philosophy: budgets are tiny so the test suite stays fast. The science
budget (SA hyperparameters, restart counts, etc.) lives in the experiment
driver, not here. We verify:
 - Row-fill initialisation produces valid row permutations.
 - Cached count arrays stay consistent with the actual board.
 - Delta computation is correct (compare against ground-truth recount).
 - Apply-swap is the exact inverse of the delta it approved.
 - sa_solve_once / sa_solve_restarts / parallel_tempering_solve return the
   required shapes and types, and solve trivially-constrained puzzles.
 - compute_violations_from_board agrees with cached n_viol after many swaps.
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import pytest

from carnot.phase3.sudoku_discrete_sa import (
    _apply_swap,
    _delta_viol,
    _init_state,
    _run_sweep,
    compute_violations_from_board,
    parallel_tempering_solve,
    sa_solve_once,
    sa_solve_restarts,
)
from carnot.phase3.sudoku_global_opt import (
    board_is_valid_solution,
    generate_full_grid,
    make_puzzle_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_puzzle() -> list[list[int]]:
    """A trivially almost-solved puzzle: valid board with one cell cleared."""
    full = generate_full_grid(42)
    puzzle = [row[:] for row in full]
    puzzle[8][8] = 0  # blank just the last cell
    return puzzle


def _easy_puzzle() -> list[list[int]]:
    """One easy-tier puzzle from the standard puzzle set."""
    puzzles = make_puzzle_set(seed=3505)
    easy = [p for p in puzzles if p.difficulty == "easy"][0]
    return easy.clues


def _fully_solved_board() -> list[list[int]]:
    return generate_full_grid(7)


# ---------------------------------------------------------------------------
# Row-fill initialisation
# ---------------------------------------------------------------------------

class TestInitState:
    """_init_state must produce a board where each row is a permutation of 1-9."""

    def test_rows_are_permutations(self):
        # REQ-KONA-3505: row uniqueness preserved from the start
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        rng = np.random.default_rng(1)
        state = _init_state(clues_arr, rng)
        board = state["board"]
        full = set(range(1, 10))
        for r in range(9):
            assert set(board[r].tolist()) == full, f"row {r} is not a permutation of 1-9"

    def test_clue_cells_unchanged(self):
        """Given digits must not be modified by row-fill."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        rng = np.random.default_rng(2)
        state = _init_state(clues_arr, rng)
        board = state["board"]
        for r in range(9):
            for c in range(9):
                if clues[r][c] != 0:
                    assert board[r, c] == clues[r][c], f"clue overwritten at ({r},{c})"

    def test_n_viol_non_negative(self):
        """Initial violation count is a non-negative integer."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(3))
        assert isinstance(state["n_viol"], int)
        assert state["n_viol"] >= 0

    def test_swappable_rows_have_ge2_free_cells(self):
        """Every row in swappable has >= 2 non-clue cells."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(4))
        for r in state["swappable"]:
            assert len(state["free_by_row"][r]) >= 2, f"row {r} in swappable but < 2 free cells"

    def test_col_counts_consistent(self):
        """col_counts must agree with counting digits directly in the board."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(5))
        board = state["board"]
        for c in range(9):
            for d in range(9):
                expected = int(np.sum(board[:, c] == d + 1))
                assert state["col_counts"][c, d] == expected, (
                    f"col_counts[{c},{d}] mismatch: got {state['col_counts'][c,d]}, expected {expected}"
                )

    def test_box_counts_consistent(self):
        """box_counts must agree with counting digits directly in each box."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(6))
        board = state["board"]
        for br in range(3):
            for bc in range(3):
                block = board[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3]
                for d in range(9):
                    expected = int(np.sum(block == d + 1))
                    assert state["box_counts"][br, bc, d] == expected

    def test_valid_board_has_zero_violations(self):
        """A fully solved board initialised as clue set has n_viol == 0."""
        full = _fully_solved_board()
        clues_arr = np.array(full, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(7))
        assert state["n_viol"] == 0


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

class TestDeltaViol:
    """_delta_viol must equal the actual change in ground-truth violations."""

    def _ground_truth_delta(self, state, r, c1, c2):
        """Independently compute delta by recounting before/after swap."""
        board = state["board"].copy()
        before = compute_violations_from_board(board.tolist())
        board[r, c1], board[r, c2] = board[r, c2], board[r, c1]
        after = compute_violations_from_board(board.tolist())
        return after - before

    def test_delta_same_as_recount_various_swaps(self):
        """Delta matches ground-truth recount for 20 random swaps."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(10))
        rng = np.random.default_rng(11)

        swappable = state["swappable"]
        for _ in range(20):
            r = swappable[int(rng.integers(len(swappable)))]
            free = state["free_by_row"][r]
            if len(free) < 2:
                continue
            i1 = int(rng.integers(len(free)))
            i2 = int(rng.integers(len(free) - 1))
            if i2 >= i1:
                i2 += 1
            c1, c2 = free[i1], free[i2]

            fast = _delta_viol(state, r, c1, c2)
            gt = self._ground_truth_delta(state, r, c1, c2)
            assert fast == gt, f"delta mismatch at row={r} c1={c1} c2={c2}: fast={fast} gt={gt}"

    def test_delta_zero_for_equal_values(self):
        """Swapping two cells with the same value has delta=0."""
        # Find two free cells in the same row with the same digit.
        full = _fully_solved_board()
        # Place the same digit in two empty cells of the same row by building
        # a custom all-zero puzzle except for that row.
        clues = [[0] * 9 for _ in range(9)]
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(12))
        # Force two cells in row 0 to the same digit.
        board = state["board"]
        c1, c2 = state["free_by_row"][0][0], state["free_by_row"][0][1]
        same_val = int(board[0, c1])
        orig = int(board[0, c2])
        state["col_counts"][c2, orig - 1] -= 1
        state["col_counts"][c2, same_val - 1] += 1
        bc2 = c2 // 3
        state["box_counts"][0, bc2, orig - 1] -= 1
        state["box_counts"][0, bc2, same_val - 1] += 1
        board[0, c2] = same_val
        assert _delta_viol(state, 0, c1, c2) == 0

    def test_delta_same_box_columns_differ(self):
        """Same-box swap: only column deltas apply (box delta must be 0)."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(13))
        # Find two free cells in the same row AND same box.
        for r in state["swappable"]:
            free = state["free_by_row"][r]
            same_box_pairs = [
                (c1, c2)
                for c1 in free
                for c2 in free
                if c1 < c2 and c1 // 3 == c2 // 3
            ]
            if same_box_pairs:
                c1, c2 = same_box_pairs[0]
                fast = _delta_viol(state, r, c1, c2)
                gt = TestDeltaViol()._ground_truth_delta(state, r, c1, c2)
                assert fast == gt, "same-box delta mismatch"
                return
        pytest.skip("No same-box free-cell pair found in easy puzzle")


# ---------------------------------------------------------------------------
# Apply swap + count consistency
# ---------------------------------------------------------------------------

class TestApplySwap:
    """After _apply_swap, cached counts must match a fresh recount."""

    def test_counts_consistent_after_20_swaps(self):
        """Apply 20 random accepted swaps; counts remain accurate."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(20))
        rng = np.random.default_rng(21)

        swappable = state["swappable"]
        for _ in range(20):
            r = swappable[int(rng.integers(len(swappable)))]
            free = state["free_by_row"][r]
            i1 = int(rng.integers(len(free)))
            i2 = int(rng.integers(len(free) - 1))
            if i2 >= i1:
                i2 += 1
            c1, c2 = free[i1], free[i2]
            delta = _delta_viol(state, r, c1, c2)
            _apply_swap(state, r, c1, c2, delta)

        board = state["board"]
        # Verify col_counts
        for c in range(9):
            for d in range(9):
                expected = int(np.sum(board[:, c] == d + 1))
                assert state["col_counts"][c, d] == expected, (
                    f"col_counts[{c},{d}] inconsistent after swaps"
                )
        # Verify box_counts
        for br in range(3):
            for bc in range(3):
                block = board[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3]
                for d in range(9):
                    expected = int(np.sum(block == d + 1))
                    assert state["box_counts"][br, bc, d] == expected

    def test_n_viol_consistent_after_swaps(self):
        """n_viol in state equals ground-truth recount after many swaps."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(22))
        rng = np.random.default_rng(23)

        swappable = state["swappable"]
        for _ in range(30):
            r = swappable[int(rng.integers(len(swappable)))]
            free = state["free_by_row"][r]
            i1 = int(rng.integers(len(free)))
            i2 = int(rng.integers(len(free) - 1))
            if i2 >= i1:
                i2 += 1
            c1, c2 = free[i1], free[i2]
            delta = _delta_viol(state, r, c1, c2)
            _apply_swap(state, r, c1, c2, delta)

        gt_viol = compute_violations_from_board(state["board"].tolist())
        assert state["n_viol"] == gt_viol, (
            f"cached n_viol={state['n_viol']} != ground-truth {gt_viol}"
        )


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def test_sweep_does_not_raise(self):
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(30))
        rng = np.random.default_rng(31)
        # Should not raise for any temperature.
        _run_sweep(state, T=0.5, rng=rng, n_moves=50)

    def test_sweep_at_zero_temperature_only_accepts_improvements(self):
        """At T=0, sweep only accepts moves that decrease violations."""
        clues = _easy_puzzle()
        clues_arr = np.array(clues, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(32))
        v_before = state["n_viol"]
        rng = np.random.default_rng(33)
        for _ in range(10):
            _run_sweep(state, T=0.0, rng=rng, n_moves=100)
        assert state["n_viol"] <= v_before

    def test_sweep_no_swappable_rows(self):
        """If there are no swappable rows (all clues fixed), sweep is a no-op."""
        full = _fully_solved_board()
        clues_arr = np.array(full, dtype=np.int64)
        state = _init_state(clues_arr, np.random.default_rng(34))
        assert state["n_viol"] == 0
        rng = np.random.default_rng(35)
        _run_sweep(state, T=1.0, rng=rng, n_moves=50)
        assert state["n_viol"] == 0


# ---------------------------------------------------------------------------
# High-level solver interfaces
# ---------------------------------------------------------------------------

class TestSaSolveOnce:
    def test_returns_correct_types(self):
        clues = _easy_puzzle()
        board, solved, n_viol = sa_solve_once(clues, n_sweeps=10, n_moves_per_sweep=20, seed=0)
        assert isinstance(board, list) and len(board) == 9
        assert all(len(row) == 9 for row in board)
        assert isinstance(solved, bool)
        assert isinstance(n_viol, int)

    def test_solved_board_passes_validity(self):
        """If sa_solve_once reports solved=True, board must pass validity oracle."""
        clues = _minimal_puzzle()
        board, solved, n_viol = sa_solve_once(
            clues, n_sweeps=2000, n_moves_per_sweep=50, T_init=0.3, T_final=0.01, seed=42
        )
        if solved:
            assert n_viol == 0
            assert board_is_valid_solution(board, clues)

    def test_fully_solved_board_is_immediately_solved(self):
        """A puzzle where all cells are clues (no free cells) is trivially valid."""
        full = _fully_solved_board()
        board, solved, n_viol = sa_solve_once(full, n_sweeps=100, n_moves_per_sweep=10, seed=0)
        assert solved
        assert n_viol == 0

    def test_early_exit_break_fires_for_two_free_cell_puzzle(self):
        """Line 251 (break) fires when SA solves a 2-free-cell puzzle.

        WHY: with only 2 free cells in one row and near-zero temperature,
        SA can only accept the 1 unique improving swap (or no move at all).
        The board reaches n_viol == 0 within a sweep and the break fires.
        """
        full = generate_full_grid(99)
        clues = [row[:] for row in full]
        # Blank 2 adjacent cells in row 5. Row 5 gets exactly 2 free cells →
        # swappable is non-empty (no early return), but SA has a trivial fix.
        clues[5][0] = 0
        clues[5][1] = 0
        # Very low T → no worsening accepted; n_viol → 0 quickly.
        board, solved, n_viol = sa_solve_once(
            clues, n_sweeps=200, n_moves_per_sweep=20, T_init=0.01, T_final=0.001, seed=1
        )
        assert solved
        assert n_viol == 0
        assert board_is_valid_solution(board, clues)


class TestSaSolveRestarts:
    def test_returns_correct_types(self):
        clues = _easy_puzzle()
        board, solved, n_viol = sa_solve_restarts(
            clues, n_sweeps=5, n_moves_per_sweep=10, n_restarts=2, seed=0
        )
        assert isinstance(board, list)
        assert isinstance(solved, bool)
        assert isinstance(n_viol, int)

    def test_fully_solved_returns_solved(self):
        full = _fully_solved_board()
        board, solved, n_viol = sa_solve_restarts(
            full, n_sweeps=5, n_moves_per_sweep=5, n_restarts=1, seed=0
        )
        assert solved
        assert n_viol == 0

    def test_fully_solved_with_callback_hits_no_swappable_branch(self):
        """Fully-solved board with callback covers line 293 (no swappable rows path)."""
        full = _fully_solved_board()
        called = []

        def cb(k, n_viol, solved):
            called.append((k, n_viol, solved))

        board, solved, n_viol = sa_solve_restarts(
            full, n_sweeps=5, n_moves_per_sweep=5, n_restarts=1, seed=0,
            progress_callback=cb
        )
        assert solved
        assert n_viol == 0
        assert len(called) == 1  # callback fired for the single restart

    def test_progress_callback_is_called(self):
        clues = _easy_puzzle()
        called = []

        def cb(k, n_viol, solved):
            called.append((k, n_viol, solved))

        sa_solve_restarts(
            clues, n_sweeps=5, n_moves_per_sweep=10, n_restarts=3, seed=0,
            progress_callback=cb
        )
        # Callback should have been called once per restart (or until solved).
        assert len(called) >= 1

    def test_solved_if_possible_with_enough_budget(self):
        """Lines 303+311 fire when SA solves a 2-free-cell puzzle during restarts."""
        full = generate_full_grid(99)
        clues = [row[:] for row in full]
        clues[5][0] = 0
        clues[5][1] = 0
        board, solved, n_viol = sa_solve_restarts(
            clues, n_sweeps=200, n_moves_per_sweep=20, T_init=0.01, T_final=0.001,
            n_restarts=2, seed=1
        )
        assert solved, "2-free-cell puzzle must be solved at low temperature"
        assert board_is_valid_solution(board, clues)
        assert n_viol == 0


class TestParallelTempering:
    def test_returns_correct_types(self):
        clues = _easy_puzzle()
        board, solved, n_viol = parallel_tempering_solve(
            clues, n_sweeps=10, n_moves_per_sweep=10, n_chains=2,
            T_min=0.1, T_max=1.0, n_exchange_interval=5, seed=0
        )
        assert isinstance(board, list) and len(board) == 9
        assert isinstance(solved, bool)
        assert isinstance(n_viol, int)

    def test_fully_solved_board_immediately_solved(self):
        full = _fully_solved_board()
        board, solved, n_viol = parallel_tempering_solve(
            full, n_sweeps=5, n_moves_per_sweep=5, n_chains=2,
            T_min=0.1, T_max=1.0, n_exchange_interval=5, seed=0
        )
        assert solved
        assert n_viol == 0

    def test_progress_callback_called(self):
        clues = _easy_puzzle()
        calls = []

        def cb(sweep, n_viol):
            calls.append((sweep, n_viol))

        parallel_tempering_solve(
            clues, n_sweeps=20, n_moves_per_sweep=10, n_chains=2,
            T_min=0.1, T_max=1.0, n_exchange_interval=10, seed=0,
            progress_callback=cb
        )
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# compute_violations_from_board
# ---------------------------------------------------------------------------

class TestComputeViolations:
    def test_valid_board_has_zero_violations(self):
        full = _fully_solved_board()
        assert compute_violations_from_board(full) == 0

    def test_board_with_duplicates_has_positive_violations(self):
        full = _fully_solved_board()
        # Corrupt by assigning the row-1 value into row-0, col-0.
        # This creates a duplicate in column 0 (and likely in the top-left box).
        # Note: swapping would NOT create a duplicate (same values, different order).
        val = full[1][0]  # save original value at (1, 0)
        full[0][0] = val  # now full[0][0] == full[1][0] → col 0 has a duplicate
        assert compute_violations_from_board(full) > 0
