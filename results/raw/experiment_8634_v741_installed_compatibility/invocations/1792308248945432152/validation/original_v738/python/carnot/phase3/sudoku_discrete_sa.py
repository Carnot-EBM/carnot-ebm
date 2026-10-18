"""Discrete simulated annealing solver for Sudoku P0.1 (Exp 3505).

WHY this module exists:
    Exp 3494's continuous-relaxation Langevin optimizer (JAX) produced
    easy_tier_solve_rate=0.0. The energy encoding was PROVEN CORRECT (E=0 on a
    valid board), so the OPTIMIZER is the bottleneck, not the representation.

    Gradient descent on the continuous relaxation gets stuck in local minima
    where adjacent cells sit ~0.5 apart and pay a small residual penalty — but
    the rounded board still violates constraints. Discrete SA sidesteps this by
    operating on the INTEGER board directly using row-swap moves:

    - Representation: each row is a permutation of 1-9 (row uniqueness preserved
      by construction at every step, so we only need to minimize column + box
      violations).
    - Move: swap two non-clue cells in the same row. Row uniqueness is preserved
      automatically because we only move values within the same row.
    - Energy: count of "excess" digit occurrences across all 9 columns + 9 boxes.
      Zero iff the board is a valid Sudoku (together with row uniqueness).
    - Delta computation: O(1) per move using cached count arrays, so the inner
      loop is fast Python without numpy overhead.
    - Literature: Simonis (2005) showed row-swap SA solves hard Sudoku in < 100K
      moves on average. Our easy puzzles (46 clues) solve in < 5K moves.

Spec: REQ-KONA-3505, SCENARIO-KONA-3505
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# A 9x9 grid as a Python list of lists of ints (0 = empty, 1-9 = digit).
Grid = list[list[int]]


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _init_state(clues_arr: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Build the mutable SA state from a 9x9 clue array.

    WHY: We need three data structures maintained in sync:
    - board: the actual 9x9 integer board (mutated in-place by swaps)
    - col_counts / box_counts: cached digit-count arrays so delta_E is O(1)
    - free_by_row: pre-computed free (non-clue) cell indices per row
    """
    board = clues_arr.copy().astype(np.int64)
    clue_mask = board > 0

    # Fill each row with a random permutation of the missing digits so that
    # every row is a permutation of 1-9 from the start (zero row violations).
    for r in range(9):
        fixed = set(int(v) for v in board[r, clue_mask[r]])
        remaining = [d for d in range(1, 10) if d not in fixed]
        rng.shuffle(remaining)
        empty_cols = np.where(~clue_mask[r])[0]
        for i, c in enumerate(empty_cols):
            board[r, c] = remaining[i]

    # col_counts[c, d] = number of times digit (d+1) appears in column c.
    col_counts = np.zeros((9, 9), dtype=np.int64)
    for c in range(9):
        for d in range(9):
            col_counts[c, d] = int(np.sum(board[:, c] == d + 1))

    # box_counts[br, bc, d] = count of digit (d+1) in 3x3 box (br, bc).
    box_counts = np.zeros((3, 3, 9), dtype=np.int64)
    for br in range(3):
        for bc in range(3):
            block = board[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3]
            for d in range(9):
                box_counts[br, bc, d] = int(np.sum(block == d + 1))

    # Total violations = sum of (count - 1) over all (col/box, digit) pairs
    # where count > 1. This counts one excess per extra copy in each group.
    n_viol = int(np.sum(np.maximum(0, col_counts - 1))) + int(
        np.sum(np.maximum(0, box_counts - 1))
    )

    free_by_row = [list(np.where(~clue_mask[r])[0]) for r in range(9)]
    swappable = [r for r in range(9) if len(free_by_row[r]) >= 2]

    return {
        "board": board,
        "clue_mask": clue_mask,
        "col_counts": col_counts,
        "box_counts": box_counts,
        "n_viol": n_viol,
        "free_by_row": free_by_row,
        "swappable": swappable,
    }


# ---------------------------------------------------------------------------
# O(1) delta and apply
# ---------------------------------------------------------------------------

def _delta_viol(state: dict[str, Any], r: int, c1: int, c2: int) -> int:
    """Return the change in total violations from swapping board[r,c1]<->board[r,c2].

    WHY O(1): We maintain cached col_counts and box_counts. Leaving digit v1
    from a column/box reduces violations by 1 iff v1 appeared more than once
    there. Entering digit v2 increases violations by 1 iff v2 already appeared
    at least once. This gives four O(1) integer comparisons per affected group
    (2 columns, up to 2 boxes).

    WHY same-box => zero box delta: both cells stay inside the same box, so the
    set of digit counts in that box is unchanged. Only column violations change.
    """
    board = state["board"]
    col_counts = state["col_counts"]
    box_counts = state["box_counts"]

    v1 = int(board[r, c1]) - 1  # 0-indexed
    v2 = int(board[r, c2]) - 1
    if v1 == v2:
        return 0

    # Column c1: v1 leaves, v2 enters.
    dc1 = (1 if col_counts[c1, v2] >= 1 else 0) - (1 if col_counts[c1, v1] > 1 else 0)
    # Column c2: v2 leaves, v1 enters.
    dc2 = (1 if col_counts[c2, v1] >= 1 else 0) - (1 if col_counts[c2, v2] > 1 else 0)

    br = r // 3
    bc1 = c1 // 3
    bc2 = c2 // 3

    if bc1 == bc2:
        # Both cells in same box → digit counts in that box are unchanged.
        db = 0
    else:
        # Box bc1: v1 leaves, v2 enters.
        db1 = (1 if box_counts[br, bc1, v2] >= 1 else 0) - (
            1 if box_counts[br, bc1, v1] > 1 else 0
        )
        # Box bc2: v2 leaves, v1 enters.
        db2 = (1 if box_counts[br, bc2, v1] >= 1 else 0) - (
            1 if box_counts[br, bc2, v2] > 1 else 0
        )
        db = db1 + db2

    return dc1 + dc2 + db


def _apply_swap(state: dict[str, Any], r: int, c1: int, c2: int, delta: int) -> None:
    """Apply the swap board[r,c1]<->board[r,c2] and update all cached counts."""
    board = state["board"]
    col_counts = state["col_counts"]
    box_counts = state["box_counts"]

    v1 = int(board[r, c1]) - 1
    v2 = int(board[r, c2]) - 1

    board[r, c1], board[r, c2] = v2 + 1, v1 + 1

    # Update column counts.
    col_counts[c1, v1] -= 1
    col_counts[c1, v2] += 1
    col_counts[c2, v2] -= 1
    col_counts[c2, v1] += 1

    # Update box counts only when the two cells are in different boxes.
    # WHY: if bc1 == bc2 (same box), the swap moves both digits within the box —
    # net count per digit is unchanged, so no update needed.
    br = r // 3
    bc1 = c1 // 3
    bc2 = c2 // 3
    if bc1 != bc2:
        box_counts[br, bc1, v1] -= 1
        box_counts[br, bc1, v2] += 1
        box_counts[br, bc2, v2] -= 1
        box_counts[br, bc2, v1] += 1

    state["n_viol"] += delta


# ---------------------------------------------------------------------------
# SA inner loop
# ---------------------------------------------------------------------------

def _run_sweep(state: dict[str, Any], T: float, rng: np.random.Generator, n_moves: int) -> None:
    """Run n_moves random row-swap attempts at temperature T.

    WHY n_moves per call rather than "all free cells once": this decouples the
    number of attempts from puzzle difficulty and makes the cooling schedule
    uniform across puzzle tiers.
    """
    swappable = state["swappable"]
    free_by_row = state["free_by_row"]
    if not swappable:
        return

    rows = rng.integers(0, len(swappable), size=n_moves)
    for ri in rows:
        r = swappable[ri]
        free = free_by_row[r]
        nf = len(free)
        i1 = int(rng.integers(nf))
        i2 = int(rng.integers(nf - 1))
        if i2 >= i1:
            i2 += 1
        c1, c2 = free[i1], free[i2]

        delta = _delta_viol(state, r, c1, c2)
        if delta <= 0:
            _apply_swap(state, r, c1, c2, delta)
        elif T > 0.0:
            if rng.random() < math.exp(-delta / T):
                _apply_swap(state, r, c1, c2, delta)


# ---------------------------------------------------------------------------
# Public solver interfaces
# ---------------------------------------------------------------------------

def sa_solve_once(
    clues: Grid,
    *,
    n_sweeps: int = 5000,
    n_moves_per_sweep: int = 100,
    T_init: float = 0.5,
    T_final: float = 0.01,
    seed: int = 0,
) -> tuple[Grid, bool, int]:
    """Run a single SA trajectory on the given Sudoku puzzle.

    Returns (final_board, solved, n_violations_at_stop).
    solved=True iff the board satisfies all Sudoku constraints.
    Early-exits as soon as n_viol == 0.
    """
    clues_arr = np.array(clues, dtype=np.int64)
    rng = np.random.default_rng(seed)
    state = _init_state(clues_arr, rng)

    if not state["swappable"]:
        board = state["board"].tolist()
        from carnot.phase3.sudoku_global_opt import board_is_valid_solution
        return board, board_is_valid_solution(board, clues), state["n_viol"]

    T = T_init
    cooling = (T_final / T_init) ** (1.0 / max(n_sweeps, 1))

    for _ in range(n_sweeps):
        _run_sweep(state, T, rng, n_moves_per_sweep)
        T *= cooling
        if state["n_viol"] == 0:
            break

    board = state["board"].tolist()
    from carnot.phase3.sudoku_global_opt import board_is_valid_solution
    solved = board_is_valid_solution(board, clues)
    return board, solved, state["n_viol"]


def sa_solve_restarts(
    clues: Grid,
    *,
    n_sweeps: int = 5000,
    n_moves_per_sweep: int = 100,
    T_init: float = 0.5,
    T_final: float = 0.01,
    n_restarts: int = 20,
    seed: int = 0,
    progress_callback: Any = None,
) -> tuple[Grid, bool, int]:
    """Run K independent SA restarts; return the best (solved or fewest violations).

    WHY K restarts: each restart is independent from a fresh random row fill.
    Early stopping across restarts (first solved result wins). If no restart
    solves, returns the restart with fewest violations for plateau analysis.

    progress_callback(restart_idx, n_viol, solved) called after each restart
    for live progress printing.
    """
    from carnot.phase3.sudoku_global_opt import board_is_valid_solution

    clues_arr = np.array(clues, dtype=np.int64)
    best_board: Grid | None = None
    best_viol = 10_000

    for k in range(n_restarts):
        rng = np.random.default_rng(seed + k * 997)
        state = _init_state(clues_arr, rng)

        if not state["swappable"]:
            board = state["board"].tolist()
            solved = board_is_valid_solution(board, clues)
            if progress_callback:
                progress_callback(k, state["n_viol"], solved)
            return board, solved, state["n_viol"]

        T = T_init
        cooling = (T_final / T_init) ** (1.0 / max(n_sweeps, 1))

        for _ in range(n_sweeps):
            _run_sweep(state, T, rng, n_moves_per_sweep)
            T *= cooling
            if state["n_viol"] == 0:
                break

        board = state["board"].tolist()
        solved = board_is_valid_solution(board, clues)
        if progress_callback:
            progress_callback(k, state["n_viol"], solved)

        if solved:
            return board, True, 0

        if state["n_viol"] < best_viol:
            best_viol = state["n_viol"]
            best_board = board

    assert best_board is not None
    return best_board, False, best_viol


def parallel_tempering_solve(
    clues: Grid,
    *,
    n_sweeps: int = 5000,
    n_moves_per_sweep: int = 100,
    n_chains: int = 4,
    T_min: float = 0.05,
    T_max: float = 1.0,
    n_exchange_interval: int = 50,
    seed: int = 0,
    progress_callback: Any = None,
) -> tuple[Grid, bool, int]:
    """Parallel tempering: N chains at different temperatures with replica exchange.

    WHY parallel tempering: hot chains explore broadly, cold chains refine.
    Replica exchange (swap adjacent chains' states based on Metropolis criterion)
    lets the cold chain escape basins found by the hot chains. This is a classical
    improvement over independent restarts for multimodal energy landscapes.

    progress_callback(sweep_idx, n_viol_coldest) called every n_exchange_interval.
    """
    from carnot.phase3.sudoku_global_opt import board_is_valid_solution

    clues_arr = np.array(clues, dtype=np.int64)

    # Logarithmically-spaced temperatures from cold to hot.
    temps = np.exp(np.linspace(math.log(T_min), math.log(T_max), n_chains)).tolist()

    rngs = [np.random.default_rng(seed + i * 1337) for i in range(n_chains)]
    states = [_init_state(clues_arr, rngs[i]) for i in range(n_chains)]

    for sweep in range(n_sweeps):
        # Advance each chain one sweep.
        for i in range(n_chains):
            _run_sweep(states[i], temps[i], rngs[i], n_moves_per_sweep)

        # Replica exchange: attempt to swap adjacent (i, i+1) chains.
        if (sweep + 1) % n_exchange_interval == 0:
            exchange_rng = rngs[0]  # use coldest chain's rng for exchange decisions
            for i in range(n_chains - 1):
                E_i = states[i]["n_viol"]
                E_ip1 = states[i + 1]["n_viol"]
                beta_i = 1.0 / temps[i] if temps[i] > 0 else 1e9
                beta_ip1 = 1.0 / temps[i + 1] if temps[i + 1] > 0 else 1e9
                delta_swap = (beta_i - beta_ip1) * (E_ip1 - E_i)
                if delta_swap >= 0 or exchange_rng.random() < math.exp(delta_swap):
                    states[i], states[i + 1] = states[i + 1], states[i]

            if progress_callback:
                progress_callback(sweep + 1, states[0]["n_viol"])

        if states[0]["n_viol"] == 0:
            break

    board = states[0]["board"].tolist()
    solved = board_is_valid_solution(board, clues)
    return board, solved, states[0]["n_viol"]


def parallel_tempering_solve_instrumented(
    clues: Grid,
    *,
    n_sweeps: int = 5000,
    n_moves_per_sweep: int = 100,
    n_chains: int = 6,
    T_min: float = 0.1,
    T_max: float = 2.0,
    n_exchange_interval: int = 50,
    seed: int = 0,
    progress_callback: Any = None,
) -> tuple[list, bool, int, float]:
    """Parallel tempering with swap-acceptance tracking and tunable ladder.

    WHY this exists: exp3505 found parallel_tempering_solve_rate=0.38, BELOW
    discrete_sa_restarts20=1.0. Root cause: n_exchange_interval=n_sweeps//5+1
    gave only ~4 total exchange attempts per puzzle — practically no chain mixing.
    This version makes exchange_interval configurable (default 50, yielding many
    exchanges) and tracks swap_acceptance_rate so the ladder quality is auditable.

    Returns (board, solved, n_viol, swap_acceptance_rate).
    swap_acceptance_rate is the fraction of proposed adjacent-chain swaps that were
    accepted; target 0.2-0.5 for effective mixing.
    """
    from carnot.phase3.sudoku_global_opt import board_is_valid_solution

    clues_arr = np.array(clues, dtype=np.int64)

    # Logarithmically-spaced temperatures from cold to hot.
    temps = np.exp(np.linspace(math.log(T_min), math.log(T_max), n_chains)).tolist()

    rngs = [np.random.default_rng(seed + i * 1337) for i in range(n_chains)]
    states = [_init_state(clues_arr, rngs[i]) for i in range(n_chains)]

    total_proposals = 0
    total_accepts = 0

    for sweep in range(n_sweeps):
        for i in range(n_chains):
            _run_sweep(states[i], temps[i], rngs[i], n_moves_per_sweep)

        if (sweep + 1) % n_exchange_interval == 0:
            exchange_rng = rngs[0]
            for i in range(n_chains - 1):
                total_proposals += 1
                E_i = states[i]["n_viol"]
                E_ip1 = states[i + 1]["n_viol"]
                beta_i = 1.0 / temps[i] if temps[i] > 0 else 1e9
                beta_ip1 = 1.0 / temps[i + 1] if temps[i + 1] > 0 else 1e9
                delta_swap = (beta_i - beta_ip1) * (E_ip1 - E_i)
                if delta_swap >= 0 or exchange_rng.random() < math.exp(delta_swap):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    total_accepts += 1

            if progress_callback:
                progress_callback(sweep + 1, states[0]["n_viol"])

        if states[0]["n_viol"] == 0:
            break

    board = states[0]["board"].tolist()
    solved = board_is_valid_solution(board, clues)
    swap_acceptance = total_accepts / total_proposals if total_proposals > 0 else 0.0
    return board, solved, states[0]["n_viol"], swap_acceptance


def compute_violations_from_board(board: Grid) -> int:
    """Count total column + box violations in a 9x9 board (all rows assumed unique).

    WHY: Used in tests and diagnostics to independently verify the cached count
    arrays stay consistent with the actual board after many swaps.
    """
    arr = np.array(board, dtype=np.int64)
    viol = 0
    for c in range(9):
        for d in range(9):
            cnt = int(np.sum(arr[:, c] == d + 1))
            if cnt > 1:
                viol += cnt - 1
    for br in range(3):
        for bc in range(3):
            block = arr[br * 3:(br + 1) * 3, bc * 3:(bc + 1) * 3].flatten()
            for d in range(9):
                cnt = int(np.sum(block == d + 1))
                if cnt > 1:
                    viol += cnt - 1
    return viol
