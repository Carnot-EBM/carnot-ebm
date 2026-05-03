"""Tests for the Nonogram WOPR cartridge.

Spec traces: REQ-NONOGRAM-001, REQ-NONOGRAM-002,
SCENARIO-NONOGRAM-001, SCENARIO-NONOGRAM-002, SCENARIO-NONOGRAM-003.
"""

from __future__ import annotations

import numpy as np

from carnot.games import NonogramIsingEBM, NonogramPuzzle, NonogramSolver
from carnot.games.nonogram import compute_runs, run_length_mismatch


def _known_5x5_puzzle() -> NonogramPuzzle:
    return NonogramPuzzle.from_clues(
        5,
        row_clues=[[1, 1], [3], [1], [2, 1], [5]],
        col_clues=[[3], [2], [2, 1], [1, 2], [2, 1]],
    )


def test_energy_zero_at_solution() -> None:
    """SCENARIO-NONOGRAM-001: a known 5x5 solution scores E=0.0."""
    puzzle = _known_5x5_puzzle()
    ebm = NonogramIsingEBM(puzzle)

    assert compute_runs(np.array([-1, 1, 1, -1, 1], dtype=np.int8)) == [2, 1]
    assert run_length_mismatch([2, 1], [2, 1]) == 0
    assert ebm.energy(puzzle.solution_spins) == 0.0
    flat_puzzle = NonogramPuzzle.from_clues(
        5,
        row_clues=puzzle.row_clues,
        col_clues=puzzle.col_clues,
        solution_spins=puzzle.solution_spins.reshape(-1),
    )
    assert NonogramIsingEBM(flat_puzzle).energy(flat_puzzle.solution_spins.reshape(-1)) == 0.0


def test_energy_nonzero_at_random() -> None:
    """SCENARIO-NONOGRAM-002: a deterministic random grid has positive energy."""
    puzzle = _known_5x5_puzzle()
    ebm = NonogramIsingEBM(puzzle)
    random_spins = np.random.default_rng(1214).choice([-1, 1], size=(5, 5)).astype(np.int8)

    assert ebm.energy(random_spins) > 0.0


def test_solver_converges() -> None:
    """SCENARIO-NONOGRAM-003: the solver invokes sampling and reduces energy."""
    puzzle = _known_5x5_puzzle()
    ebm = NonogramIsingEBM(puzzle)
    init_spins = np.full((5, 5), -1, dtype=np.int8)
    initial_energy = ebm.energy(init_spins)
    solver = NonogramSolver(puzzle, n_warmup=4, n_samples=3, steps_per_sample=1, seed=1214)

    solved_spins, solved_energy = solver.solve(init_spins=init_spins)

    assert solver.sampler_used
    assert solved_energy <= initial_energy
    assert solved_energy == 0.0
    assert ebm.is_valid_solution(solved_spins)
