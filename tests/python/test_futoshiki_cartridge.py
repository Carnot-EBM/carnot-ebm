"""Tests for the Futoshiki WOPR cartridge.

Spec traces: REQ-FUTOSHIKI-001, REQ-FUTOSHIKI-002,
SCENARIO-FUTOSHIKI-001, SCENARIO-FUTOSHIKI-002,
SCENARIO-FUTOSHIKI-003, SCENARIO-FUTOSHIKI-004.
"""

from __future__ import annotations

import numpy as np

from carnot.games import FutoshikiIsingEBM, FutoshikiPuzzle, FutoshikiSolver


def _known_5x5_puzzle() -> FutoshikiPuzzle:
    return FutoshikiPuzzle.generate(n=5, n_inequalities=6)


def test_energy_zero_at_solution() -> None:
    """SCENARIO-FUTOSHIKI-001: a known 5x5 solution scores E=0.0."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)

    assert puzzle.grid_size == 5
    assert puzzle.solution.shape == (5, 5)
    assert len(puzzle.inequalities) == 6
    assert ebm.energy(puzzle.solution) == 0.0
    assert ebm.is_valid_solution(puzzle.solution)


def test_energy_nonzero_at_random() -> None:
    """SCENARIO-FUTOSHIKI-002: a deterministic random grid has positive energy."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    random_values = np.random.default_rng(1227).integers(1, 6, size=(5, 5), dtype=np.int8)

    assert ebm.energy(random_values) > 0.0


def test_energy_nonzero_at_inequality_violation() -> None:
    """SCENARIO-FUTOSHIKI-003: a Latin-valid inequality violation has E>0."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    latin_valid_but_reversed = (puzzle.grid_size + 1 - puzzle.solution).astype(np.int8)
    expected_values = list(range(1, puzzle.grid_size + 1))

    for row in latin_valid_but_reversed:
        assert sorted(int(value) for value in row) == expected_values
    for col_index in range(puzzle.grid_size):
        assert sorted(int(value) for value in latin_valid_but_reversed[:, col_index]) == (
            expected_values
        )
    assert ebm.energy(latin_valid_but_reversed) > 0.0


def test_solver_reduces_energy() -> None:
    """SCENARIO-FUTOSHIKI-004: the solver invokes sampling and reduces energy."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    init_values = np.ones((5, 5), dtype=np.int8)
    initial_energy = ebm.energy(init_values)
    solver = FutoshikiSolver(n_warmup=2, n_samples=2, steps_per_sample=1, seed=1227)

    solved_values = solver.solve(puzzle, max_iter=2)
    solved_energy = ebm.energy(solved_values)

    assert solver.sampler_used
    assert solved_values.shape == (5, 5)
    assert solved_energy <= initial_energy
    assert solved_energy == 0.0
