"""Tests for the Futoshiki WOPR cartridge.

Spec traces: REQ-FUTOSHIKI-001, REQ-FUTOSHIKI-002,
SCENARIO-FUTOSHIKI-001, SCENARIO-FUTOSHIKI-002,
SCENARIO-FUTOSHIKI-003, SCENARIO-FUTOSHIKI-004.
"""

from __future__ import annotations

import numpy as np

from carnot.games.futoshiki import FutoshikiIsingEBM, FutoshikiPuzzle, FutoshikiSolver


def _known_5x5_puzzle() -> FutoshikiPuzzle:
    solution = np.array(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1],
            [3, 4, 5, 1, 2],
            [4, 5, 1, 2, 3],
            [5, 1, 2, 3, 4],
        ],
        dtype=np.int8,
    )
    return FutoshikiPuzzle(
        grid_size=5,
        solution=solution,
        inequalities=[
            (0, 0, 0, 1, "<"),
            (0, 4, 1, 4, ">"),
            (1, 1, 1, 2, "<"),
            (2, 2, 2, 3, ">"),
            (3, 2, 4, 2, "<"),
            (4, 0, 4, 1, ">"),
        ],
    )


def test_energy_zero_at_solution() -> None:
    """SCENARIO-FUTOSHIKI-001: a known 5x5 solution scores E=0.0."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    generated = FutoshikiPuzzle.generate(n=5)

    assert ebm.energy(puzzle.solution) == 0.0
    assert ebm.energy(puzzle.solution.reshape(-1)) == 0.0
    assert ebm.is_valid_solution(puzzle.solution)
    assert generated.grid_size == 5
    assert len(generated.inequalities) == 6
    assert FutoshikiIsingEBM(generated).energy(generated.solution) == 0.0


def test_energy_nonzero_at_random() -> None:
    """SCENARIO-FUTOSHIKI-002: a deterministic random grid has positive energy."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    random_values = np.random.default_rng(1227).integers(1, 6, size=(5, 5), dtype=np.int8)

    assert ebm.energy(random_values) > 0.0


def test_energy_nonzero_at_inequality_violation() -> None:
    """SCENARIO-FUTOSHIKI-003: a Latin-valid inequality violation scores positive."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    violating_values = puzzle.solution.copy()
    violating_values[:, [0, 1]] = violating_values[:, [1, 0]]

    assert sorted(violating_values[0].tolist()) == [1, 2, 3, 4, 5]
    assert sorted(violating_values[:, 0].tolist()) == [1, 2, 3, 4, 5]
    assert ebm.energy(violating_values) > 0.0


def test_solver_reduces_energy() -> None:
    """SCENARIO-FUTOSHIKI-004: the solver invokes sampling and reduces energy."""
    puzzle = _known_5x5_puzzle()
    ebm = FutoshikiIsingEBM(puzzle)
    init_values = np.ones((5, 5), dtype=np.int8)
    initial_energy = ebm.energy(init_values)
    solver = FutoshikiSolver(n_warmup=4, n_samples=3, steps_per_sample=1, seed=1227)

    solved_values = solver.solve(puzzle, max_iter=16, init_values=init_values)
    solved_energy = ebm.energy(solved_values)

    assert solver.sampler_used
    assert solved_values.shape == (5, 5)
    assert solved_energy <= initial_energy
    assert solved_energy == 0.0
