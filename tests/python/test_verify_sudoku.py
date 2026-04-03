"""Tests for Sudoku constraint satisfaction — JAX implementation.

Spec coverage: SCENARIO-VERIFY-001, SCENARIO-VERIFY-002, SCENARIO-VERIFY-003
"""

import jax.numpy as jnp

from carnot.verify.constraint import repair
from carnot.verify.sudoku import build_sudoku_energy, grid_to_array

VALID_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


class TestSudokuVerification:
    """Tests for SCENARIO-VERIFY-001."""

    def test_valid_solution_energy_zero(self) -> None:
        """SCENARIO-VERIFY-001: correct solution has total energy = 0."""
        energy_fn = build_sudoku_energy(PUZZLE)
        x = grid_to_array(VALID_SOLUTION)
        result = energy_fn.verify(x)

        assert result.is_verified(), f"Failing: {result.failing_constraints()}"
        assert result.total_energy < 0.1

    def test_invalid_solution_has_violations(self) -> None:
        """SCENARIO-VERIFY-001: incorrect solution has energy > 0."""
        energy_fn = build_sudoku_energy(PUZZLE)

        bad = [row[:] for row in VALID_SOLUTION]
        bad[0][0] = 1  # should be 5 (clue)
        x = grid_to_array(bad)
        result = energy_fn.verify(x)

        assert not result.is_verified()
        assert result.total_energy > 0.1

    def test_per_constraint_decomposition(self) -> None:
        """SCENARIO-VERIFY-003: per-constraint energy sums to total."""
        energy_fn = build_sudoku_energy(PUZZLE)
        x = grid_to_array(VALID_SOLUTION)
        reports = energy_fn.decompose(x)

        # Should have >= 27 constraints (9 rows + 9 cols + 9 boxes + clues)
        assert len(reports) >= 27

        total = float(energy_fn.energy(x))
        decomposed_sum = sum(r.weighted_energy for r in reports)
        assert abs(total - decomposed_sum) < 0.01


class TestSudokuRepair:
    """Tests for SCENARIO-VERIFY-002."""

    def test_repair_reduces_energy(self) -> None:
        """SCENARIO-VERIFY-002: repair reduces violations."""
        energy_fn = build_sudoku_energy(PUZZLE)
        x = jnp.full(81, 5.0)  # all 5s — very wrong

        _, history = repair(energy_fn, x, step_size=0.01, max_steps=100)

        initial_e = history[0].total_energy
        final_e = history[-1].total_energy
        assert final_e < initial_e, f"Energy should decrease: {initial_e} -> {final_e}"
