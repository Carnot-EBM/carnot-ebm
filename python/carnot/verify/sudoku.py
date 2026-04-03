"""Sudoku constraint satisfaction — JAX implementation.

Spec: SCENARIO-VERIFY-001, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint, ComposedEnergy


class UniquenessConstraint(BaseConstraint):
    """All values in a group of 9 cells should be distinct.

    Energy = sum over pairs of max(0, 1 - |xi - xj|)^2
    """

    def __init__(self, name: str, indices: list[int]) -> None:
        self._name = name
        self._indices = indices

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        vals = x[jnp.array(self._indices)]
        total = jnp.float32(0.0)
        for i in range(9):
            for j in range(i + 1, 9):
                diff = jnp.abs(vals[i] - vals[j])
                violation = jnp.maximum(1.0 - diff, 0.0)
                total = total + violation**2
        return total


class ClueConstraint(BaseConstraint):
    """A cell should equal a given clue value.

    Energy = (x[index] - value)^2
    """

    def __init__(self, name: str, index: int, value: float) -> None:
        self._name = name
        self._index = index
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        return (x[self._index] - self._value) ** 2


def build_sudoku_energy(clues: list[list[int]]) -> ComposedEnergy:
    """Build a Sudoku energy function from a puzzle.

    clues: 9x9 grid, 0=empty, 1-9=given.

    Spec: SCENARIO-VERIFY-001
    """
    composed = ComposedEnergy(input_dim=81)

    # Row constraints
    for row in range(9):
        indices = [row * 9 + col for col in range(9)]
        composed.add_constraint(UniquenessConstraint(f"row_{row}", indices), 1.0)

    # Column constraints
    for col in range(9):
        indices = [row * 9 + col for row in range(9)]
        composed.add_constraint(UniquenessConstraint(f"col_{col}", indices), 1.0)

    # Box constraints
    for box_row in range(3):
        for box_col in range(3):
            indices = []
            for r in range(3):
                for c in range(3):
                    indices.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
            composed.add_constraint(
                UniquenessConstraint(f"box_{box_row}_{box_col}", indices), 1.0
            )

    # Clue constraints
    for row in range(9):
        for col in range(9):
            if clues[row][col] != 0:
                composed.add_constraint(
                    ClueConstraint(f"clue_r{row}c{col}", row * 9 + col, float(clues[row][col])),
                    10.0,
                )

    return composed


def grid_to_array(grid: list[list[int]]) -> jax.Array:
    """Convert 9x9 grid to flat 81-element JAX array."""
    flat = []
    for row in grid:
        flat.extend(float(v) for v in row)
    return jnp.array(flat)
