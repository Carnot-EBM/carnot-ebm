"""Sudoku constraint satisfaction -- JAX implementation.

**Researcher summary:**
    Encodes Sudoku rules (row/column/box uniqueness + given clues) as
    differentiable energy terms. Builds a ComposedEnergy with 27 uniqueness
    constraints (9 rows + 9 columns + 9 boxes) plus one clue constraint per
    given digit. Enables solving Sudoku via energy minimization / sampling.

**Detailed explanation for engineers:**
    This module demonstrates Carnot's verifiable reasoning capability on a
    concrete, well-known problem: Sudoku. A 9x9 Sudoku puzzle is represented
    as a flat vector of 81 real numbers (not integers!), and the rules are
    encoded as differentiable energy functions:

    1. **Uniqueness constraints** (27 total): Each row, column, and 3x3 box
       must contain 9 distinct values. The energy for a group of 9 cells is
       the sum of pairwise "collision" penalties: for each pair (i,j), we
       penalize max(0, 1 - |x_i - x_j|)^2. This is zero when all values
       differ by at least 1, and positive when values are close together.

    2. **Clue constraints**: For each pre-filled cell, the energy is
       (x[index] - value)^2. This is zero when the cell equals the clue
       and increases quadratically as it deviates.

    **Why real-valued instead of integer?**
    EBMs work with continuous variables and gradients. By relaxing integers to
    reals, we can use gradient-based optimization (Langevin dynamics, repair)
    to find solutions. The uniqueness energy naturally pushes values apart,
    and rounding the final result gives integer solutions.

    **Why is this interesting?**
    Sudoku is a concrete testbed for verifiable reasoning. The same approach
    (encode constraints as energy, sample/optimize, verify) generalizes to:
    - Type checking (constraints on program types)
    - Logical inference (constraints on truth values)
    - Planning (constraints on action sequences)

Spec: SCENARIO-VERIFY-001, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint, ComposedEnergy


class UniquenessConstraint(BaseConstraint):
    """All 9 values in a group of cells should be distinct.

    **Researcher summary:**
        Pairwise repulsion energy: E = sum_{i<j} max(0, 1 - |x_i - x_j|)^2.
        Zero when all pairwise differences >= 1.

    **Detailed explanation for engineers:**
        For a group of 9 cell indices (e.g., all cells in row 3), this
        constraint checks that no two cells have the same value. The energy
        is:
            E = sum over all pairs (i,j) of max(0, 1 - |x_i - x_j|)^2

        Intuition:
        - If |x_i - x_j| >= 1: the pair contributes 0 (no penalty).
        - If |x_i - x_j| < 1: the pair contributes (1 - |x_i - x_j|)^2,
          which is largest when x_i == x_j (penalty of 1.0).

        The squared penalty makes the energy differentiable everywhere,
        which is needed for gradient-based optimization. The threshold of 1
        means that values just need to differ by at least 1 unit (matching
        the Sudoku requirement that digits 1-9 are all different).

    For example::

        # Check that cells 0-8 (first row) have distinct values
        constraint = UniquenessConstraint("row_0", list(range(9)))
        x = jnp.array([1,2,3,4,5,6,7,8,9] + [0]*72, dtype=jnp.float32)
        assert constraint.is_satisfied(x)  # True — all distinct
    """

    def __init__(self, name: str, indices: list[int]) -> None:
        """Create a uniqueness constraint for a group of cell indices.

        Args:
            name: Human-readable name (e.g., "row_3", "col_7", "box_1_2").
            indices: List of 9 indices into the flat 81-element Sudoku vector.
        """
        self._name = name
        self._indices = indices

    @property
    def name(self) -> str:
        """Human-readable name for this constraint."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Relaxed threshold (0.01) to account for floating-point approximation.

        Since we work with real-valued (not integer) representations, the
        energy may not be exactly zero even for a valid Sudoku. A threshold
        of 0.01 allows for small numerical imprecision.
        """
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute pairwise repulsion energy for this group of 9 cells.

        **Detailed explanation for engineers:**
            Extracts the 9 values from the flat vector using the stored
            indices, then computes the sum of squared violations for all
            C(9,2) = 36 pairs. Uses ``jnp.maximum`` (not Python ``max``)
            to keep the computation differentiable and JAX-traceable.

        Args:
            x: Flat 81-element configuration vector.

        Returns:
            Scalar energy (0 if all 9 values differ by >= 1).
        """
        # Extract the 9 values from the flat vector using our stored indices
        vals = x[jnp.array(self._indices)]
        total = jnp.float32(0.0)
        # Check all C(9,2) = 36 pairs
        for i in range(9):
            for j in range(i + 1, 9):
                diff = jnp.abs(vals[i] - vals[j])
                # If diff >= 1, violation = 0. If diff < 1, violation > 0.
                violation = jnp.maximum(1.0 - diff, 0.0)
                # Squared penalty for smoothness (differentiable at boundary)
                total = total + violation**2
        return total


class ClueConstraint(BaseConstraint):
    """A specific cell must equal a given clue value.

    **Researcher summary:**
        Quadratic penalty: E = (x[index] - value)^2. Zero at target value.

    **Detailed explanation for engineers:**
        For a pre-filled Sudoku cell, this constraint penalizes any deviation
        from the given value. The energy is simply the squared difference
        between the cell's current value and the target. This creates a
        "potential well" centered at the clue value, with gradient pointing
        toward it from any direction.

    For example::

        # Cell at row 0, col 2 must be 7
        constraint = ClueConstraint("clue_r0c2", index=2, value=7.0)
        x = jnp.zeros(81)
        print(constraint.energy(x))  # 49.0 (= (0-7)^2)
    """

    def __init__(self, name: str, index: int, value: float) -> None:
        """Create a clue constraint for a single cell.

        Args:
            name: Human-readable name (e.g., "clue_r2c5").
            index: Index into the flat 81-element vector (row*9 + col).
            value: The target value this cell must equal (1.0 through 9.0).
        """
        self._name = name
        self._index = index
        self._value = value

    @property
    def name(self) -> str:
        """Human-readable name for this constraint."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Relaxed threshold (0.01) for floating-point tolerance."""
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute squared deviation from the clue value.

        Args:
            x: Flat 81-element configuration vector.

        Returns:
            Scalar energy: (x[index] - value)^2.
        """
        return (x[self._index] - self._value) ** 2


def build_sudoku_energy(clues: list[list[int]]) -> ComposedEnergy:
    """Build a complete Sudoku energy function from a puzzle grid.

    **Researcher summary:**
        Constructs 27 uniqueness constraints (rows + cols + boxes) + N clue
        constraints. Clues weighted 10x to prioritize preserving given digits.

    **Detailed explanation for engineers:**
        Given a 9x9 grid where 0 means empty and 1-9 means a given clue,
        this function builds a ComposedEnergy with:

        - 9 row uniqueness constraints (weight 1.0)
        - 9 column uniqueness constraints (weight 1.0)
        - 9 box (3x3 block) uniqueness constraints (weight 1.0)
        - One clue constraint per non-zero cell (weight 10.0)

        The higher weight on clues (10x) ensures the optimizer/sampler
        strongly prioritizes keeping the given digits at their correct
        values, rather than moving them to satisfy uniqueness.

        The resulting ComposedEnergy can be used with any sampler (Langevin,
        HMC) or with the ``repair()`` function to find valid solutions.

    Args:
        clues: A 9x9 list of lists. 0 = empty cell, 1-9 = given digit.

    Returns:
        A ComposedEnergy with input_dim=81 containing all Sudoku constraints.

    For example::

        puzzle = [
            [5,3,0, 0,7,0, 0,0,0],
            [6,0,0, 1,9,5, 0,0,0],
            [0,9,8, 0,0,0, 0,6,0],
            [8,0,0, 0,6,0, 0,0,3],
            [4,0,0, 8,0,3, 0,0,1],
            [7,0,0, 0,2,0, 0,0,6],
            [0,6,0, 0,0,0, 2,8,0],
            [0,0,0, 4,1,9, 0,0,5],
            [0,0,0, 0,8,0, 0,7,9],
        ]
        energy = build_sudoku_energy(puzzle)
        print(energy.num_constraints)  # 27 + (number of clues)

    Spec: SCENARIO-VERIFY-001
    """
    composed = ComposedEnergy(input_dim=81)

    # --- Row constraints: each row has 9 cells that must be distinct ---
    for row in range(9):
        # Indices for row `row`: positions row*9+0, row*9+1, ..., row*9+8
        indices = [row * 9 + col for col in range(9)]
        composed.add_constraint(UniquenessConstraint(f"row_{row}", indices), 1.0)

    # --- Column constraints: each column has 9 cells that must be distinct ---
    for col in range(9):
        # Indices for column `col`: positions 0*9+col, 1*9+col, ..., 8*9+col
        indices = [row * 9 + col for row in range(9)]
        composed.add_constraint(UniquenessConstraint(f"col_{col}", indices), 1.0)

    # --- Box constraints: each 3x3 block has 9 cells that must be distinct ---
    for box_row in range(3):
        for box_col in range(3):
            indices = []
            for r in range(3):
                for c in range(3):
                    # Map (box_row, box_col, r, c) to flat index
                    indices.append((box_row * 3 + r) * 9 + (box_col * 3 + c))
            composed.add_constraint(
                UniquenessConstraint(f"box_{box_row}_{box_col}", indices), 1.0
            )

    # --- Clue constraints: given digits must not change ---
    for row in range(9):
        for col in range(9):
            if clues[row][col] != 0:
                # Weight 10.0 — much stronger than uniqueness constraints,
                # so the optimizer will preserve clues at the cost of
                # temporarily violating uniqueness rather than the reverse.
                composed.add_constraint(
                    ClueConstraint(f"clue_r{row}c{col}", row * 9 + col, float(clues[row][col])),
                    10.0,
                )

    return composed


def grid_to_array(grid: list[list[int]]) -> jax.Array:
    """Convert a 9x9 integer grid to a flat 81-element JAX array.

    **Detailed explanation for engineers:**
        Sudoku puzzles are naturally represented as 9x9 grids, but the
        energy functions operate on flat 1-D vectors of length 81. This
        utility function handles the conversion, casting integers to
        float32 (required for JAX gradient computation).

    Args:
        grid: A 9x9 list of lists of integers (0-9).

    Returns:
        A 1-D JAX array of shape (81,) with float32 dtype.

    For example::

        grid = [[5,3,0,...], [6,0,0,...], ...]
        x = grid_to_array(grid)
        assert x.shape == (81,)
    """
    flat = []
    for row in grid:
        flat.extend(float(v) for v in row)
    return jnp.array(flat)
