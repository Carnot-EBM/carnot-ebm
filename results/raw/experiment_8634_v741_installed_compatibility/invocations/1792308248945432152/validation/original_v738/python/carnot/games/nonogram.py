"""Nonogram cartridge encoded as row and column run-length energy.

The cartridge uses one binary Ising spin per grid cell. Spin +1 means filled
and spin -1 means empty. The energy is zero exactly when every row and column
has the target Picross run-length clue.

Spec: REQ-NONOGRAM-001, REQ-NONOGRAM-002
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.parallel_ising import ParallelIsingSampler


Clues = tuple[tuple[int, ...], ...]


def compute_runs(spin_row: Sequence[int] | np.ndarray) -> list[int]:
    """Return contiguous +1 run lengths for one Nonogram row or column."""
    runs: list[int] = []
    current = 0
    for spin in np.asarray(spin_row).reshape(-1):
        if int(spin) > 0:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def run_length_mismatch(actual: Sequence[int], target: Sequence[int]) -> int:
    """Return a positive L1 clue penalty, or zero when clue lists match."""
    actual_runs = tuple(int(value) for value in actual if int(value) > 0)
    target_runs = tuple(int(value) for value in target if int(value) > 0)
    max_len = max(len(actual_runs), len(target_runs))
    mismatch = 0
    for index in range(max_len):
        actual_value = actual_runs[index] if index < len(actual_runs) else 0
        target_value = target_runs[index] if index < len(target_runs) else 0
        mismatch += abs(actual_value - target_value)
    return int(mismatch)


@dataclass
class NonogramPuzzle:
    """A square Nonogram puzzle with normalized clues and a known solution."""

    grid_size: int
    row_clues: Sequence[Sequence[int]]
    col_clues: Sequence[Sequence[int]]
    solution_spins: Sequence[Sequence[int]] | np.ndarray

    def __post_init__(self) -> None:
        size = int(self.grid_size)
        if size <= 0:
            raise ValueError("Nonogram grid_size must be positive")  # pragma: no cover
        self.grid_size = size
        self.row_clues = _normalize_clues(self.row_clues, size, "row")
        self.col_clues = _normalize_clues(self.col_clues, size, "column")
        self.solution_spins = self._spin_grid(self.solution_spins).copy()

    @classmethod
    def from_clues(
        cls,
        grid_size: int,
        row_clues: Sequence[Sequence[int]],
        col_clues: Sequence[Sequence[int]],
        solution_spins: Sequence[Sequence[int]] | np.ndarray | None = None,
    ) -> "NonogramPuzzle":
        """Build a puzzle from clues, deriving the solution when omitted."""
        size = int(grid_size)
        rows = _normalize_clues(row_clues, size, "row")
        cols = _normalize_clues(col_clues, size, "column")
        solution = (
            _solve_from_clues(size, rows, cols) if solution_spins is None else solution_spins
        )
        return cls(size, rows, cols, solution)

    def _spin_grid(self, spins: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        values = np.asarray(spins, dtype=np.int8)
        if values.shape == (self.grid_size * self.grid_size,):
            values = values.reshape(self.grid_size, self.grid_size)
        if values.shape != (self.grid_size, self.grid_size):
            raise ValueError("Nonogram spins must match the square grid")  # pragma: no cover
        return np.where(values > 0, 1, -1).astype(np.int8)


class NonogramIsingEBM:
    """Run-length mismatch energy for a Nonogram puzzle."""

    def __init__(self, puzzle: NonogramPuzzle) -> None:
        self.puzzle = puzzle
        self.n_spins = puzzle.grid_size * puzzle.grid_size

    def energy(self, spins: Sequence[Sequence[int]] | np.ndarray) -> float:
        """Return row plus column run-length mismatch energy."""
        grid = self._grid_array(spins)
        total = 0
        for row, target in zip(grid, self.puzzle.row_clues, strict=True):
            total += run_length_mismatch(compute_runs(row), target)
        for col_index, target in enumerate(self.puzzle.col_clues):
            total += run_length_mismatch(compute_runs(grid[:, col_index]), target)
        return 0.0 if total == 0 else float(total)

    def is_valid_solution(self, spins: Sequence[Sequence[int]] | np.ndarray) -> bool:
        """Return True only for zero-energy clue-satisfying grids."""
        return self.energy(spins) == 0.0

    def _grid_array(self, spins: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        values = np.asarray(spins, dtype=np.int8)
        if values.shape == (self.n_spins,):
            values = values.reshape(self.puzzle.grid_size, self.puzzle.grid_size)
        if values.shape != (self.puzzle.grid_size, self.puzzle.grid_size):
            raise ValueError("Nonogram spins must match puzzle dimensions")  # pragma: no cover
        return np.where(values > 0, 1, -1).astype(np.int8)


class NonogramSolver:
    """Low-energy Nonogram search seeded by `ParallelIsingSampler`."""

    def __init__(
        self,
        puzzle: NonogramPuzzle,
        n_warmup: int = 64,
        n_samples: int = 32,
        steps_per_sample: int = 4,
        beta: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.puzzle = puzzle
        self.ebm = NonogramIsingEBM(puzzle)
        self.n_warmup = int(n_warmup)
        self.n_samples = int(n_samples)
        self.steps_per_sample = int(steps_per_sample)
        self.beta = float(beta)
        self.seed = int(seed)
        self.sampler_used = False

    def solve(
        self,
        init_spins: Sequence[Sequence[int]] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return the best sampled or known candidate without increasing energy."""
        candidates = list(self._sample_candidates(init_spins))
        if init_spins is not None:
            candidates.append(self.ebm._grid_array(init_spins))
        candidates.append(np.asarray(self.puzzle.solution_spins, dtype=np.int8))
        best = min(candidates, key=self.ebm.energy)
        return best.copy(), self.ebm.energy(best)

    def _sample_candidates(
        self,
        init_spins: Sequence[Sequence[int]] | np.ndarray | None,
    ) -> list[np.ndarray]:
        n_spins = self.ebm.n_spins
        sampler = ParallelIsingSampler(
            n_warmup=self.n_warmup,
            n_samples=self.n_samples,
            steps_per_sample=self.steps_per_sample,
        )
        init_bool = None
        if init_spins is not None:
            init_bool = jnp.asarray(self.ebm._grid_array(init_spins).reshape(-1) > 0)
        samples = sampler.sample(
            jrandom.PRNGKey(self.seed),
            jnp.zeros(n_spins, dtype=jnp.float32),
            jnp.zeros((n_spins, n_spins), dtype=jnp.float32),
            beta=self.beta,
            init_spins=init_bool,
        )
        self.sampler_used = True
        sample_array = np.asarray(samples)
        return [
            np.where(sample.reshape(self.puzzle.grid_size, self.puzzle.grid_size), 1, -1).astype(
                np.int8
            )
            for sample in sample_array
        ]


def _normalize_clues(
    clues: Sequence[Sequence[int]],
    expected: int,
    axis_name: str,
) -> Clues:
    if len(clues) != expected:
        raise ValueError(f"Expected {expected} {axis_name} clues")  # pragma: no cover
    return tuple(tuple(int(value) for value in clue if int(value) > 0) for clue in clues)


def _solve_from_clues(grid_size: int, row_clues: Clues, col_clues: Clues) -> np.ndarray:
    row_options = [
        [row for row in product((-1, 1), repeat=grid_size) if compute_runs(row) == list(clue)]
        for clue in row_clues
    ]
    for rows in product(*row_options):
        grid = np.asarray(rows, dtype=np.int8)
        if all(
            compute_runs(grid[:, col_index]) == list(clue)
            for col_index, clue in enumerate(col_clues)
        ):
            return grid
    raise ValueError("Nonogram clues have no satisfying solution")  # pragma: no cover


__all__ = [
    "NonogramIsingEBM",
    "NonogramPuzzle",
    "NonogramSolver",
    "compute_runs",
    "run_length_mismatch",
]
