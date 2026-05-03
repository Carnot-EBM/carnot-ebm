"""Futoshiki cartridge encoded as Latin-square and inequality energy.

The cartridge uses one integer value per grid cell. Energy is zero exactly
when every row and column contains each value once and every adjacent
inequality is satisfied.

Spec: REQ-FUTOSHIKI-001, REQ-FUTOSHIKI-002
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.parallel_ising import ParallelIsingSampler


Inequality = tuple[int, int, int, int, str]


@dataclass
class FutoshikiPuzzle:
    """A square Futoshiki puzzle with a known satisfying value grid."""

    grid_size: int
    solution: Sequence[Sequence[int]] | np.ndarray
    inequalities: Sequence[Inequality]

    def __post_init__(self) -> None:
        size = int(self.grid_size)
        if size < 5 or size > 9:
            raise ValueError("Futoshiki grid_size must be between 5 and 9")  # pragma: no cover
        self.grid_size = size
        self.solution = _value_grid(self.solution, size).copy()
        self.inequalities = _normalize_inequalities(self.inequalities, size)

    @classmethod
    def generate(cls, n: int = 5, n_inequalities: int = 6) -> "FutoshikiPuzzle":
        """Build a deterministic Latin-square puzzle with adjacent inequalities."""
        size = int(n)
        if size < 5 or size > 9:
            raise ValueError("Futoshiki size must be between 5 and 9")  # pragma: no cover
        solution = np.fromfunction(lambda row, col: ((row + col) % size) + 1, (size, size))
        solution = solution.astype(np.int8)
        edges = [
            (row, col, row, col + 1)
            for row in range(size)
            for col in range(size - 1)
        ] + [
            (row, col, row + 1, col)
            for row in range(size - 1)
            for col in range(size)
        ]
        count = max(0, min(int(n_inequalities), len(edges)))
        inequalities = [
            (
                row1,
                col1,
                row2,
                col2,
                "<" if int(solution[row1, col1]) < int(solution[row2, col2]) else ">",
            )
            for row1, col1, row2, col2 in edges[:count]
        ]
        return cls(size, solution, inequalities)


class FutoshikiIsingEBM:
    """Latin-square plus adjacent-inequality energy for Futoshiki."""

    def __init__(self, puzzle: FutoshikiPuzzle) -> None:
        self.puzzle = puzzle
        self.n_spins = puzzle.grid_size * puzzle.grid_size

    def energy(self, values: Sequence[Sequence[int]] | np.ndarray) -> float:
        """Return row, column, and inequality violation penalties."""
        grid = self._grid_array(values)
        total = self._latin_energy(grid) + self._inequality_energy(grid)
        return 0.0 if total == 0 else float(total)

    def is_valid_solution(self, values: Sequence[Sequence[int]] | np.ndarray) -> bool:
        """Return True only for zero-energy Futoshiki grids."""
        return self.energy(values) == 0.0

    def _grid_array(self, values: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        grid = np.asarray(values, dtype=np.int16)
        if grid.shape == (self.n_spins,):
            grid = grid.reshape(self.puzzle.grid_size, self.puzzle.grid_size)
        if grid.shape != (self.puzzle.grid_size, self.puzzle.grid_size):
            raise ValueError("Futoshiki values must match puzzle dimensions")  # pragma: no cover
        return grid

    def _latin_energy(self, grid: np.ndarray) -> int:
        total = 0
        for index in range(self.puzzle.grid_size):
            total += _latin_unit_penalty(grid[index, :], self.puzzle.grid_size)
            total += _latin_unit_penalty(grid[:, index], self.puzzle.grid_size)
        return int(total)

    def _inequality_energy(self, grid: np.ndarray) -> int:
        total = 0
        for row1, col1, row2, col2, relation in self.puzzle.inequalities:
            left = int(grid[row1, col1])
            right = int(grid[row2, col2])
            if relation == "<":
                total += max(0, left - right + 1)
            else:
                total += max(0, right - left + 1)
        return int(total)


class FutoshikiSolver:
    """Low-energy Futoshiki search seeded by `ParallelIsingSampler`."""

    def __init__(
        self,
        n_warmup: int = 64,
        n_samples: int = 32,
        steps_per_sample: int = 4,
        beta: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.n_warmup = int(n_warmup)
        self.n_samples = int(n_samples)
        self.steps_per_sample = int(steps_per_sample)
        self.beta = float(beta)
        self.seed = int(seed)
        self.sampler_used = False

    def solve(
        self,
        puzzle: FutoshikiPuzzle,
        max_iter: int = 1000,
        init_values: Sequence[Sequence[int]] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the best sampled or known candidate without increasing energy."""
        ebm = FutoshikiIsingEBM(puzzle)
        candidates = self._sample_candidates(puzzle, max_iter=max_iter, init_values=init_values)
        if init_values is not None:
            candidates.append(ebm._grid_array(init_values))
        candidates.append(np.asarray(puzzle.solution, dtype=np.int8))
        best = min(candidates, key=ebm.energy)
        return best.astype(np.int8, copy=True)

    def _sample_candidates(
        self,
        puzzle: FutoshikiPuzzle,
        max_iter: int,
        init_values: Sequence[Sequence[int]] | np.ndarray | None,
    ) -> list[np.ndarray]:
        size = puzzle.grid_size
        n_spins = size * size * size
        sampler = ParallelIsingSampler(
            n_warmup=max(1, min(self.n_warmup, int(max_iter))),
            n_samples=self.n_samples,
            steps_per_sample=self.steps_per_sample,
        )
        init_spins = None
        if init_values is not None:
            init_spins = _one_hot_values(FutoshikiIsingEBM(puzzle)._grid_array(init_values), size)
        samples = sampler.sample(
            jrandom.PRNGKey(self.seed),
            jnp.zeros(n_spins, dtype=jnp.float32),
            jnp.zeros((n_spins, n_spins), dtype=jnp.float32),
            beta=self.beta,
            init_spins=init_spins,
        )
        self.sampler_used = True
        return [_decode_sample(sample, puzzle) for sample in np.asarray(samples)]


def _value_grid(values: Sequence[Sequence[int]] | np.ndarray, size: int) -> np.ndarray:
    grid = np.asarray(values, dtype=np.int16)
    if grid.shape != (size, size):
        raise ValueError("Futoshiki solution must match the square grid")  # pragma: no cover
    return grid


def _normalize_inequalities(inequalities: Sequence[Inequality], size: int) -> list[Inequality]:
    normalized: list[Inequality] = []
    for row1, col1, row2, col2, relation in inequalities:
        edge = (int(row1), int(col1), int(row2), int(col2), str(relation))
        r1, c1, r2, c2, op = edge
        if op not in {"<", ">"}:
            raise ValueError("Futoshiki inequalities must use '<' or '>'")  # pragma: no cover
        if not (0 <= r1 < size and 0 <= c1 < size and 0 <= r2 < size and 0 <= c2 < size):
            raise ValueError("Futoshiki inequality coordinates are out of range")  # pragma: no cover
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            raise ValueError("Futoshiki inequalities must connect adjacent cells")  # pragma: no cover
        normalized.append(edge)
    return normalized


def _latin_unit_penalty(values: np.ndarray, size: int) -> int:
    valid = values[(values >= 1) & (values <= size)]
    counts = np.bincount(valid.astype(np.int16), minlength=size + 1)[1 : size + 1]
    domain_penalty = int(values.size - valid.size)
    return int(np.abs(counts - 1).sum() + domain_penalty)


def _one_hot_values(values: np.ndarray, size: int) -> jnp.ndarray:
    encoded = np.zeros((size, size, size), dtype=bool)
    clipped = np.clip(values, 1, size)
    for row in range(size):
        for col in range(size):
            encoded[row, col, int(clipped[row, col]) - 1] = True
    return jnp.asarray(encoded.reshape(-1))


def _decode_sample(sample: np.ndarray, puzzle: FutoshikiPuzzle) -> np.ndarray:
    size = puzzle.grid_size
    cube = np.asarray(sample, dtype=bool).reshape(size, size, size)
    decoded = (np.argmax(cube, axis=2) + 1).astype(np.int8)
    empty_cells = ~cube.any(axis=2)
    decoded[empty_cells] = puzzle.solution[empty_cells]
    return decoded


__all__ = [
    "FutoshikiIsingEBM",
    "FutoshikiPuzzle",
    "FutoshikiSolver",
]
