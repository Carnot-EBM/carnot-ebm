"""N-Queens cartridge -- 8x8 CSP as a 64-spin Ising model.

Each square is a binary spin s_(row,col): 1 means a queen is placed there.
The coupling matrix is antiferromagnetic for every attacking pair:
same row, same column, or same diagonal. With one queen per row, zero Ising
energy means no two queens attack each other.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence  # noqa: TC003 — used at runtime, not just type-check
from dataclasses import dataclass, field

import numpy as np

from games._base import StepResult, WOPRGame

BOARD_SIZE = 8
N_SPINS = BOARD_SIZE * BOARD_SIZE
DEFAULT_PENALTY = 1.0
ANIMATION_INTERVAL = 1000

THREAT_ASSESSMENT = "THREAT ASSESSMENT: N-QUEENS CONFIGURATION — COMPUTING OPTIMAL PLACEMENT..."
OPTIMAL_MESSAGE = "OPTIMAL CONFIGURATION ACHIEVED. ENERGY = 0. NO TWO PIECES THREATEN EACH OTHER."


try:
    import jax.numpy as jnp
    import jax.random as jrandom

    _carnot_python = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
    )
    if _carnot_python not in sys.path:
        sys.path.insert(0, _carnot_python)

    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    _ISING_AVAILABLE = True
except Exception:
    _ISING_AVAILABLE = False


def _idx(row: int, col: int, n: int = BOARD_SIZE) -> int:
    return row * n + col


def build_nqueens_coupling_matrix(
    n: int = BOARD_SIZE, penalty: float = DEFAULT_PENALTY
) -> np.ndarray:
    """Build the N-Queens Ising coupling matrix.

    J[a,b] = -penalty when squares a and b cannot both contain queens.
    The matrix is symmetric with a zero diagonal. Energy is computed over
    the upper triangle, so each attacking queen pair contributes +penalty.
    """
    J = np.zeros((n * n, n * n), dtype=np.float32)  # noqa: N806 — coupling matrix (Ising convention)

    for row in range(n):
        for col_a in range(n):
            for col_b in range(col_a + 1, n):
                a = _idx(row, col_a, n)
                b = _idx(row, col_b, n)
                J[a, b] = J[b, a] = -penalty

    for col in range(n):
        for row_a in range(n):
            for row_b in range(row_a + 1, n):
                a = _idx(row_a, col, n)
                b = _idx(row_b, col, n)
                J[a, b] = J[b, a] = -penalty

    for row_a in range(n):
        for col_a in range(n):
            for row_b in range(row_a + 1, n):
                delta = row_b - row_a
                for col_b in (col_a - delta, col_a + delta):
                    if 0 <= col_b < n:
                        a = _idx(row_a, col_a, n)
                        b = _idx(row_b, col_b, n)
                        J[a, b] = J[b, a] = -penalty

    np.fill_diagonal(J, 0.0)
    return J


_COUPLING_MATRIX = build_nqueens_coupling_matrix()


def columns_to_spins(columns: Sequence[int], n: int = BOARD_SIZE) -> np.ndarray:
    """Convert row-indexed queen columns into a flat 0/1 spin vector."""
    spins = np.zeros(n * n, dtype=np.int8)
    for row, col in enumerate(columns):
        if 0 <= col < n:
            spins[_idx(row, int(col), n)] = 1
    return spins


def spins_to_columns(spins: Sequence[int] | np.ndarray, n: int = BOARD_SIZE) -> list[int]:
    """Return the first queen column in each row, or -1 when a row is empty."""
    grid = np.asarray(spins, dtype=np.int8).reshape((n, n))
    columns: list[int] = []
    for row in range(n):
        occupied = np.flatnonzero(grid[row])
        columns.append(int(occupied[0]) if len(occupied) else -1)
    return columns


def nqueens_energy(
    spins: Sequence[int] | np.ndarray,
    coupling_matrix: np.ndarray | None = None,
) -> float:
    """Conflict energy for a flat 0/1 spin vector.

    E = -sum_{a<b} s_a * s_b * J[a,b]. With J[a,b] = -1 for attacking
    pairs, valid non-attacking placements have E=0 and each conflict adds 1.
    """
    s = np.asarray(spins, dtype=np.float32).reshape((N_SPINS,))
    J = _COUPLING_MATRIX if coupling_matrix is None else coupling_matrix  # noqa: N806
    energy = -0.5 * float(s @ J @ s)
    return 0.0 if abs(energy) < 1e-7 else float(energy)


def is_valid_nqueens(spins: Sequence[int] | np.ndarray, n: int = BOARD_SIZE) -> bool:
    """True when the spin vector is a complete non-attacking N-Queens board."""
    grid = np.asarray(spins, dtype=np.int8).reshape((n, n))
    if int(grid.sum()) != n:
        return False
    if any(int(grid[row].sum()) != 1 for row in range(n)):
        return False
    if any(int(grid[:, col].sum()) != 1 for col in range(n)):
        return False

    positions = np.argwhere(grid == 1)
    for i, (row_a, col_a) in enumerate(positions):
        for row_b, col_b in positions[i + 1 :]:
            if abs(int(row_a) - int(row_b)) == abs(int(col_a) - int(col_b)):
                return False
    return True


def _solve_nqueens_backtracking(n: int = BOARD_SIZE) -> list[int]:
    """Deterministic exact N-Queens solver used when stochastic sampling misses."""
    columns: list[int] = []
    used_cols: set[int] = set()
    used_diag_down: set[int] = set()
    used_diag_up: set[int] = set()

    def search(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            diag_down = row - col
            diag_up = row + col
            if col in used_cols or diag_down in used_diag_down or diag_up in used_diag_up:
                continue
            columns.append(col)
            used_cols.add(col)
            used_diag_down.add(diag_down)
            used_diag_up.add(diag_up)
            if search(row + 1):
                return True
            columns.pop()
            used_cols.remove(col)
            used_diag_down.remove(diag_down)
            used_diag_up.remove(diag_up)
        return False

    if not search(0):
        raise RuntimeError(f"No N-Queens solution exists for n={n}")
    return columns


def _starting_columns(target_columns: Sequence[int]) -> list[int]:
    """A deterministic near-solution with several visible conflicts."""
    columns = list(target_columns)
    for row, col in [(1, 0), (2, 2), (3, 3), (7, 4)]:
        columns[row] = col if col != target_columns[row] else (col + 1) % BOARD_SIZE
    return columns


@dataclass
class NQueensState:
    """Flat spin board plus the target zero-energy board used for animation."""

    spins: list[int]
    target_spins: list[int] = field(default_factory=list)
    step_idx: int = 0

    def clone(self) -> NQueensState:
        return NQueensState(
            spins=list(self.spins),
            target_spins=list(self.target_spins),
            step_idx=self.step_idx,
        )

    def queen_positions(self) -> list[tuple[int, int]]:
        grid = np.asarray(self.spins, dtype=np.int8).reshape((BOARD_SIZE, BOARD_SIZE))
        return [(int(row), int(col)) for row, col in np.argwhere(grid == 1)]


class NQueensGame(WOPRGame[NQueensState, tuple[int, int]]):
    """WOPR cartridge for the classic 8-Queens placement CSP."""

    name = "N_QUEENS"
    description = "8x8 CLASSICAL CSP. 64-SPIN ISING MODEL. ENERGY=ATTACKING PAIRS."
    accent_color = "#66d9ef"

    def __init__(self, seed: int = 17, penalty: float = DEFAULT_PENALTY) -> None:
        self._seed = seed
        self._penalty = penalty
        self._J_np = build_nqueens_coupling_matrix(BOARD_SIZE, penalty)
        self.ising_solver_used = False

        if _ISING_AVAILABLE:
            self._J = jnp.asarray(self._J_np, dtype=jnp.float32)
            self._biases = jnp.ones((N_SPINS,), dtype=jnp.float32) * (0.6 * penalty)
            self._sampler = ParallelIsingSampler(
                n_warmup=300,
                n_samples=96,
                steps_per_sample=12,
                schedule=AnnealingSchedule(
                    beta_init=0.25, beta_final=12.0, schedule_type="geometric"
                ),
                use_checkerboard=True,
            )

    def coupling_matrix(self) -> np.ndarray:
        """Return a copy of the 64x64 Ising coupling matrix."""
        return self._J_np.copy()

    def _ising_solve(self) -> np.ndarray | None:
        """Try ParallelIsingSampler and return a valid zero-energy sample if found."""
        if not _ISING_AVAILABLE:
            return None

        key = jrandom.PRNGKey(self._seed ^ 0x51A7)
        samples = self._sampler.sample(key, self._biases, self._J, beta=12.0)
        self.ising_solver_used = True

        for sample in np.asarray(samples, dtype=np.int8):
            if is_valid_nqueens(sample) and nqueens_energy(sample, self._J_np) == 0.0:
                return sample.astype(np.int8)
        return None

    def _target_spins(self) -> np.ndarray:
        sampled = self._ising_solve()
        if sampled is not None:
            return sampled
        return columns_to_spins(_solve_nqueens_backtracking())

    def initial_state(self) -> NQueensState:
        target_spins = self._target_spins()
        target_columns = spins_to_columns(target_spins)
        start_spins = columns_to_spins(_starting_columns(target_columns))
        return NQueensState(
            spins=start_spins.astype(int).tolist(),
            target_spins=target_spins.astype(int).tolist(),
        )

    def energy(self, state: NQueensState) -> float:
        return nqueens_energy(state.spins, self._J_np)

    def is_solved(self, state: NQueensState) -> bool:
        return self.energy(state) == 0.0 and is_valid_nqueens(state.spins)

    def _move_one_row_toward_target(self, state: NQueensState) -> tuple[NQueensState, int | None]:
        current = np.asarray(state.spins, dtype=np.int8).reshape((BOARD_SIZE, BOARD_SIZE))
        target = np.asarray(state.target_spins, dtype=np.int8).reshape((BOARD_SIZE, BOARD_SIZE))
        mismatched_rows = [
            row for row in range(BOARD_SIZE) if not np.array_equal(current[row], target[row])
        ]
        if not mismatched_rows:
            return state, None

        best_row = mismatched_rows[0]
        best_energy = float("inf")
        best_grid: np.ndarray | None = None
        for row in mismatched_rows:
            trial = current.copy()
            trial[row, :] = target[row, :]
            trial_energy = nqueens_energy(trial.reshape(-1), self._J_np)
            if trial_energy < best_energy:
                best_row = row
                best_energy = trial_energy
                best_grid = trial

        moved = state.clone()
        moved.spins = (
            best_grid.reshape(-1).astype(int).tolist() if best_grid is not None else state.spins
        )
        moved.step_idx += 1
        return moved, best_row

    def carnot_step(self, state: NQueensState, iteration: int) -> StepResult[NQueensState]:
        if self.is_solved(state):
            return StepResult(
                state=state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation=OPTIMAL_MESSAGE,
            )

        new_state = state
        moved_row: int | None = None
        if iteration % ANIMATION_INTERVAL == 0:
            new_state, moved_row = self._move_one_row_toward_target(state)

        new_energy = self.energy(new_state)
        solved = self.is_solved(new_state)
        if solved:
            annotation = OPTIMAL_MESSAGE
        elif moved_row is not None:
            positions = " ".join(f"({row},{col})" for row, col in new_state.queen_positions())
            annotation = (
                f"ITER {iteration:05d}. ROW {moved_row + 1} REPOSITIONED. "
                f"ENERGY={new_energy:.0f}. QUEENS {positions}."
            )
        else:
            annotation = THREAT_ASSESSMENT

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=solved,
            annotation=annotation,
        )

    def visualize(self, state: NQueensState, energy: float) -> str:
        """Render an 8x8 ASCII chessboard in WOPR terminal styling."""
        grid = np.asarray(state.spins, dtype=np.int8).reshape((BOARD_SIZE, BOARD_SIZE))
        rows = []
        for row in range(BOARD_SIZE):
            cells = ["Q" if grid[row, col] else "." for col in range(BOARD_SIZE)]
            rows.append(" ".join(cells))

        headline = OPTIMAL_MESSAGE if energy == 0.0 and self.is_solved(state) else THREAT_ASSESSMENT
        board = "\n".join(rows)
        return (
            f'<div style="color:{self.accent_color};font-family:JetBrains Mono,monospace;'
            f'font-size:13px;line-height:1.45;">'
            f'<div style="padding-bottom:8px;">{headline}</div>'
            f'<pre style="margin:0;color:{self.accent_color};font-family:JetBrains Mono,'
            f'monospace;font-size:18px;line-height:1.35;">{board}</pre>'
            f'<div style="padding-top:8px;">ENERGY = {energy:.0f}</div>'
            "</div>"
        )


__all__ = [
    "ANIMATION_INTERVAL",
    "BOARD_SIZE",
    "DEFAULT_PENALTY",
    "NQueensGame",
    "NQueensState",
    "N_SPINS",
    "OPTIMAL_MESSAGE",
    "THREAT_ASSESSMENT",
    "build_nqueens_coupling_matrix",
    "columns_to_spins",
    "is_valid_nqueens",
    "nqueens_energy",
    "spins_to_columns",
]
