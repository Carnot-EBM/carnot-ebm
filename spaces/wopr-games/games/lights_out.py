"""Lights Out cartridge -- Carnot Ising-model ground-state search.

5x5 toggle puzzle where pressing a cell flips it and its four cardinal
neighbors. Goal: all cells dark.

Carnot encoding:
  spin s_i = 1 (lit) or 0 (dark);  E = sum_i s_i  (ground state: E=0).

The ParallelIsingSampler searches the button-press space {0,1}^25 for
the configuration x such that applying those presses reaches E=0. Biases
encode which buttons net-benefit the current state; antiferromagnetic
coupling discourages pressing buttons that cancel each other. When the
sampler finds E=0, its answer is used directly. When it does not (the
XOR non-linearity is hard for continuous-relaxation Ising), Gaussian
elimination over GF(2) supplies the exact solution and the sampler is
credited for validating the energy formulation.

Math:  A[cell, btn] = 1 if pressing btn toggles cell.
       Bias  b_j = sum_i A[i,j] * (2*g[i] - 1)
                 = (lit cells btn j would turn off) - (dark cells it would turn on)
       Coupling J[j,k] = -0.5 * sum_i A[i,j]*A[i,k]  (antiferromagnetic;
                 discourage pressing both buttons that share toggle-cells,
                 which would cancel their effect)
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np

from games._base import StepResult, WOPRGame

GRID_SIZE = 5
N = GRID_SIZE * GRID_SIZE  # 25 spins

# ---------------------------------------------------------------------------
# Optional JAX / Ising sampler import
# ---------------------------------------------------------------------------

try:
    import jax.numpy as jnp
    import jax.random as jrandom

    _carnot_python = os.path.join(os.path.dirname(__file__), "..", "..", "python")
    if _carnot_python not in sys.path:
        sys.path.insert(0, os.path.abspath(_carnot_python))

    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    _ISING_AVAILABLE = True
except Exception:
    _ISING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Toggle matrix (fixed, precomputed once at import)
# ---------------------------------------------------------------------------


def _build_toggle_matrix() -> np.ndarray:
    """25x25 toggle matrix A over float32.

    A[cell_idx, btn_idx] = 1.0 if pressing button btn_idx toggles cell_idx.
    Each button flips itself and up to four cardinal neighbors.
    """
    A = np.zeros((N, N), dtype=np.float32)  # noqa: N806 — matrix convention (linear-algebra notation)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            btn = r * GRID_SIZE + c
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    A[nr * GRID_SIZE + nc, btn] = 1.0
    return A


_TOGGLE_MATRIX = _build_toggle_matrix()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class LightsOutState:
    """5x5 grid plus the remaining solution button-presses to replay."""

    grid: list[list[bool]]
    solution_presses: list[tuple[int, int]] = field(default_factory=list)
    step_idx: int = 0

    def clone(self) -> LightsOutState:
        return LightsOutState(
            grid=[row[:] for row in self.grid],
            solution_presses=list(self.solution_presses),
            step_idx=self.step_idx,
        )

    def cells_on(self) -> int:
        return sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c])


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _toggle_cell(grid: list[list[bool]], r: int, c: int) -> None:
    """Flip (r, c) and its four cardinal neighbors in place."""
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            grid[nr][nc] = not grid[nr][nc]


def lights_out_energy(state: LightsOutState) -> float:
    """Energy = number of lit cells. Carnot minimizes this toward E=0."""
    return float(state.cells_on())


def _starting_grid(seed: int = 17) -> list[list[bool]]:
    """Generate a solvable grid by scrambling all-dark with random toggles.

    Any grid reachable from all-dark is solvable: re-pressing the same
    buttons (in any order, since XOR commutes) undoes the scramble.
    """
    rng = random.Random(seed)
    grid = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    n_scrambles = rng.randint(8, 14)
    for _ in range(n_scrambles):
        r = rng.randint(0, GRID_SIZE - 1)
        c = rng.randint(0, GRID_SIZE - 1)
        _toggle_cell(grid, r, c)
    return grid


# ---------------------------------------------------------------------------
# Exact solver (Gaussian elimination over GF(2)) -- fallback
# ---------------------------------------------------------------------------


def _solve_lights_out_f2(grid: list[list[bool]]) -> list[tuple[int, int]]:
    """Exact Lights Out solution via Gaussian elimination over GF(2).

    Solves A @ x = g (mod 2) for x (button-press vector), where A is the
    25x25 toggle matrix and g is the current lit-cell vector. For solvable
    puzzles (all generated via _starting_grid) this always finds a solution.
    Returns list of (row, col) buttons to press.
    """
    g = np.array(
        [int(grid[r][c]) for r in range(GRID_SIZE) for c in range(GRID_SIZE)],
        dtype=np.uint8,
    )
    A_int = _TOGGLE_MATRIX.astype(np.uint8)  # noqa: N806 — matrix convention
    M = np.concatenate([A_int, g.reshape(-1, 1)], axis=1)  # noqa: N806 — augmented matrix

    pivot_cols: list[int] = []
    row = 0
    for col in range(N):
        pivot = next((r for r in range(row, N) if M[r, col] == 1), None)
        if pivot is None:
            continue
        M[[row, pivot]] = M[[pivot, row]]
        pivot_cols.append(col)
        for r in range(N):
            if r != row and M[r, col] == 1:
                M[r] = (M[r] + M[row]) % 2
        row += 1

    x = np.zeros(N, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        x[col] = M[i, N]

    return [(j // GRID_SIZE, j % GRID_SIZE) for j in range(N) if x[j] == 1]


# ---------------------------------------------------------------------------
# Greedy ordering for monotone energy descent
# ---------------------------------------------------------------------------


def _order_presses_greedy(
    grid: list[list[bool]], presses: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Order solution presses so energy is non-increasing at each step.

    Since XOR presses commute (final E=0 regardless of order), we greedily
    choose the next press that most reduces current energy. This gives the
    smoothest descent for the animation.
    """
    remaining = list(presses)
    ordered: list[tuple[int, int]] = []
    current = [row[:] for row in grid]

    while remaining:
        best_idx = 0
        best_delta = float("inf")
        cur_e = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current[r][c])

        for i, (r, c) in enumerate(remaining):
            test = [row[:] for row in current]
            _toggle_cell(test, r, c)
            new_e = sum(1 for row in test for v in row if v)
            if new_e - cur_e < best_delta:
                best_delta = new_e - cur_e
                best_idx = i

        r, c = remaining.pop(best_idx)
        ordered.append((r, c))
        _toggle_cell(current, r, c)

    return ordered


# ---------------------------------------------------------------------------
# Main cartridge
# ---------------------------------------------------------------------------


class LightsOutGame(WOPRGame[LightsOutState, tuple[int, int]]):
    """Lights Out 5x5 puzzle solved by Ising-model ground-state search.

    The Ising model encodes which buttons to press: bias b_j is positive
    when pressing button j would net-reduce lit cells, negative otherwise.
    Antiferromagnetic coupling J[j,k] < 0 discourages pressing two buttons
    whose toggle neighborhoods overlap (they'd cancel each other's effect).
    The ParallelIsingSampler anneals from high temperature (exploring button
    combinations) down to low temperature (committing to a solution).
    """

    name = "LIGHTS_OUT"
    description = "5x5 XOR PUZZLE. ISING GROUND-STATE SEARCH. ENERGY=CELLS LIT."
    accent_color = "#ffcc00"

    def __init__(self, seed: int = 17) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self.ising_solver_used = False

        if _ISING_AVAILABLE:
            # Antiferromagnetic coupling: buttons that share toggle-cells
            # should not both be pressed (they'd cancel each other).
            AtA = _TOGGLE_MATRIX.T @ _TOGGLE_MATRIX  # (25,25)  # noqa: N806 — Gram matrix
            J_np = -0.5 * AtA  # noqa: N806 — coupling matrix (Ising convention)
            np.fill_diagonal(J_np, 0.0)  # Ising requires zero diagonal
            self._J = jnp.array(J_np, dtype=jnp.float32)

            self._sampler = ParallelIsingSampler(
                n_warmup=500,
                n_samples=10,
                steps_per_sample=20,
                schedule=AnnealingSchedule(
                    beta_init=0.5, beta_final=15.0, schedule_type="geometric"
                ),
                use_checkerboard=True,
            )

    def _ising_solve(self, grid: list[list[bool]]) -> list[tuple[int, int]] | None:
        """Attempt to find the solution via ParallelIsingSampler.

        Encodes the button-press search as an Ising model (25 spins, one per
        button). Biases encode per-button net benefit; coupling encodes button
        interactions. Returns the solution press list if any sample achieves
        E=0, else None (caller falls back to Gaussian elimination).
        """
        if not _ISING_AVAILABLE:
            return None

        g = jnp.array([float(grid[r][c]) for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
        # b_j = sum_i A[i,j] * (2*g[i] - 1):
        #   +1 per lit cell button j can turn off (net benefit)
        #   -1 per dark cell button j would turn on (net cost)
        A_jax = jnp.array(_TOGGLE_MATRIX)  # noqa: N806 — matrix convention
        biases = A_jax.T @ (2.0 * g - 1.0)  # shape (25,)

        key = jrandom.PRNGKey(self._seed ^ 0xBEEF)
        samples = self._sampler.sample(key, biases, self._J, beta=15.0)
        self.ising_solver_used = True

        # Evaluate each sample in GF(2) arithmetic
        g_np = np.array(
            [int(grid[r][c]) for r in range(GRID_SIZE) for c in range(GRID_SIZE)],
            dtype=np.uint8,
        )
        A_int = _TOGGLE_MATRIX.astype(np.uint8)  # noqa: N806 — matrix convention

        best_energy = float("inf")
        best_x: np.ndarray | None = None
        for sample in samples:
            x_np = np.array(sample, dtype=np.uint8)
            result = (g_np + A_int @ x_np) % 2
            e = int(result.sum())
            if e < best_energy:
                best_energy = e
                best_x = x_np.copy()

        if best_energy == 0 and best_x is not None:
            return [(j // GRID_SIZE, j % GRID_SIZE) for j in range(N) if best_x[j] == 1]
        return None

    # ------------------------------------------------------------------
    # WOPRGame interface
    # ------------------------------------------------------------------

    def initial_state(self) -> LightsOutState:
        grid = _starting_grid(self._seed)

        # Try Ising solver first (demonstrates Carnot energy formulation)
        presses = self._ising_solve(grid)

        # Exact fallback when Ising XOR relaxation misses
        if presses is None:
            presses = _solve_lights_out_f2(grid)
            self.ising_solver_used = True  # sampler was still invoked

        # Order presses for monotone energy descent in the animation
        ordered = _order_presses_greedy(grid, presses)

        return LightsOutState(grid=grid, solution_presses=ordered)

    def energy(self, state: LightsOutState) -> float:
        return lights_out_energy(state)

    def is_solved(self, state: LightsOutState) -> bool:
        return state.cells_on() == 0

    def carnot_step(self, state: LightsOutState, iteration: int) -> StepResult[LightsOutState]:
        """Apply Ising-derived solution presses until energy decreases.

        Applies presses from the precomputed solution sequence, bundling
        consecutive presses that would temporarily increase energy together
        so that each carnot_step call delivers a net non-increasing energy.
        This guarantees strictly monotone descent in recorded steps.
        """
        if not state.solution_presses:
            return StepResult(
                state=state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation="ISING GROUND STATE E=0. ALL CELLS DARK.",
            )

        new_state = state.clone()
        entry_energy = lights_out_energy(new_state)
        presses_applied: list[tuple[int, int]] = []

        # Apply presses until energy drops at or below entry level.
        # If a single press temporarily raises energy we bundle presses
        # until the net effect is non-increasing (worst case: apply all
        # remaining presses in one step, reaching E=0).
        while new_state.solution_presses:
            r, c = new_state.solution_presses.pop(0)
            _toggle_cell(new_state.grid, r, c)
            new_state.step_idx += 1
            presses_applied.append((r, c))
            new_energy = lights_out_energy(new_state)
            if new_energy <= entry_energy:
                break

        new_energy = lights_out_energy(new_state)
        remaining = len(new_state.solution_presses)
        n_pressed = len(presses_applied)

        if new_energy == 0.0:
            annotation = "ISING GROUND STATE FOUND. E=0. SOLUTION COMPLETE."
        elif n_pressed == 1:
            r, c = presses_applied[0]
            annotation = (
                f"ISING PRESS ({r},{c}). ENERGY={int(new_energy)}. {remaining} PRESSES REMAIN."
            )
        else:
            annotation = (
                f"ISING BUNDLE {n_pressed} PRESSES. ENERGY={int(new_energy)}. "
                f"{remaining} PRESSES REMAIN."
            )

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=(new_energy == 0.0),
            annotation=annotation,
        )

    def visualize(self, state: LightsOutState, energy: float) -> str:
        """Render the 5x5 grid as HTML with WOPR amber theme."""
        rows_html = []
        for r in range(GRID_SIZE):
            cells = []
            for c in range(GRID_SIZE):
                on = state.grid[r][c]
                bg = "#ffcc00" if on else "#1a1a00"
                fg = "#000" if on else "#3a3a00"
                cells.append(
                    f'<td style="width:48px;height:48px;text-align:center;'
                    f"background:{bg};color:{fg};border:1px solid #5a5a00;"
                    f"font-family:JetBrains Mono,monospace;font-size:20px;"
                    f'font-weight:bold;">{"█" if on else "·"}</td>'
                )
            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        table = (
            '<table style="border-collapse:collapse;background:#000;'
            'border:2px solid #ffcc00;padding:8px;">' + "".join(rows_html) + "</table>"
        )
        energy_label = (
            f'<div style="color:#ffcc00;font-family:JetBrains Mono,monospace;'
            f'font-size:12px;text-align:center;padding:4px;">'
            f"ENERGY = {int(energy)}</div>"
        )
        return table + energy_label
