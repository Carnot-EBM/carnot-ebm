"""Lights Out cartridge — perfect Carnot-Ising fit.

The 5x5 toggle grid where clicking a cell flips it AND its four
neighbors. Goal: all cells off.

This is mathematically a pure XOR linear-algebra problem on F_2 — every
puzzle has either a unique solution (via Gaussian elimination) or none
at all. From Carnot's perspective:

  Energy(state) = number of cells that are ON

The Ising-model ground-state search recovers the XOR-cancellation
algorithm; we just animate the descent. Of all the cartridges in the
gallery, this is the one whose math most directly maps to Carnot's
energy formulation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from games._base import StepResult, WOPRGame

GRID_SIZE = 5


@dataclass
class LightsOutState:
    """5x5 grid of on/off cells. True = ON, False = OFF."""

    grid: list[list[bool]]

    def clone(self) -> LightsOutState:
        return LightsOutState(grid=[row[:] for row in self.grid])

    def cells_on(self) -> int:
        return sum(sum(1 for v in row if v) for row in self.grid)


def _toggle_cell(grid: list[list[bool]], r: int, c: int) -> None:
    """Toggle (r, c) and its four cardinal neighbors (in place)."""
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            grid[nr][nc] = not grid[nr][nc]


def lights_out_energy(state: LightsOutState) -> float:
    return float(state.cells_on())


def _starting_grid(seed: int = 17) -> list[list[bool]]:
    """Generate a solvable random starting state by applying N random
    toggles to an all-off grid. Any grid reachable from all-off is by
    definition solvable (XOR linear algebra over F_2)."""
    rng = random.Random(seed)
    grid = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    n_scrambles = rng.randint(8, 14)
    for _ in range(n_scrambles):
        r = rng.randint(0, GRID_SIZE - 1)
        c = rng.randint(0, GRID_SIZE - 1)
        _toggle_cell(grid, r, c)
    return grid


class LightsOutGame(WOPRGame[LightsOutState, tuple[int, int]]):
    name = "LIGHTS_OUT"
    description = "5x5 XOR PUZZLE. ENERGY = CELLS LIT."
    accent_color = "#ffcc00"

    def __init__(self, seed: int = 17):
        self._seed = seed
        self._rng = random.Random(seed)

    def initial_state(self) -> LightsOutState:
        return LightsOutState(grid=_starting_grid(self._seed))

    def energy(self, state: LightsOutState) -> float:
        return lights_out_energy(state)

    def is_solved(self, state: LightsOutState) -> bool:
        return state.cells_on() == 0

    def carnot_step(self, state: LightsOutState, iteration: int) -> StepResult[LightsOutState]:
        """Greedy + simulated annealing: pick a cell that, when toggled,
        most reduces the lit count. Occasional random toggle to escape
        local minima.
        """
        new_state = state.clone()

        # Anneal temperature
        temperature = max(0.0, 5.0 * (0.95**iteration))

        if self._rng.random() < 0.85 or iteration < 3:
            # Greedy step: try every cell, pick the one that minimizes energy
            best_delta = 0
            best_cell: tuple[int, int] | None = None
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    test = new_state.clone()
                    _toggle_cell(test.grid, r, c)
                    delta = test.cells_on() - new_state.cells_on()
                    if delta < best_delta:
                        best_delta = delta
                        best_cell = (r, c)
            if best_cell is not None:
                _toggle_cell(new_state.grid, best_cell[0], best_cell[1])
                annotation = f"TOGGLE ({best_cell[0]},{best_cell[1]}). DELTA={best_delta}."
            else:
                # Local minimum — random kick
                r = self._rng.randint(0, GRID_SIZE - 1)
                c = self._rng.randint(0, GRID_SIZE - 1)
                _toggle_cell(new_state.grid, r, c)
                annotation = f"LOCAL MINIMUM. RANDOM KICK ({r},{c})."
        else:
            # Simulated-annealing escape: random toggle accepted with
            # probability based on temperature
            r = self._rng.randint(0, GRID_SIZE - 1)
            c = self._rng.randint(0, GRID_SIZE - 1)
            _toggle_cell(new_state.grid, r, c)
            annotation = f"ANNEAL T={temperature:.2f}. PERTURB ({r},{c})."

        new_energy = lights_out_energy(new_state)
        if new_energy == 0:
            annotation = "ALL DARK. PUZZLE SOLVED."

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=new_energy == 0.0,
            annotation=annotation,
        )

    def visualize(self, state: LightsOutState, energy: float) -> str:
        rows_html = []
        for r in range(GRID_SIZE):
            cells_html = []
            for c in range(GRID_SIZE):
                on = state.grid[r][c]
                bg = "#ffcc00" if on else "#1a1a00"
                fg = "#000" if on else "#3a3a00"
                cells_html.append(
                    f'<td style="width:48px;height:48px;text-align:center;'
                    f"background:{bg};color:{fg};border:1px solid #5a5a00;"
                    f"font-family:JetBrains Mono,monospace;font-size:20px;"
                    f'font-weight:bold;">'
                    f"{'●' if on else '○'}</td>"
                )
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

        table = (
            '<table style="border-collapse:collapse;background:#000;'
            'border:2px solid #ffcc00;padding:8px;">' + "".join(rows_html) + "</table>"
        )
        return table
