"""Sudoku cartridge — the canonical Carnot energy-based reasoning demo.

Energy formulation:
  E(grid) = number of constraint violations
         = duplicates_in_rows + duplicates_in_cols + duplicates_in_3x3_boxes

Carnot minimizes this energy via simulated-annealing Gibbs sampling
over cell assignments, holding clues fixed. As the temperature cools,
the sampler converges on the unique 0-energy solution.

This is the pedagogical demo: every other cartridge in the gallery
is a variation on the same energy-minimization theme.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from games._base import StepResult, WOPRGame

SudokuGrid = list[list[int]]  # 9x9 grid of ints, 0 = empty


# A classic "easy" puzzle for the demo (real solvers find it in <1s)
DEMO_PUZZLE: SudokuGrid = [
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


@dataclass
class SudokuState:
    """A Sudoku grid + which cells are clues (immutable)."""

    grid: SudokuGrid
    clues: list[list[bool]]  # True where the cell is a clue
    temperature: float = 1.0

    def clone(self) -> SudokuState:
        return SudokuState(
            grid=[row[:] for row in self.grid],
            clues=[row[:] for row in self.clues],
            temperature=self.temperature,
        )


def _count_duplicates(values: list[int]) -> int:
    """Count duplicate non-zero values in a list of nine cells."""
    seen: dict[int, int] = {}
    for v in values:
        if v == 0:
            continue
        seen[v] = seen.get(v, 0) + 1
    return sum(c - 1 for c in seen.values() if c > 1)


def sudoku_energy(grid: SudokuGrid) -> float:
    """Energy = total constraint violations across rows, cols, 3x3 boxes."""
    violations = 0
    # Rows
    for row in grid:
        violations += _count_duplicates(row)
    # Cols
    for c in range(9):
        violations += _count_duplicates([grid[r][c] for r in range(9)])
    # 3x3 boxes
    for box_r in range(3):
        for box_c in range(3):
            cells = [grid[box_r * 3 + dr][box_c * 3 + dc] for dr in range(3) for dc in range(3)]
            violations += _count_duplicates(cells)
    return float(violations)


def _empty_cells_per_row(grid: SudokuGrid, clues: list[list[bool]]) -> list[list[int]]:
    """For each row, list the column indices of non-clue cells."""
    return [[c for c in range(9) if not clues[r][c]] for r in range(9)]


def _initial_fill(grid: SudokuGrid, clues: list[list[bool]]) -> SudokuGrid:
    """Fill empty cells row-by-row with values 1-9 such that each row has
    no duplicates within itself. This gives the sampler a sane warm-start
    where row constraints are satisfied; only column + box violations
    remain to be reduced.
    """
    g = [row[:] for row in grid]
    for r in range(9):
        present = {v for v in g[r] if v != 0}
        missing = [v for v in range(1, 10) if v not in present]
        random.shuffle(missing)
        idx = 0
        for c in range(9):
            if not clues[r][c]:
                g[r][c] = missing[idx]
                idx += 1
    return g


class SudokuGame(WOPRGame[SudokuState, tuple[int, int, int]]):
    name = "SUDOKU"
    description = "9x9 CONSTRAINT-SATISFACTION. ENERGY = VIOLATIONS."
    accent_color = "#39ff14"

    def __init__(self, puzzle: SudokuGrid | None = None, seed: int | None = 42):
        self.puzzle = [row[:] for row in (puzzle or DEMO_PUZZLE)]
        self.clues = [[v != 0 for v in row] for row in self.puzzle]
        self.empties_per_row = _empty_cells_per_row(self.puzzle, self.clues)
        self._rng = random.Random(seed)

    def initial_state(self) -> SudokuState:
        return SudokuState(
            grid=_initial_fill(self.puzzle, self.clues),
            clues=self.clues,
            temperature=2.0,
        )

    def energy(self, state: SudokuState) -> float:
        return sudoku_energy(state.grid)

    def is_solved(self, state: SudokuState) -> bool:
        return self.energy(state) == 0.0

    def carnot_step(self, state: SudokuState, iteration: int) -> StepResult[SudokuState]:
        """One Metropolis step: pick a row, swap two non-clue cells, accept
        based on energy delta and temperature.
        """
        new_state = state.clone()
        # Cool the temperature geometrically across 5000 iterations
        new_state.temperature = max(0.05, 2.0 * (0.9995**iteration))

        # Pick a row with at least 2 empty cells; swap two of them
        candidate_rows = [r for r in range(9) if len(self.empties_per_row[r]) >= 2]
        if not candidate_rows:
            # No swappable rows (puzzle was filled by clues alone)
            return StepResult(
                state=new_state,
                energy=self.energy(new_state),
                iteration=iteration,
                is_solved=self.is_solved(new_state),
                annotation="NO MOVES AVAILABLE.",
            )

        row = self._rng.choice(candidate_rows)
        cols = self.empties_per_row[row]
        c1, c2 = self._rng.sample(cols, 2)

        old_energy = sudoku_energy(new_state.grid)
        new_state.grid[row][c1], new_state.grid[row][c2] = (
            new_state.grid[row][c2],
            new_state.grid[row][c1],
        )
        new_energy = sudoku_energy(new_state.grid)

        delta = new_energy - old_energy
        if delta > 0:
            # Metropolis: accept worse moves with prob exp(-delta/T)
            accept_prob = math.exp(-delta / max(new_state.temperature, 1e-9))
            if self._rng.random() > accept_prob:
                # Reject — undo the swap
                new_state.grid[row][c1], new_state.grid[row][c2] = (
                    new_state.grid[row][c2],
                    new_state.grid[row][c1],
                )
                new_energy = old_energy

        # WOPR flavour
        if new_energy == 0:
            annotation = "SOLVED. ALL CONSTRAINTS SATISFIED."
        elif iteration < 50:
            annotation = f"EVALUATING ROW {row + 1}. ENERGY DESCENT IN PROGRESS..."
        elif new_energy < 5:
            annotation = f"NEAR SOLUTION. {int(new_energy)} VIOLATIONS REMAINING."
        else:
            annotation = f"COOLING. T={new_state.temperature:.2f} | E={int(new_energy)}"

        return StepResult(
            state=new_state,
            energy=new_energy,
            iteration=iteration,
            is_solved=new_energy == 0.0,
            annotation=annotation,
        )

    def visualize(self, state: SudokuState, energy: float) -> str:
        """Render the grid as monospace HTML with WOPR colors."""
        rows_html = []
        for r in range(9):
            cells_html = []
            for c in range(9):
                value = state.grid[r][c]
                cell = " " if value == 0 else str(value)
                style = "color:#39ff14;font-weight:bold" if state.clues[r][c] else "color:#9aff9a"
                # Borders for 3x3 boxes
                border = ""
                if r % 3 == 0 and r > 0:
                    border += "border-top:2px solid #39ff14;"
                if c % 3 == 0 and c > 0:
                    border += "border-left:2px solid #39ff14;"
                cells_html.append(
                    f'<td style="width:32px;height:32px;text-align:center;'
                    f"font-family:JetBrains Mono,monospace;font-size:18px;"
                    f'{style};{border}">{cell}</td>'
                )
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

        table = (
            '<table style="border-collapse:collapse;background:#000;'
            'border:2px solid #39ff14;padding:8px;">' + "".join(rows_html) + "</table>"
        )
        return table
