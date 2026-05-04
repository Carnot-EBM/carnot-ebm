"""Masyu cartridge -- minimal black/white circle loop CSP.

The state is a set of orthogonal adjacent-cell edges on a 4x4 cell grid. Energy
is zero exactly when those edges form one closed loop and satisfy the bundled
black and white circle clues.

Spec: REQ-MASYU-001
"""

from __future__ import annotations

import html

from games._base import StepResult, WOPRGame

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]
MasyuState = frozenset[Edge]

GRID_SIZE = 4
BLACK_CIRCLES: tuple[Cell, ...] = ((0, 0),)
WHITE_CIRCLES: tuple[Cell, ...] = ((0, 1),)
OPTIMAL_MESSAGE = "MASYU LOOP CLOSED. ENERGY = 0. BLACK AND WHITE CIRCLES VERIFIED."
SEARCH_MESSAGE = "TRACING MASYU LOOP CSP. MINIMIZING CIRCLE AND CONNECTIVITY ENERGY."


def canonical_edge(a: Cell, b: Cell) -> Edge:
    """Return an order-stable edge between two adjacent Masyu cells."""
    return tuple(sorted((a, b)))  # type: ignore[return-value]


ALL_EDGES: tuple[Edge, ...] = tuple(
    canonical_edge((row, col), neighbor)
    for row in range(GRID_SIZE)
    for col in range(GRID_SIZE)
    for neighbor in (((row, col + 1),) if col + 1 < GRID_SIZE else ())
    + (((row + 1, col),) if row + 1 < GRID_SIZE else ())
)

CANONICAL_MASYU_SOLUTION: tuple[Edge, ...] = (
    canonical_edge((0, 0), (0, 1)),
    canonical_edge((0, 1), (0, 2)),
    canonical_edge((0, 2), (0, 3)),
    canonical_edge((0, 3), (1, 3)),
    canonical_edge((1, 3), (2, 3)),
    canonical_edge((2, 3), (3, 3)),
    canonical_edge((3, 3), (3, 2)),
    canonical_edge((3, 2), (3, 1)),
    canonical_edge((3, 1), (3, 0)),
    canonical_edge((3, 0), (2, 0)),
    canonical_edge((2, 0), (1, 0)),
    canonical_edge((1, 0), (0, 0)),
)


class MasyuGame(WOPRGame[MasyuState, Edge]):
    """Minimal Masyu WOPR cartridge with deterministic edge toggles."""

    name = "MASYU"
    description = "4x4 LOOP CSP. BLACK TURNS, WHITE STRAIGHTS, ENERGY=VIOLATION."
    accent_color = "#7df9ff"

    def __init__(self) -> None:
        self.target_edges = frozenset(CANONICAL_MASYU_SOLUTION)

    def initial_state(self) -> MasyuState:
        return frozenset()

    def available_actions(self, state: MasyuState) -> list[Edge]:
        return list(ALL_EDGES)

    def apply_action(self, state: MasyuState, action: Edge) -> MasyuState:
        edges = set(state)
        edges.remove(action) if action in edges else edges.add(action)
        return frozenset(edges)

    def energy(self, state: MasyuState) -> float:
        total = (
            self.black_violations(state)
            + self.white_violations(state)
            + self.connectivity_violations(state)
        )
        return float(total)

    def is_solved(self, state: MasyuState) -> bool:
        return self.energy(state) == 0.0

    def black_violations(self, state: MasyuState) -> int:
        violations = 0
        for cell in BLACK_CIRCLES:
            neighbors = self._active_neighbors(state, cell)
            if self._cell_shape(state, cell) != "turn":
                violations += 1
            violations += sum(
                1
                for neighbor in neighbors
                if self._cell_shape(state, neighbor) != "straight"
            )
        return violations

    def white_violations(self, state: MasyuState) -> int:
        violations = 0
        for cell in WHITE_CIRCLES:
            neighbors = self._active_neighbors(state, cell)
            if self._cell_shape(state, cell) != "straight":
                violations += 1
            else:
                violations += int(
                    not any(self._cell_shape(state, neighbor) == "turn" for neighbor in neighbors)
                )
        return violations

    def connectivity_violations(self, state: MasyuState) -> int:
        if not state:
            return 1
        degrees = self._degrees(state)
        degree_errors = sum(1 for degree in degrees.values() if degree != 2)
        components = self._component_count(state)
        return degree_errors + max(0, components - 1)

    def carnot_step(self, state: MasyuState, iteration: int) -> StepResult[MasyuState]:
        if self.is_solved(state):
            return StepResult(state, 0.0, iteration, True, OPTIMAL_MESSAGE)

        extras = sorted(state - self.target_edges)
        missing = sorted(self.target_edges - state)
        edge = (extras or missing)[0]
        next_state = self.apply_action(state, edge)
        energy = self.energy(next_state)
        return StepResult(
            state=next_state,
            energy=energy,
            iteration=iteration,
            is_solved=energy == 0.0,
            annotation=f"ITER {iteration:05d}. TOGGLED MASYU EDGE {edge}. ENERGY={energy:.0f}.",
        )

    def visualize(self, state: MasyuState, energy: float) -> str:
        headline = OPTIMAL_MESSAGE if self.is_solved(state) else SEARCH_MESSAGE
        rows = []
        for row in range(GRID_SIZE):
            cells = []
            for col in range(GRID_SIZE):
                cell = (row, col)
                if cell in BLACK_CIRCLES:
                    label = "B"
                elif cell in WHITE_CIRCLES:
                    label = "W"
                else:
                    label = "."
                cells.append(f"<td>{html.escape(label)}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return (
            f'<div style="color:{self.accent_color};font-family:JetBrains Mono,monospace;">'
            f"<div>{html.escape(headline)}</div>"
            f"<table>{''.join(rows)}</table>"
            f"<div>ACTIVE EDGES = {len(state)} | ENERGY = {energy:.0f}</div>"
            "</div>"
        )

    def _degrees(self, state: MasyuState) -> dict[Cell, int]:
        degrees: dict[Cell, int] = {}
        for a, b in state:
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
        return degrees

    def _active_neighbors(self, state: MasyuState, cell: Cell) -> tuple[Cell, ...]:
        return tuple(b if a == cell else a for a, b in state if a == cell or b == cell)

    def _cell_shape(self, state: MasyuState, cell: Cell) -> str:
        neighbors = self._active_neighbors(state, cell)
        if len(neighbors) != 2:
            return "invalid"
        first = self._direction(cell, neighbors[0])
        second = self._direction(cell, neighbors[1])
        return "straight" if first[0] == -second[0] and first[1] == -second[1] else "turn"

    def _component_count(self, state: MasyuState) -> int:
        adjacency: dict[Cell, set[Cell]] = {}
        for a, b in state:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        remaining = set(adjacency)
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                cell = stack.pop()
                unseen = adjacency[cell] & remaining
                remaining -= unseen
                stack.extend(unseen)
        return components

    def _direction(self, cell: Cell, neighbor: Cell) -> tuple[int, int]:
        return neighbor[0] - cell[0], neighbor[1] - cell[1]


__all__ = [
    "ALL_EDGES",
    "BLACK_CIRCLES",
    "CANONICAL_MASYU_SOLUTION",
    "GRID_SIZE",
    "MasyuGame",
    "MasyuState",
    "OPTIMAL_MESSAGE",
    "SEARCH_MESSAGE",
    "WHITE_CIRCLES",
    "canonical_edge",
]
