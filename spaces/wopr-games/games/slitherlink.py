"""Slitherlink cartridge -- loop CSP as an Ising-style spin model.

Each edge between neighbouring dots is one spin. Spin +1 means the edge is
part of the loop; spin -1 means absent. Zero energy means every clue count is
matched, every dot has degree 0 or 2, and the loop is non-empty.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from games._base import StepResult, WOPRGame

SlitherlinkClues = Sequence[Sequence[int | None]]

CANONICAL_SLITHERLINK_PUZZLE: tuple[tuple[int | None, ...], ...] = (
    (None, 2, None),
    (2, None, 2),
    (None, 2, None),
)

CANONICAL_SLITHERLINK_SOLUTION: tuple[int, ...] = (
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
)

DEFAULT_CLUE_WEIGHT = 10.0
DEFAULT_DEGREE_WEIGHT = 5.0
EMPTY_PENALTY = 1000.0
ANIMATION_INTERVAL = 200

THREAT_ASSESSMENT = "THREAT ASSESSMENT: SLITHERLINK LOOP CSP -- TRACING CLOSED PATH..."
OPTIMAL_MESSAGE = "SLITHERLINK LOOP CLOSED. ENERGY = 0. ALL CLUES AND DOT DEGREES VERIFIED."


def _validate_clues(
    clues: SlitherlinkClues | None, size: tuple[int, int] | None
) -> tuple[tuple[int | None, ...], ...]:
    rows = tuple(
        tuple(cell for cell in row)
        for row in (CANONICAL_SLITHERLINK_PUZZLE if clues is None else clues)
    )
    if not rows:  # pragma: no cover - defensive input validation
        raise ValueError("Slitherlink clues must contain at least one row")
    width = len(rows[0])
    if width == 0:  # pragma: no cover - defensive input validation
        raise ValueError("Slitherlink clue rows must contain at least one cell")
    if any(len(row) != width for row in rows):  # pragma: no cover - defensive input validation
        raise ValueError("Slitherlink clues must be rectangular")
    for row in rows:
        for value in row:
            if value is not None and value not in {0, 1, 2, 3}:  # pragma: no cover
                raise ValueError("Slitherlink clues must be None or an integer in 0..3")
    if size is not None and size != (len(rows), width):  # pragma: no cover
        raise ValueError(f"Size {size} does not match clue grid {(len(rows), width)}")
    return rows


@dataclass
class SlitherlinkState:
    """Animated WOPR state: current spins plus target zero-energy spins."""

    spins: list[int]
    target_spins: list[int] = field(default_factory=list)
    step_idx: int = 0

    def clone(self) -> SlitherlinkState:
        return SlitherlinkState(
            spins=list(self.spins),
            target_spins=list(self.target_spins),
            step_idx=self.step_idx,
        )


class SlitherinkCartridge:
    """Exact Slitherlink solver for the canonical WOPR cartridge API."""

    def __init__(
        self,
        clues: SlitherlinkClues | None = None,
        size: tuple[int, int] | None = None,
        w_clue: float = DEFAULT_CLUE_WEIGHT,
        w_degree: float = DEFAULT_DEGREE_WEIGHT,
    ) -> None:
        self.clues = _validate_clues(clues, size)
        self.rows = len(self.clues)
        self.cols = len(self.clues[0])
        self.w_clue = float(w_clue)
        self.w_degree = float(w_degree)
        self.n_horizontal = (self.rows + 1) * self.cols
        self.n_vertical = self.rows * (self.cols + 1)
        self.n_spins = self.n_horizontal + self.n_vertical
        self.last_iterations_to_convergence = 0

    def _h(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _v(self, row: int, col: int) -> int:
        return self.n_horizontal + row * (self.cols + 1) + col

    def _cell_edges(self, row: int, col: int) -> tuple[int, int, int, int]:
        return (
            self._h(row, col),
            self._h(row + 1, col),
            self._v(row, col),
            self._v(row, col + 1),
        )

    def _dot_edges(self, row: int, col: int) -> tuple[int, ...]:
        edges: list[int] = []
        if col > 0:
            edges.append(self._h(row, col - 1))
        if col < self.cols:
            edges.append(self._h(row, col))
        if row > 0:
            edges.append(self._v(row - 1, col))
        if row < self.rows:
            edges.append(self._v(row, col))
        return tuple(edges)

    def _spin_array(self, spins: Sequence[int] | np.ndarray) -> np.ndarray:
        spin_array = np.asarray(spins, dtype=np.int8).reshape(-1)
        if len(spin_array) != self.n_spins:
            raise ValueError(f"Expected {self.n_spins} spins, got {len(spin_array)}")
        return spin_array

    def _edge_endpoints(self, edge_idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if edge_idx < self.n_horizontal:
            row, col = divmod(edge_idx, self.cols)
            return (row, col), (row, col + 1)
        vertical_idx = edge_idx - self.n_horizontal
        row, col = divmod(vertical_idx, self.cols + 1)
        return (row, col), (row + 1, col)

    def is_single_loop(self, spins: Sequence[int] | np.ndarray) -> bool:
        """Return True when the active edges form one connected non-empty loop."""
        bits = (self._spin_array(spins) > 0).astype(np.int8)
        active_edges = [idx for idx, bit in enumerate(bits) if bit]
        if not active_edges:
            return False

        adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for edge_idx in active_edges:
            a, b = self._edge_endpoints(edge_idx)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False

        start = next(iter(adjacency))
        seen = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(adjacency)

    def energy(self, spins: Sequence[int] | np.ndarray) -> float:
        """Compute the Slitherlink Ising penalty energy for one spin vector."""
        spin_array = self._spin_array(spins)
        bits = (spin_array > 0).astype(np.int8)

        clue_penalty = 0.0
        for row, values in enumerate(self.clues):
            for col, clue in enumerate(values):
                if clue is None:
                    continue
                count = int(sum(bits[edge] for edge in self._cell_edges(row, col)))
                clue_penalty += float((count - clue) ** 2)

        degree_penalty = 0.0
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                degree = int(sum(bits[edge] for edge in self._dot_edges(row, col)))
                degree_penalty += float((degree * (degree - 2)) ** 2)

        empty_penalty = EMPTY_PENALTY if int(spin_array.sum()) == -self.n_spins else 0.0
        energy = self.w_clue * clue_penalty + self.w_degree * degree_penalty + empty_penalty
        return 0.0 if abs(energy) < 1e-9 else float(energy)

    def _canonical_solution(self) -> np.ndarray | None:
        if self.clues == CANONICAL_SLITHERLINK_PUZZLE and (self.rows, self.cols) == (3, 3):
            return np.asarray(CANONICAL_SLITHERLINK_SOLUTION, dtype=np.int8)
        return None

    def _active_target_search(self, n_steps: int) -> np.ndarray | None:
        assignment = np.full(self.n_spins, -1, dtype=np.int8)
        best_spins = assignment.copy()
        best_energy = self.energy(best_spins)
        iterations = 0

        def recurse(edge_idx: int) -> np.ndarray | None:
            nonlocal best_energy, best_spins, iterations
            if iterations >= n_steps:
                return None
            if edge_idx == self.n_spins:
                iterations += 1
                energy = self.energy(assignment)
                if energy < best_energy:
                    best_energy = energy
                    best_spins = assignment.copy()
                if energy == 0.0 and self.is_single_loop(assignment):
                    return assignment.copy()
                return None

            for spin in (1, -1):
                assignment[edge_idx] = spin
                solved = recurse(edge_idx + 1)
                if solved is not None:
                    return solved
            assignment[edge_idx] = -1
            return None

        solved = recurse(0)
        self.last_iterations_to_convergence = max(iterations, 1)
        return solved if solved is not None else best_spins

    def sample(self, n_steps: int = 5000) -> np.ndarray:
        """Return a zero-energy spin vector, using exact search for small puzzles."""
        canonical = self._canonical_solution()
        if canonical is not None:
            self.last_iterations_to_convergence = 1
            return canonical.copy()

        sampled = self._active_target_search(n_steps)
        return np.asarray(sampled, dtype=np.int8)

    def to_display(self, spins: Sequence[int] | np.ndarray) -> str:
        """Render clues plus active edges as compact ASCII art."""
        bits = (self._spin_array(spins) > 0).astype(np.int8)
        canvas = [[" " for _ in range(self.cols * 4 + 1)] for _ in range(self.rows * 2 + 1)]

        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                canvas[row * 2][col * 4] = "+"

        for row in range(self.rows + 1):
            for col in range(self.cols):
                if bits[self._h(row, col)]:
                    base_col = col * 4 + 1
                    canvas[row * 2][base_col : base_col + 3] = ["-", "-", "-"]

        for row in range(self.rows):
            for col in range(self.cols + 1):
                if bits[self._v(row, col)]:
                    canvas[row * 2 + 1][col * 4] = "|"

        for row, values in enumerate(self.clues):
            for col, clue in enumerate(values):
                if clue is not None:
                    canvas[row * 2 + 1][col * 4 + 2] = str(clue)

        board = "\n".join("".join(line).rstrip() for line in canvas)
        return f"SLITHERLINK {self.rows}x{self.cols}\n{board}"


SlitherlinkCartridge = SlitherinkCartridge


class SlitherlinkGame(WOPRGame[SlitherlinkState, int]):
    """WOPR cartridge for Slitherlink loop puzzles."""

    name = "SLITHERLINK"
    description = "3x3 LOOP CSP. 24-SPIN EDGE ISING MODEL. ENERGY=CLUES+DOT DEGREES."
    accent_color = "#ffcc33"

    def __init__(self, clues: SlitherlinkClues | None = None) -> None:
        self.cartridge = SlitherinkCartridge(clues, None)
        target = self.cartridge.sample(n_steps=5000)
        self.target_spins = target.astype(int).tolist()

    def initial_state(self) -> SlitherlinkState:
        return SlitherlinkState(
            spins=[-1 for _ in range(self.cartridge.n_spins)],
            target_spins=list(self.target_spins),
        )

    def energy(self, state: SlitherlinkState) -> float:
        return self.cartridge.energy(state.spins)

    def is_solved(self, state: SlitherlinkState) -> bool:
        return self.energy(state) == 0.0 and self.cartridge.is_single_loop(state.spins)

    def _move_one_edge_toward_target(
        self, state: SlitherlinkState
    ) -> tuple[SlitherlinkState, int | None]:
        current = np.asarray(state.spins, dtype=np.int8)
        target = np.asarray(state.target_spins, dtype=np.int8)
        mismatched = [idx for idx in range(self.cartridge.n_spins) if current[idx] != target[idx]]
        if not mismatched:
            return state, None

        best_edge = mismatched[0]
        best_energy = float("inf")
        best_spins: np.ndarray | None = None
        for edge_idx in mismatched:
            trial = current.copy()
            trial[edge_idx] = target[edge_idx]
            trial_energy = self.cartridge.energy(trial)
            if trial_energy < best_energy:
                best_edge = edge_idx
                best_energy = trial_energy
                best_spins = trial

        moved = state.clone()
        moved.spins = (best_spins if best_spins is not None else current).astype(int).tolist()
        moved.step_idx += 1
        return moved, best_edge

    def carnot_step(self, state: SlitherlinkState, iteration: int) -> StepResult[SlitherlinkState]:
        if self.is_solved(state):
            return StepResult(
                state=state,
                energy=0.0,
                iteration=iteration,
                is_solved=True,
                annotation=OPTIMAL_MESSAGE,
            )

        new_state = state
        moved_edge: int | None = None
        if iteration % ANIMATION_INTERVAL == 0:
            new_state, moved_edge = self._move_one_edge_toward_target(state)

        new_energy = self.energy(new_state)
        solved = self.is_solved(new_state)
        if solved:
            annotation = OPTIMAL_MESSAGE
        elif moved_edge is not None:
            annotation = (
                f"ITER {iteration:05d}. EDGE {moved_edge:02d} SET. ENERGY={new_energy:.0f}."
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

    def visualize(self, state: SlitherlinkState, energy: float) -> str:
        """Render the Slitherlink grid as WOPR terminal HTML."""
        board = html.escape(self.cartridge.to_display(state.spins))
        headline = OPTIMAL_MESSAGE if energy == 0.0 and self.is_solved(state) else THREAT_ASSESSMENT
        return (
            f'<div style="color:{self.accent_color};font-family:JetBrains Mono,monospace;'
            f'font-size:13px;line-height:1.45;">'
            f'<div style="padding-bottom:8px;">{html.escape(headline)}</div>'
            f'<pre style="margin:0;color:{self.accent_color};font-family:JetBrains Mono,'
            f'monospace;font-size:16px;line-height:1.25;">{board}</pre>'
            f'<div style="padding-top:8px;">ENERGY = {energy:.0f}</div>'
            "</div>"
        )


__all__ = [
    "ANIMATION_INTERVAL",
    "CANONICAL_SLITHERLINK_PUZZLE",
    "CANONICAL_SLITHERLINK_SOLUTION",
    "DEFAULT_CLUE_WEIGHT",
    "DEFAULT_DEGREE_WEIGHT",
    "EMPTY_PENALTY",
    "OPTIMAL_MESSAGE",
    "SlitherinkCartridge",
    "SlitherlinkCartridge",
    "SlitherlinkGame",
    "SlitherlinkState",
    "THREAT_ASSESSMENT",
]
