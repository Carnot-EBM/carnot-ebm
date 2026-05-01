"""Hashi cartridge -- bridge-count CSP as an Ising-style spin model.

Each possible bridge between adjacent visible islands is one spin. Spin +1
means the bridge is present; spin -1 means absent. The energy is zero only
when island bridge counts match the clues, no orthogonal bridges cross, and
all islands belong to one connected component.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from games._base import StepResult, WOPRGame

HashiGrid = Sequence[Sequence[int]]

CANONICAL_HASHI_PUZZLE: tuple[tuple[int, ...], ...] = (
    (2, 0, 3, 0, 2),
    (0, 0, 0, 0, 0),
    (3, 0, 4, 0, 3),
    (0, 0, 0, 0, 0),
    (2, 0, 3, 0, 2),
)

DEFAULT_DEGREE_WEIGHT = 1.0
DEFAULT_CROSSING_WEIGHT = 1.0
DEFAULT_CONNECTIVITY_WEIGHT = 1.0
ANIMATION_INTERVAL = 200

THREAT_ASSESSMENT = "THREAT ASSESSMENT: HASHI BRIDGE NETWORK -- SOLVING PLANAR CSP..."
OPTIMAL_MESSAGE = "HASHI NETWORK CONNECTED. ENERGY = 0. ALL BRIDGE COUNTS VERIFIED."


@dataclass(frozen=True)
class HashiIsland:
    """One numbered island in the Hashi grid."""

    index: int
    row: int
    col: int
    target: int


@dataclass(frozen=True)
class HashiEdge:
    """One possible bridge variable between adjacent visible islands."""

    index: int
    a: int
    b: int
    orientation: str
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HashiModel:
    """Parsed Hashi puzzle with bridge variables and crossing constraints."""

    height: int
    width: int
    islands: tuple[HashiIsland, ...]
    edges: tuple[HashiEdge, ...]
    incident_edges: tuple[tuple[int, ...], ...]
    crossing_pairs: tuple[tuple[int, int], ...]

    @property
    def n_spins(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class HashiSolution:
    """Solved bridge assignment for a Hashi model."""

    model: HashiModel
    spins: tuple[int, ...]

    @property
    def present_edges(self) -> tuple[HashiEdge, ...]:
        binary = spins_to_binary(self.spins)
        return tuple(edge for edge in self.model.edges if binary[edge.index] == 1)


@dataclass
class HashiState:
    """Animated WOPR state: current spins plus the zero-energy target spins."""

    spins: list[int]
    target_spins: list[int] = field(default_factory=list)
    step_idx: int = 0

    def clone(self) -> HashiState:
        return HashiState(
            spins=list(self.spins),
            target_spins=list(self.target_spins),
            step_idx=self.step_idx,
        )


def _validate_grid(puzzle: HashiGrid) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(cell) for cell in row) for row in puzzle)
    if not rows:
        raise ValueError("Hashi puzzle must contain at least one row")
    width = len(rows[0])
    if width == 0:
        raise ValueError("Hashi puzzle rows must contain at least one cell")
    if any(len(row) != width for row in rows):
        raise ValueError("Hashi puzzle must be rectangular")
    for row in rows:
        for value in row:
            if value < 0 or value > 8:
                raise ValueError("Hashi island values must be in the range 0..8")
    return rows


def build_hashi_model(puzzle: HashiGrid) -> HashiModel:
    """Parse a grid into islands, bridge variables, and crossing pairs."""
    grid = _validate_grid(puzzle)
    height = len(grid)
    width = len(grid[0])

    islands: list[HashiIsland] = []
    island_at: dict[tuple[int, int], int] = {}
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            if value <= 0:
                continue
            index = len(islands)
            islands.append(HashiIsland(index=index, row=row, col=col, target=value))
            island_at[(row, col)] = index

    if not islands:
        raise ValueError("Hashi puzzle must contain at least one island")

    edges: list[HashiEdge] = []

    for row in range(height):
        row_islands = sorted(
            (col, island_at[(row, col)]) for col in range(width) if (row, col) in island_at
        )
        for (col_a, island_a), (col_b, island_b) in zip(row_islands, row_islands[1:], strict=False):
            cells = tuple((row, col) for col in range(col_a + 1, col_b))
            edges.append(
                HashiEdge(
                    index=len(edges),
                    a=island_a,
                    b=island_b,
                    orientation="H",
                    cells=cells,
                )
            )

    for col in range(width):
        col_islands = sorted(
            (row, island_at[(row, col)]) for row in range(height) if (row, col) in island_at
        )
        for (row_a, island_a), (row_b, island_b) in zip(col_islands, col_islands[1:], strict=False):
            cells = tuple((row, col) for row in range(row_a + 1, row_b))
            edges.append(
                HashiEdge(
                    index=len(edges),
                    a=island_a,
                    b=island_b,
                    orientation="V",
                    cells=cells,
                )
            )

    incident: list[list[int]] = [[] for _ in islands]
    for edge in edges:
        incident[edge.a].append(edge.index)
        incident[edge.b].append(edge.index)

    crossing_pairs: list[tuple[int, int]] = []
    for edge_a in edges:
        cells_a = set(edge_a.cells)
        if not cells_a:
            continue
        for edge_b in edges[edge_a.index + 1 :]:
            if edge_a.orientation == edge_b.orientation:
                continue
            if cells_a.intersection(edge_b.cells):
                crossing_pairs.append((edge_a.index, edge_b.index))

    return HashiModel(
        height=height,
        width=width,
        islands=tuple(islands),
        edges=tuple(edges),
        incident_edges=tuple(tuple(edge_ids) for edge_ids in incident),
        crossing_pairs=tuple(crossing_pairs),
    )


def spins_to_binary(spins: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert Ising spins (+1/-1) into bridge-presence bits (1/0)."""
    return (np.asarray(spins, dtype=np.int8).reshape(-1) > 0).astype(np.int8)


def binary_to_spins(binary: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert bridge-presence bits (1/0) into Ising spins (+1/-1)."""
    bits = np.asarray(binary, dtype=np.int8).reshape(-1)
    return np.where(bits > 0, 1, -1).astype(np.int8)


def connected_components(spins: Sequence[int] | np.ndarray, model: HashiModel) -> int:
    """Return the number of island components induced by present bridges."""
    if len(model.islands) <= 1:
        return len(model.islands)

    binary = spins_to_binary(spins)
    adjacency: list[list[int]] = [[] for _ in model.islands]
    for edge in model.edges:
        if binary[edge.index] == 0:
            continue
        adjacency[edge.a].append(edge.b)
        adjacency[edge.b].append(edge.a)

    seen: set[int] = set()
    n_components = 0
    for island in range(len(model.islands)):
        if island in seen:
            continue
        n_components += 1
        stack = [island]
        seen.add(island)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
    return n_components


def hashi_energy(
    spins: Sequence[int] | np.ndarray,
    model: HashiModel,
    degree_weight: float = DEFAULT_DEGREE_WEIGHT,
    crossing_weight: float = DEFAULT_CROSSING_WEIGHT,
    connectivity_weight: float = DEFAULT_CONNECTIVITY_WEIGHT,
) -> float:
    """Energy for a Hashi bridge assignment.

    Degree and crossing terms are the direct binary Ising/QUBO constraints.
    Connectivity is evaluated as a soft global penalty so E=0 still means a
    complete valid Hashi solution.
    """
    binary = spins_to_binary(spins)
    if len(binary) != model.n_spins:
        raise ValueError(f"Expected {model.n_spins} spins, got {len(binary)}")

    energy = 0.0
    for island in model.islands:
        count = int(sum(binary[edge_idx] for edge_idx in model.incident_edges[island.index]))
        energy += degree_weight * float((count - island.target) ** 2)

    for edge_a, edge_b in model.crossing_pairs:
        energy += crossing_weight * float(binary[edge_a] * binary[edge_b])

    if connectivity_weight and len(model.islands) > 1:
        n_components = connected_components(binary, model)
        energy += connectivity_weight * float((n_components - 1) ** 2)

    return 0.0 if abs(energy) < 1e-9 else float(energy)


def is_valid_hashi_solution(spins: Sequence[int] | np.ndarray, model: HashiModel) -> bool:
    """True when all Hashi bridge-count, crossing, and connectivity rules hold."""
    binary = spins_to_binary(spins)
    if len(binary) != model.n_spins:
        return False

    for island in model.islands:
        count = int(sum(binary[edge_idx] for edge_idx in model.incident_edges[island.index]))
        if count != island.target:
            return False

    for edge_a, edge_b in model.crossing_pairs:
        if binary[edge_a] and binary[edge_b]:
            return False

    return connected_components(binary, model) == 1


class HashiCartridge:
    """Exact Hashi solver for the WOPR cartridge API."""

    def __init__(
        self,
        degree_weight: float = DEFAULT_DEGREE_WEIGHT,
        crossing_weight: float = DEFAULT_CROSSING_WEIGHT,
        connectivity_weight: float = DEFAULT_CONNECTIVITY_WEIGHT,
    ) -> None:
        self.degree_weight = degree_weight
        self.crossing_weight = crossing_weight
        self.connectivity_weight = connectivity_weight

    def solve(self, puzzle: HashiGrid | None = None) -> tuple[HashiSolution, float, int]:
        """Solve a Hashi puzzle and return (solution, final_energy, iterations)."""
        model = build_hashi_model(CANONICAL_HASHI_PUZZLE if puzzle is None else puzzle)
        spins, final_energy, n_iterations = self._search(model)
        solution = HashiSolution(model=model, spins=tuple(int(spin) for spin in spins))
        return solution, final_energy, n_iterations

    def _search(self, model: HashiModel) -> tuple[np.ndarray, float, int]:
        n_edges = model.n_spins
        assignment = np.full(n_edges, -1, dtype=np.int8)
        counts = [0 for _ in model.islands]
        remaining = [len(edge_ids) for edge_ids in model.incident_edges]
        crossings_by_edge: list[set[int]] = [set() for _ in model.edges]
        for edge_a, edge_b in model.crossing_pairs:
            crossings_by_edge[edge_a].add(edge_b)
            crossings_by_edge[edge_b].add(edge_a)

        best_spins = binary_to_spins(np.zeros(n_edges, dtype=np.int8))
        best_energy = hashi_energy(
            best_spins,
            model,
            self.degree_weight,
            self.crossing_weight,
            self.connectivity_weight,
        )
        iterations = 0

        def feasible() -> bool:
            for island in model.islands:
                idx = island.index
                if counts[idx] > island.target:
                    return False
                if counts[idx] + remaining[idx] < island.target:
                    return False
            return True

        def recurse(edge_pos: int) -> np.ndarray | None:
            nonlocal best_energy, best_spins, iterations
            if edge_pos == n_edges:
                iterations += 1
                spins = binary_to_spins(assignment)
                energy = hashi_energy(
                    spins,
                    model,
                    self.degree_weight,
                    self.crossing_weight,
                    self.connectivity_weight,
                )
                if energy < best_energy:
                    best_energy = energy
                    best_spins = spins.copy()
                if energy == 0.0 and is_valid_hashi_solution(spins, model):
                    return spins
                return None

            edge = model.edges[edge_pos]
            for present in (1, 0):
                has_crossing = any(
                    assignment[crossing] == 1 for crossing in crossings_by_edge[edge.index]
                )
                if present and has_crossing:
                    continue

                assignment[edge.index] = present
                for island_idx in (edge.a, edge.b):
                    remaining[island_idx] -= 1
                    counts[island_idx] += present

                solved = recurse(edge_pos + 1) if feasible() else None

                for island_idx in (edge.a, edge.b):
                    counts[island_idx] -= present
                    remaining[island_idx] += 1
                assignment[edge.index] = -1

                if solved is not None:
                    return solved
            return None

        solution = recurse(0)
        if solution is not None:
            return solution, 0.0, max(iterations, 1)
        return best_spins, float(best_energy), max(iterations, 1)


def render_hashi_ascii(model: HashiModel, spins: Sequence[int] | np.ndarray) -> str:
    """Render islands and present bridges as compact ASCII art."""
    chars = [["." for _ in range(model.width)] for _ in range(model.height)]
    for island in model.islands:
        chars[island.row][island.col] = str(island.target)

    binary = spins_to_binary(spins)
    for edge in model.edges:
        if binary[edge.index] == 0:
            continue
        bridge_char = "-" if edge.orientation == "H" else "|"
        for row, col in edge.cells:
            chars[row][col] = bridge_char if chars[row][col] == "." else "X"

    return "\n".join("".join(row) for row in chars)


class HashiGame(WOPRGame[HashiState, int]):
    """WOPR cartridge for Hashiwokakero bridge-count puzzles."""

    name = "HASHI"
    description = "5x5 BRIDGE CSP. ONE SPIN PER POSSIBLE BRIDGE. ENERGY=RULE VIOLATIONS."
    accent_color = "#39ff14"

    def __init__(self, puzzle: HashiGrid | None = None) -> None:
        self.puzzle = CANONICAL_HASHI_PUZZLE if puzzle is None else puzzle
        self.cartridge = HashiCartridge()
        self.solution, self.solution_energy, self.solve_iterations = self.cartridge.solve(
            self.puzzle
        )
        self.model = self.solution.model

    def initial_state(self) -> HashiState:
        return HashiState(
            spins=[-1 for _ in range(self.model.n_spins)],
            target_spins=list(self.solution.spins),
        )

    def energy(self, state: HashiState) -> float:
        return hashi_energy(state.spins, self.model)

    def is_solved(self, state: HashiState) -> bool:
        return self.energy(state) == 0.0 and is_valid_hashi_solution(state.spins, self.model)

    def _move_one_bridge_toward_target(self, state: HashiState) -> tuple[HashiState, int | None]:
        current = np.asarray(state.spins, dtype=np.int8)
        target = np.asarray(state.target_spins, dtype=np.int8)
        mismatched = [idx for idx in range(self.model.n_spins) if current[idx] != target[idx]]
        if not mismatched:
            return state, None

        best_edge = mismatched[0]
        best_spins: np.ndarray | None = None
        best_energy = float("inf")
        for edge_idx in mismatched:
            trial = current.copy()
            trial[edge_idx] = target[edge_idx]
            trial_energy = hashi_energy(trial, self.model)
            if trial_energy < best_energy:
                best_edge = edge_idx
                best_energy = trial_energy
                best_spins = trial

        moved = state.clone()
        moved.spins = (best_spins if best_spins is not None else current).astype(int).tolist()
        moved.step_idx += 1
        return moved, best_edge

    def carnot_step(self, state: HashiState, iteration: int) -> StepResult[HashiState]:
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
            new_state, moved_edge = self._move_one_bridge_toward_target(state)

        new_energy = self.energy(new_state)
        solved = self.is_solved(new_state)
        if solved:
            annotation = OPTIMAL_MESSAGE
        elif moved_edge is not None:
            edge = self.model.edges[moved_edge]
            a = self.model.islands[edge.a]
            b = self.model.islands[edge.b]
            annotation = (
                f"ITER {iteration:05d}. BRIDGE {moved_edge:02d} "
                f"({a.row},{a.col})-({b.row},{b.col}) SET. ENERGY={new_energy:.0f}."
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

    def visualize(self, state: HashiState, energy: float) -> str:
        """Render the Hashi grid as WOPR-green terminal HTML."""
        board = html.escape(render_hashi_ascii(self.model, state.spins))
        headline = OPTIMAL_MESSAGE if energy == 0.0 and self.is_solved(state) else THREAT_ASSESSMENT
        return (
            f'<div style="color:{self.accent_color};font-family:JetBrains Mono,monospace;'
            f'font-size:13px;line-height:1.45;">'
            f'<div style="padding-bottom:8px;">{html.escape(headline)}</div>'
            f'<pre style="margin:0;color:{self.accent_color};font-family:JetBrains Mono,'
            f'monospace;font-size:20px;line-height:1.25;">{board}</pre>'
            f'<div style="padding-top:8px;">ENERGY = {energy:.0f}</div>'
            "</div>"
        )


__all__ = [
    "ANIMATION_INTERVAL",
    "CANONICAL_HASHI_PUZZLE",
    "DEFAULT_CONNECTIVITY_WEIGHT",
    "DEFAULT_CROSSING_WEIGHT",
    "DEFAULT_DEGREE_WEIGHT",
    "HashiCartridge",
    "HashiEdge",
    "HashiGame",
    "HashiIsland",
    "HashiModel",
    "HashiSolution",
    "HashiState",
    "OPTIMAL_MESSAGE",
    "THREAT_ASSESSMENT",
    "binary_to_spins",
    "build_hashi_model",
    "connected_components",
    "hashi_energy",
    "is_valid_hashi_solution",
    "render_hashi_ascii",
    "spins_to_binary",
]
