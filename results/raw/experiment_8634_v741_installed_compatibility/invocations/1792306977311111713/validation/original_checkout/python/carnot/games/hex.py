"""Hex cartridge for WOPR energy-player experiments.

Spec: REQ-HEX-001, REQ-HEX-002, REQ-HEX-003
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

import numpy as np

from carnot.samplers.phase4_sampler import Phase4Sampler

Action = tuple[int, int]


class HexBoard:
    """Small value wrapper for an `n x n` Hex board."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(
        self,
        n: int,
        cells: Sequence[Sequence[int]] | np.ndarray | None = None,
    ) -> None:
        size = int(n)
        if size <= 0:
            raise ValueError("Hex board size must be positive")
        self.n = size
        if cells is None:
            self.cells = np.zeros((self.n, self.n), dtype=np.int8)
        else:
            board = np.asarray(cells, dtype=np.int8)
            if board.shape != (self.n, self.n):
                raise ValueError(f"Expected board shape ({self.n}, {self.n}), got {board.shape}")
            if np.any((board < self.EMPTY) | (board > self.WHITE)):
                raise ValueError("Hex board cells must be 0, 1, or 2")
            self.cells = board.copy()

    def copy(self) -> "HexBoard":
        """Return a detached copy of this board wrapper."""
        return HexBoard(self.n, self.cells)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """Let NumPy consumers view the wrapped cell array."""
        return np.asarray(self.cells, dtype=dtype)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def connected(self, left: int, right: int) -> bool:
        return self.find(left) == self.find(right)


class HexGame:
    """Rules and energy helper for a standard two-player Hex board."""

    EMPTY = HexBoard.EMPTY
    BLACK = HexBoard.BLACK
    WHITE = HexBoard.WHITE
    NEIGHBORS: tuple[Action, ...] = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

    def __init__(self, n: int = 7) -> None:
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("Hex game size must be positive")

    def reset(self) -> np.ndarray:
        """Return a fresh empty board."""
        return HexBoard(self.n).cells

    def legal_actions(self, board: Sequence[Sequence[int]] | np.ndarray | HexBoard) -> list[Action]:
        """Return every empty cell as a row-major `(row, col)` action."""
        cells = self._board_array(board)
        return [
            (row, col)
            for row in range(self.n)
            for col in range(self.n)
            if int(cells[row, col]) == self.EMPTY
        ]

    def step(
        self,
        board: Sequence[Sequence[int]] | np.ndarray | HexBoard,
        action: Action,
        player: int,
    ) -> tuple[np.ndarray, bool, int | None]:
        """Apply one legal move and return `(new_board, done, winner)`."""
        self._validate_player(player)
        row, col = self._validate_action(action)
        cells = self._board_array(board).copy()
        if int(cells[row, col]) != self.EMPTY:
            raise ValueError(f"Hex action {action} targets an occupied cell")

        cells[row, col] = int(player)
        winner = self.check_winner(cells)
        done = winner is not None or not self.legal_actions(cells)
        return cells, done, winner

    def check_winner(self, board: Sequence[Sequence[int]] | np.ndarray | HexBoard) -> int | None:
        """Return None, 1 for Black, or 2 for White using union-find connectivity."""
        cells = self._board_array(board)
        if self._player_connected(cells, self.BLACK):
            return self.BLACK
        if self._player_connected(cells, self.WHITE):
            return self.WHITE
        return None

    def path_strength(
        self,
        board: Sequence[Sequence[int]] | np.ndarray | HexBoard,
        player: int,
    ) -> int:
        """Return the longest connected component span toward `player`'s goal."""
        self._validate_player(player)
        cells = self._board_array(board)
        visited: set[Action] = set()
        best = 0
        for row in range(self.n):
            for col in range(self.n):
                if int(cells[row, col]) != player or (row, col) in visited:
                    continue
                component = self._component(cells, (row, col), player, visited)
                rows = [cell_row for cell_row, _ in component]
                cols = [cell_col for _, cell_col in component]
                if player == self.BLACK:
                    best = max(best, max(rows) - min(rows) + 1)
                else:
                    best = max(best, max(cols) - min(cols) + 1)
        return best

    def energy(self, board: Sequence[Sequence[int]] | np.ndarray | HexBoard, player: int) -> float:
        """Return E(board) = -1 * longest path strength for `player`."""
        return -float(self.path_strength(board, player))

    def energy_after_action(
        self,
        board: Sequence[Sequence[int]] | np.ndarray | HexBoard,
        action: Action,
        player: int,
    ) -> float:
        """Return the current-player energy after applying a candidate action."""
        next_board, _, _ = self.step(board, action, player)
        return self.energy(next_board, player)

    def _board_array(self, board: Sequence[Sequence[int]] | np.ndarray | HexBoard) -> np.ndarray:
        cells = board.cells if isinstance(board, HexBoard) else np.asarray(board, dtype=np.int8)
        if cells.shape != (self.n, self.n):
            raise ValueError(f"Expected board shape ({self.n}, {self.n}), got {cells.shape}")
        if np.any((cells < self.EMPTY) | (cells > self.WHITE)):
            raise ValueError("Hex board cells must be 0, 1, or 2")
        return cells

    def _validate_player(self, player: int) -> None:
        if int(player) not in (self.BLACK, self.WHITE):
            raise ValueError("Hex player must be 1 (Black) or 2 (White)")

    def _validate_action(self, action: Action) -> Action:
        row, col = int(action[0]), int(action[1])
        if row < 0 or row >= self.n or col < 0 or col >= self.n:
            raise ValueError(f"Hex action {(row, col)} is outside the board")
        return row, col

    def _cell_index(self, row: int, col: int) -> int:
        return row * self.n + col

    def _player_connected(self, cells: np.ndarray, player: int) -> bool:
        top_or_left = self.n * self.n
        bottom_or_right = top_or_left + 1
        union_find = _UnionFind(self.n * self.n + 2)

        for row in range(self.n):
            for col in range(self.n):
                if int(cells[row, col]) != player:
                    continue
                cell_index = self._cell_index(row, col)
                if player == self.BLACK:
                    if row == 0:
                        union_find.union(cell_index, top_or_left)
                    if row == self.n - 1:
                        union_find.union(cell_index, bottom_or_right)
                else:
                    if col == 0:
                        union_find.union(cell_index, top_or_left)
                    if col == self.n - 1:
                        union_find.union(cell_index, bottom_or_right)

                for next_row, next_col in self._neighbors(row, col):
                    if int(cells[next_row, next_col]) == player:
                        union_find.union(cell_index, self._cell_index(next_row, next_col))

        return union_find.connected(top_or_left, bottom_or_right)

    def _component(
        self,
        cells: np.ndarray,
        start: Action,
        player: int,
        visited: set[Action],
    ) -> list[Action]:
        stack = [start]
        visited.add(start)
        component: list[Action] = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for next_row, next_col in self._neighbors(row, col):
                neighbor = (next_row, next_col)
                if neighbor not in visited and int(cells[next_row, next_col]) == player:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return component

    def _neighbors(self, row: int, col: int) -> list[Action]:
        neighbors: list[Action] = []
        for row_delta, col_delta in self.NEIGHBORS:
            next_row = row + row_delta
            next_col = col + col_delta
            if 0 <= next_row < self.n and 0 <= next_col < self.n:
                neighbors.append((next_row, next_col))
        return neighbors


@dataclass
class RandomPlayer:
    """Uniform legal-action player."""

    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select_action(self, game: HexGame, board: np.ndarray, player: int) -> Action:
        del player
        actions = game.legal_actions(board)
        if not actions:
            raise ValueError("RandomPlayer cannot move on a full board")
        return actions[int(self._rng.integers(0, len(actions)))]


@dataclass
class GreedyEnergyPlayer:
    """One-ply player that minimizes the current player's post-move energy."""

    def select_action(self, game: HexGame, board: np.ndarray, player: int) -> Action:
        actions = game.legal_actions(board)
        if not actions:
            raise ValueError("GreedyEnergyPlayer cannot move on a full board")
        return min(
            actions, key=lambda action: (game.energy_after_action(board, action, player), action)
        )


@dataclass
class GibbsEnergyPlayer:
    """Blocked-Gibbs player over candidate moves using k=5 composed free energy."""

    seed: int = 1188
    n_steps: int = 96
    k_verifiers: int = 5
    temperature: float = 0.35
    last_diagnostics: dict[str, float | None] = field(default_factory=dict, init=False)

    def select_action(self, game: HexGame, board: np.ndarray, player: int) -> Action:
        actions = game.legal_actions(board)
        if not actions:
            raise ValueError("GibbsEnergyPlayer cannot move on a full board")
        if len(actions) == 1:
            self.last_diagnostics = {"n_candidates": 1.0, "best_free_energy": None}
            return actions[0]

        free_energies = np.asarray(
            [self._free_energy(game, board, action, player) for action in actions],
            dtype=np.float64,
        )
        exact_best = int(np.argmin(free_energies))
        init_state = -np.ones(len(actions), dtype=np.float64)
        init_state[exact_best] = 1.0

        sampler = Phase4Sampler(
            algorithm="blocked_gibbs",
            seed=self.seed,
            temperature=self.temperature,
            discrete_indices=tuple(range(len(actions))),
            continuous_indices=(),
        )

        def latent_energy(state: np.ndarray) -> float:
            selected = int(np.argmax(state))
            return float(free_energies[selected])

        chain = sampler.sample(latent_energy, init_state, max(1, int(self.n_steps)))
        candidate_indices = {exact_best}
        candidate_indices.update(int(np.argmax(state)) for state in chain)
        best_index = min(
            candidate_indices, key=lambda index: (free_energies[index], actions[index])
        )
        self.last_diagnostics = {
            **sampler.last_diagnostics,
            "n_candidates": float(len(actions)),
            "best_free_energy": float(free_energies[best_index]),
        }
        return actions[best_index]

    def _free_energy(self, game: HexGame, board: np.ndarray, action: Action, player: int) -> float:
        return float(self.k_verifiers * game.energy_after_action(board, action, player))


__all__ = [
    "Action",
    "GibbsEnergyPlayer",
    "GreedyEnergyPlayer",
    "HexBoard",
    "HexGame",
    "RandomPlayer",
]
