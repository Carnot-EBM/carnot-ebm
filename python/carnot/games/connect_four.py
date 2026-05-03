"""Connect Four cartridge encoded as an Ising-style occupancy energy.

The cartridge uses one binary spin per board cell. Spin +1 means occupied and
spin -1 means empty. The energy is zero only when occupied cells obey gravity
and the occupied-cell count matches the configured initial piece count.

Spec: REQ-CONNECT4-001, REQ-CONNECT4-002, REQ-CONNECT4-003
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ConnectFourIsingCartridge:
    """6x7 Connect Four occupancy cartridge with gravity and count penalties."""

    BOARD_ROWS = 6
    BOARD_COLS = 7
    RED = 1
    YELLOW = 2

    def __init__(
        self,
        initial_board: Sequence[Sequence[int]] | np.ndarray | None = None,
        initial_pieces: int | None = None,
        gravity_penalty: float = 10.0,
        count_penalty: float = 1.0,
    ) -> None:
        self.n_spins = self.BOARD_ROWS * self.BOARD_COLS
        self.gravity_penalty = float(gravity_penalty)
        self.count_penalty = float(count_penalty)
        self.initial_board = (
            None if initial_board is None else self._board_array(initial_board).copy()
        )
        derived_pieces = 0
        if self.initial_board is not None:
            derived_pieces = int(self._occupancy(self.initial_board).sum())
        self.initial_pieces = int(derived_pieces if initial_pieces is None else initial_pieces)
        if self.initial_pieces < 0 or self.initial_pieces > self.n_spins:
            raise ValueError(f"initial_pieces must be in 0..{self.n_spins}")

    def _board_array(self, state: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        board = np.asarray(state)
        if board.shape == (self.n_spins,):
            return board.reshape(self.BOARD_ROWS, self.BOARD_COLS)
        if board.shape == (self.BOARD_ROWS, self.BOARD_COLS):
            return board
        raise ValueError(
            f"Expected shape ({self.n_spins},) or "
            f"({self.BOARD_ROWS}, {self.BOARD_COLS}), got {board.shape}"
        )

    def _occupancy(self, state: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        board = self._board_array(state)
        if np.any(board < 0):
            return (board > 0).astype(np.int8)
        return (board != 0).astype(np.int8)

    def _color_board(self, board: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        values = self._board_array(board)
        if np.any(values < 0):
            return np.where(values > 0, self.RED, 0).astype(np.int8)
        return np.where(values == self.YELLOW, self.YELLOW, np.where(values != 0, self.RED, 0))

    def _gravity_violations(self, occupied: np.ndarray) -> int:
        violations = 0
        for row in range(self.BOARD_ROWS - 1):
            for col in range(self.BOARD_COLS):
                if occupied[row, col] and not occupied[row + 1, col]:
                    violations += 1
        return violations

    def energy(self, state: Sequence[Sequence[int]] | np.ndarray) -> float:
        """Return the gravity plus piece-conservation penalty energy."""
        occupied = self._occupancy(state)
        gravity = self._gravity_violations(occupied)
        piece_delta = int(occupied.sum()) - self.initial_pieces
        energy = self.gravity_penalty * gravity + self.count_penalty * float(piece_delta**2)
        return 0.0 if abs(energy) < 1e-9 else float(energy)

    def is_valid(self, board: Sequence[Sequence[int]] | np.ndarray) -> bool:
        """Return True when gravity and configured piece count are satisfied."""
        occupied = self._occupancy(board)
        return (
            self._gravity_violations(occupied) == 0 and int(occupied.sum()) == self.initial_pieces
        )

    def _compact_board(self, board: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        values = self._board_array(board)
        occupied = self._occupancy(values)
        compacted = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=np.int8)
        for col in range(self.BOARD_COLS):
            tokens = [
                int(values[row, col])
                for row in range(self.BOARD_ROWS - 1, -1, -1)
                if occupied[row, col]
            ]
            for offset, token in enumerate(tokens):
                compacted[self.BOARD_ROWS - 1 - offset, col] = token if token > 0 else self.RED
        return compacted

    def sample(self, n_steps: int = 1000, beta: float = 2.0) -> np.ndarray:
        """Return a deterministic zero-energy board for the configured count."""
        del n_steps, beta
        if self.initial_board is not None:
            return self._compact_board(self.initial_board)

        board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=np.int8)
        remaining = self.initial_pieces
        for col in range(self.BOARD_COLS):
            for row in range(self.BOARD_ROWS - 1, -1, -1):
                if remaining <= 0:
                    return board
                board[row, col] = self.RED
                remaining -= 1
        return board

    def _has_four_from(self, colors: np.ndarray, row: int, col: int, dr: int, dc: int) -> bool:
        color = int(colors[row, col])
        for offset in range(1, 4):
            next_row = row + dr * offset
            next_col = col + dc * offset
            if (
                next_row < 0
                or next_row >= self.BOARD_ROWS
                or next_col < 0
                or next_col >= self.BOARD_COLS
                or int(colors[next_row, next_col]) != color
            ):
                return False
        return True

    def check_winner(self, board: Sequence[Sequence[int]] | np.ndarray) -> str:
        """Return RED, YELLOW, DRAW, or ONGOING for a Connect Four board."""
        colors = self._color_board(board)
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                color = int(colors[row, col])
                if color == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                    if self._has_four_from(colors, row, col, dr, dc):
                        return "RED" if color == self.RED else "YELLOW"

        if int((colors != 0).sum()) == self.n_spins:
            return "DRAW"
        return "ONGOING"


__all__ = ["ConnectFourIsingCartridge"]
