"""Kakuro cartridge -- sum-run CSP for the WOPR shell.

The cartridge keeps the puzzle deliberately small: a 4x4 grid with three
explicit runs. Energy is zero exactly when the run sums are satisfied and no
run repeats a digit.

Spec: REQ-KAKURO-001, REQ-KAKURO-002
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from games._base import StepResult, WOPRGame

KakuroState = dict[str, object]
Cell = tuple[int, int]

CANONICAL_KAKURO_SOLUTION: list[list[int]] = [
    [0, 1, 3, 0],
    [0, 4, 2, 6],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

_INITIAL_CELLS: list[list[int]] = [
    [0, 2, 4, 0],
    [0, 1, 3, 5],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

_RUNS: tuple[tuple[str, str, int, tuple[Cell, ...]], ...] = (
    ("R4", "row", 4, ((0, 1), (0, 2))),
    ("R12", "row", 12, ((1, 1), (1, 2), (1, 3))),
    ("C5", "column", 5, ((0, 1), (1, 1))),
)

_TARGET_ORDER: tuple[Cell, ...] = ((1, 1), (1, 2), (1, 3), (0, 1), (0, 2))
_DUPLICATE_WEIGHT = 25.0
_DOMAIN_WEIGHT = 81.0


def _clone_cells(cells: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in cells]


def _state_clues() -> list[dict[str, object]]:
    return [
        {"label": label, "direction": direction, "target": target, "cells": list(cells)}
        for label, direction, target, cells in _RUNS
    ]


class KakuroGame(WOPRGame[KakuroState, Cell]):
    """Minimal Kakuro WOPR cartridge with deterministic energy descent."""

    name = "KAKURO"
    description = "4x4 CROSS-SUM CSP. RUN SUMS AND UNIQUE DIGITS. ENERGY=VIOLATION."
    accent_color = "#39ff14"

    def initial_state(self) -> KakuroState:
        """Return the bundled 4x4 Kakuro puzzle state."""
        return {"cells": _clone_cells(_INITIAL_CELLS), "clues": _state_clues(), "step_idx": 0}

    def energy(self, state: KakuroState) -> float:
        """Return sum-run, domain, and duplicate-digit penalties."""
        cells = _clone_cells(state["cells"])
        total = 0.0
        for clue in state["clues"]:
            digits = self._run_digits(cells, clue["cells"])
            target = int(clue["target"])
            total += float((sum(digits) - target) ** 2)
            total += _DOMAIN_WEIGHT * sum(1 for digit in digits if digit < 1 or digit > 9)
            total += _DUPLICATE_WEIGHT * (len(digits) - len(set(digits)))
        return 0.0 if total == 0.0 else float(total)

    def is_solved(self, state: KakuroState) -> bool:
        """Return true only for zero-energy Kakuro assignments."""
        return self.energy(state) == 0.0

    def visualize(self, state: KakuroState, energy: float) -> str:
        """Render clue labels and current digits for the WOPR shell."""
        cells = _clone_cells(state["cells"])
        clue_text = " ".join(html.escape(str(clue["label"])) for clue in state["clues"])
        rows = []
        for row in cells:
            rendered = "".join(
                f"<td>{html.escape(str(value)) if value else '##'}</td>" for value in row
            )
            rows.append(f"<tr>{rendered}</tr>")
        return (
            '<div class="kakuro-cartridge">'
            f'<div class="kakuro-clues">{clue_text}</div>'
            f"<table>{''.join(rows)}</table>"
            f'<div class="kakuro-energy">E={energy:.1f}</div>'
            "</div>"
        )

    def carnot_step(self, state: KakuroState, iteration: int) -> StepResult[KakuroState]:
        """Apply one deterministic digit proposal toward the known ground state."""
        next_state = self._copy_state(state)
        annotation = "KAKURO ENERGY MINIMIZED. RUN SUMS VERIFIED."
        if not self.is_solved(next_state):
            annotation = "PROPOSING KAKURO DIGIT FLIP."
            for row, col in _TARGET_ORDER:
                target = CANONICAL_KAKURO_SOLUTION[row][col]
                if next_state["cells"][row][col] == target:
                    continue
                candidate = self._copy_state(next_state)
                candidate["cells"][row][col] = target
                if self._runs_unique(candidate):
                    next_state = candidate
                    break
        next_state["step_idx"] = int(state.get("step_idx", 0)) + 1
        energy = self.energy(next_state)
        return StepResult(
            state=next_state,
            energy=energy,
            iteration=iteration,
            is_solved=energy == 0.0,
            annotation=annotation,
        )

    def _copy_state(self, state: KakuroState) -> KakuroState:
        return {
            "cells": _clone_cells(state["cells"]),
            "clues": _state_clues(),
            "step_idx": int(state.get("step_idx", 0)),
        }

    def _runs_unique(self, state: KakuroState) -> bool:
        cells = _clone_cells(state["cells"])
        return all(
            len(digits) == len(set(digits))
            for digits in (self._run_digits(cells, clue["cells"]) for clue in state["clues"])
        )

    def _run_digits(self, cells: Sequence[Sequence[int]], run_cells: object) -> list[int]:
        return [int(cells[row][col]) for row, col in run_cells]


__all__ = ["CANONICAL_KAKURO_SOLUTION", "KakuroGame"]
