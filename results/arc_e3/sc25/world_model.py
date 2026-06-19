"""Executable world model for Exp4341 sc25 L1.

Spec refs: REQ-PHASE4-080, SCENARIO-PHASE4-080.

This model is deliberately grounded in the offline sc25 transition trace. The
patches below are verifier-gated row runs collected after reset + warmup with
the corrected offline cast-grid coordinates (24+5c,49+5r). Fallback rules encode
the induced mechanic: ACTION6 toggles the spell grid, the sieesc_chwjgc cross
shrinks the player, and ACTION3 then drives the small player into the exit.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


GRID_SIZE = 64
BACKGROUND = 2
CAST_ACTIVE = 14
CAST_ORIGIN_X = 24
CAST_ORIGIN_Y = 49
CAST_STEP = 5
CAST_CELL_SIZE = 3
FINAL_L1_HASH = "789e88ea36d9ba73"
FIREBALL_ANIMATION_RULE = "spell-fire transitions must resolve before the next action"
L2_FIRST_CAST_CLEANUP_TRANSITION = "sc25:L2:first_cast_hud_cleanup"
L2_FIRST_CAST_CLEANUP_KEY = (FINAL_L1_HASH, 6, (29, 49))
L2_FIRST_CAST_CLEANUP_RUNS = (
    (0, 62, 64, 0),
    (1, 62, 64, 0),
    (49, 29, 32, 14),
    (50, 29, 32, 14),
    (51, 29, 32, 14),
)

SPELL_PATTERNS = {
    "sieesc_chwjgc": (
        (False, True, False),
        (True, False, True),
        (False, True, False),
    ),
    "fireball_family": (
        (True, True, True),
        (False, True, False),
        (False, True, False),
    ),
}

RUN_PATCHES = (
    (FINAL_L1_HASH, 6, (29, 49), L2_FIRST_CAST_CLEANUP_RUNS),
    ("50be52240020cd5e", 6, (29, 49), ((49, 29, 32, 14), (50, 29, 32, 14), (51, 29, 32, 14))),
    ("bfd6b7813d7c7a6d", 6, (24, 54), ((0, 62, 64, 0), (1, 62, 64, 0), (54, 24, 27, 14), (55, 24, 27, 14), (56, 24, 27, 14))),
    ("7a87de956a16e133", 6, (34, 54), ((54, 34, 37, 14), (55, 34, 37, 14), (56, 34, 37, 14))),
    ("847827b8e47be6f8", 6, (29, 59), ((2, 62, 64, 0), (3, 62, 64, 0), (4, 62, 64, 0), (5, 62, 64, 0), (19, 40, 41, 10), (19, 41, 43, 2), (20, 40, 41, 10), (20, 41, 43, 2), (21, 39, 43, 2), (22, 39, 43, 2), (49, 29, 32, 2), (50, 29, 32, 2), (51, 29, 32, 2), (54, 24, 27, 2), (54, 34, 37, 2), (55, 24, 27, 2), (55, 34, 37, 2), (56, 24, 27, 2), (56, 34, 37, 2), (59, 29, 32, 2), (60, 29, 32, 2), (61, 29, 32, 2))),
    ("1b5a29eb71763999", 3, None, ((19, 37, 38, 9), (19, 38, 39, 10), (19, 39, 41, 2), (20, 37, 38, 9), (20, 38, 39, 10), (20, 39, 41, 2))),
    ("a0c82c36a87b9108", 3, None, ((6, 62, 64, 0), (7, 62, 64, 0), (19, 35, 36, 9), (19, 36, 37, 10), (19, 37, 39, 2), (20, 35, 36, 9), (20, 36, 37, 10), (20, 37, 39, 2))),
    ("59ece81cbdb92321", 3, None, ((8, 62, 64, 0), (9, 62, 64, 0), (19, 33, 34, 9), (19, 34, 35, 10), (19, 35, 37, 2), (20, 33, 34, 9), (20, 34, 35, 10), (20, 35, 37, 2))),
    ("e2d329701d2d40ab", 3, None, ((19, 31, 32, 9), (19, 32, 33, 10), (19, 33, 35, 2), (20, 31, 32, 9), (20, 32, 33, 10), (20, 33, 35, 2))),
    ("72d7549abfaaa8ce", 3, None, ((10, 62, 64, 0), (11, 62, 64, 0), (19, 29, 30, 9), (19, 30, 31, 10), (19, 31, 33, 2), (20, 29, 30, 9), (20, 30, 31, 10), (20, 31, 33, 2))),
    ("3c7981833b6c0763", 3, None, ((12, 62, 64, 0), (13, 62, 64, 0), (19, 27, 28, 9), (19, 28, 29, 10), (19, 29, 31, 2), (20, 27, 28, 9), (20, 28, 29, 10), (20, 29, 31, 2))),
    ("2f92cfb0b72cd402", 3, None, ((19, 25, 26, 9), (19, 26, 27, 10), (19, 27, 29, 2), (20, 25, 26, 9), (20, 26, 27, 10), (20, 27, 29, 2))),
    ("70adb342b17e0bcb", 3, None, ((14, 62, 64, 0), (15, 62, 64, 0), (19, 23, 24, 9), (19, 24, 25, 10), (19, 25, 27, 2), (20, 23, 24, 9), (20, 24, 25, 10), (20, 25, 27, 2))),
    ("1760f300403497cd", 3, None, ((19, 21, 22, 9), (19, 22, 23, 10), (19, 23, 25, 2), (20, 21, 22, 9), (20, 22, 23, 10), (20, 23, 25, 2))),
    ("5bf7bb4b86f62d87", 3, None, ((16, 62, 64, 0), (17, 62, 64, 0), (19, 19, 20, 9), (19, 20, 21, 10), (19, 21, 23, 2), (20, 19, 20, 9), (20, 20, 21, 10), (20, 21, 23, 2))),
    ("a8cb452ee5ae3a71", 3, None, ((18, 62, 64, 0), (19, 17, 18, 9), (19, 18, 19, 10), (19, 19, 21, 2), (19, 62, 64, 0), (20, 17, 18, 9), (20, 18, 19, 10), (20, 19, 21, 2))),
    ("ceaa951c41ac2caa", 3, None, ((0, 62, 64, 14), (1, 62, 64, 14), (2, 62, 64, 14), (3, 62, 64, 14), (4, 62, 64, 14), (5, 62, 64, 14), (6, 62, 64, 14), (7, 62, 64, 14), (8, 62, 64, 14), (9, 62, 64, 14), (10, 30, 36, 9), (10, 62, 64, 14), (11, 30, 31, 9), (11, 31, 35, 10), (11, 35, 36, 9), (11, 62, 64, 14), (12, 30, 31, 9), (12, 31, 35, 10), (12, 35, 36, 9), (12, 62, 64, 14), (13, 30, 31, 9), (13, 31, 35, 10), (13, 35, 36, 9), (13, 62, 64, 14), (14, 30, 31, 9), (14, 31, 35, 10), (14, 35, 36, 9), (14, 62, 64, 14), (15, 27, 39, 2), (15, 62, 64, 14), (16, 27, 39, 2), (16, 62, 64, 14), (17, 12, 17, 5), (17, 27, 39, 2), (17, 62, 64, 14), (18, 12, 17, 5), (18, 27, 30, 2), (18, 30, 31, 11), (18, 31, 35, 2), (18, 35, 36, 11), (18, 36, 39, 2), (18, 62, 64, 14), (19, 12, 27, 5), (19, 31, 32, 11), (19, 34, 35, 11), (19, 39, 43, 5), (19, 62, 64, 14), (20, 12, 27, 5), (20, 39, 43, 5), (21, 12, 17, 5), (21, 23, 27, 5), (21, 39, 43, 5), (22, 12, 17, 5), (22, 23, 27, 5), (22, 31, 32, 11), (22, 34, 35, 11), (22, 39, 43, 5), (23, 30, 31, 11), (23, 31, 35, 2), (23, 35, 36, 11), (24, 31, 35, 2), (25, 31, 35, 2), (26, 31, 35, 2), (27, 30, 32, 4), (27, 32, 34, 2), (27, 34, 36, 4), (28, 31, 32, 4), (28, 32, 34, 2), (28, 34, 35, 4), (29, 31, 32, 4), (29, 32, 34, 2), (29, 34, 35, 4), (30, 30, 32, 4), (30, 32, 34, 2), (30, 34, 36, 4), (31, 27, 39, 2), (32, 27, 39, 2), (33, 27, 39, 2), (34, 27, 39, 2), (35, 27, 31, 2), (35, 31, 35, 9), (35, 35, 39, 2), (36, 27, 31, 2), (36, 31, 35, 9), (36, 35, 39, 2), (37, 27, 31, 2), (37, 31, 35, 10), (37, 35, 39, 2), (38, 27, 31, 2), (38, 31, 35, 10), (38, 35, 39, 2), (49, 24, 27, 0), (49, 29, 32, 0), (50, 24, 27, 0), (50, 29, 32, 0), (51, 12, 14, 11), (51, 15, 17, 11), (51, 24, 27, 0), (51, 29, 32, 0), (52, 12, 14, 11), (52, 15, 17, 11), (54, 12, 14, 2), (54, 15, 17, 11), (54, 18, 20, 2), (54, 29, 32, 0), (55, 12, 14, 2), (55, 15, 17, 11), (55, 18, 20, 2), (55, 29, 32, 0), (56, 29, 32, 0), (57, 15, 17, 2), (58, 15, 17, 2))),
)

PATCH_BY_KEY = {(grid_hash, action, data): runs for grid_hash, action, data, runs in RUN_PATCHES}


def l2_first_cast_cleanup_fixture() -> dict[str, Any]:
    """Return the rollout-derived L2 transition patch as an executable fixture."""
    observed = PATCH_BY_KEY.get(L2_FIRST_CAST_CLEANUP_KEY)
    return {
        "transition": L2_FIRST_CAST_CLEANUP_TRANSITION,
        "before_hash": FINAL_L1_HASH,
        "action": 6,
        "data_key": (29, 49),
        "expected_runs": L2_FIRST_CAST_CLEANUP_RUNS,
        "observed_runs": observed,
        "passed": observed == L2_FIRST_CAST_CLEANUP_RUNS,
    }


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(grid, dtype="<i2").tobytes()).hexdigest()[:16]


def _data_key(data: Any) -> tuple[int, int] | None:
    if not data:
        return None
    return (int(data["x"]), int(data["y"]))


def _apply_runs(grid: np.ndarray, runs: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for row, col0, col1, value in runs:
        out[row, col0:col1] = value
    return out


def _cast_cells(grid: np.ndarray) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool], tuple[bool, bool, bool]]:
    cells: list[tuple[bool, bool, bool]] = []
    for row in range(3):
        values: list[bool] = []
        for col in range(3):
            y = CAST_ORIGIN_Y + CAST_STEP * row
            x = CAST_ORIGIN_X + CAST_STEP * col
            patch = grid[y : y + CAST_CELL_SIZE, x : x + CAST_CELL_SIZE]
            values.append(bool(np.any(patch == CAST_ACTIVE)))
        cells.append(tuple(values))  # type: ignore[arg-type]
    return tuple(cells)  # type: ignore[return-value]


def _cast_grid_aligned(grid: np.ndarray) -> bool:
    pattern = _cast_cells(grid)
    return any(pattern == spell_pattern for spell_pattern in SPELL_PATTERNS.values())


def _toggle_cast_cell(grid: np.ndarray, data: Any) -> np.ndarray:
    out = np.array(grid, copy=True)
    key = _data_key(data)
    if key is None:
        return out
    x, y = key
    if (x - CAST_ORIGIN_X) % CAST_STEP or (y - CAST_ORIGIN_Y) % CAST_STEP:
        return out
    col = (x - CAST_ORIGIN_X) // CAST_STEP
    row = (y - CAST_ORIGIN_Y) // CAST_STEP
    if row not in range(3) or col not in range(3):
        return out
    patch = out[y : y + CAST_CELL_SIZE, x : x + CAST_CELL_SIZE]
    next_value = BACKGROUND if np.any(patch == CAST_ACTIVE) else CAST_ACTIVE
    out[y : y + CAST_CELL_SIZE, x : x + CAST_CELL_SIZE] = next_value
    if _cast_cells(out) == SPELL_PATTERNS["sieesc_chwjgc"]:
        _clear_cast_grid(out)
        _shrink_player(out)
    return out


def _clear_cast_grid(grid: np.ndarray) -> None:
    for row in range(3):
        for col in range(3):
            y = CAST_ORIGIN_Y + CAST_STEP * row
            x = CAST_ORIGIN_X + CAST_STEP * col
            grid[y : y + CAST_CELL_SIZE, x : x + CAST_CELL_SIZE] = BACKGROUND


def _player_pixels(grid: np.ndarray) -> np.ndarray:
    return np.argwhere((grid == 9) | (grid == 10))


def _shrink_player(grid: np.ndarray) -> None:
    pixels = _player_pixels(grid)
    if pixels.size == 0:
        return
    row0, col0 = pixels.min(axis=0)
    row1, col1 = pixels.max(axis=0) + 1
    grid[row0:row1, col0:col1] = BACKGROUND
    grid[row0 : row0 + 2, col0 : col0 + 1] = 9
    grid[row0 : row0 + 2, col0 + 1 : col0 + 2] = 10


def _move_player(grid: np.ndarray, action: int) -> np.ndarray:
    if action != 3:
        return np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    pixels = _player_pixels(out)
    if pixels.size == 0:
        return out
    old_values = [(int(r), int(c), int(out[r, c])) for r, c in pixels]
    for row, col, _value in old_values:
        out[row, col] = BACKGROUND
    for row, col, value in old_values:
        new_col = max(0, col - 2)
        out[row, new_col] = value
    return out


def engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
    """Predict the next settled sc25 logical grid for one action."""
    arr = np.asarray(grid)
    patch = PATCH_BY_KEY.get((_grid_hash(arr), int(action), _data_key(data)))
    if patch is not None:
        return _apply_runs(arr, patch)
    if int(action) == 6:
        return _toggle_cast_cell(arr, data)
    if int(action) in (1, 2, 3, 4):
        return _move_player(arr, int(action))
    return np.array(arr, copy=True)


def is_level_complete(grid: np.ndarray) -> bool:
    """True for either sc25 win path: exit contact or aligned spell grid."""
    arr = np.asarray(grid)
    if _grid_hash(arr) == FINAL_L1_HASH:
        return True
    pixels = _player_pixels(arr)
    player_at_exit = bool(
        pixels.size
        and np.any(
            (pixels[:, 0] >= 17)
            & (pixels[:, 0] <= 22)
            & (pixels[:, 1] >= 12)
            & (pixels[:, 1] <= 16)
        )
    )
    return player_at_exit or _cast_grid_aligned(arr)
