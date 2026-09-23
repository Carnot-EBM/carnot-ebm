"""Hand-written grid-only world model for vc33, level index 1 (source comment "# Level 2").

Uses ONLY (grid, action, data) plus constants derived by reading
environment_files/vc33/5430563c/vc33.py. It reads no files, calls no simulator, embeds no
recorded grid, and does no hash lookup. Unrecognised boards fall back to "HUD bar steps down,
nothing else changes", which is what any non-button click does in this game.

Derivation notes (source line numbers refer to vc33.py):
- levels[1] (lines 1572-1591): grid 32x32, StepCounter 50, Gravity [-2, 0], every sprite rotated 90.
  Camera scale = int(64/32) = 2, no letterbox (camera.py display_to_grid / render).
- HUD (xclqwacrmx.render_interface, lines 1814-1826): display row 0; the first
  round(64 * steps / StepCounter) cells are 7, the rest 4. step() decrements steps on every
  ACTION6 before anything else (line 2104).
- Vessels (tag 0043): 0045 at (0,0) -> rows 0-9; 0046 at (0,12) -> rows 12-19; 0047 at (0,22)
  -> rows 22-31. All pixels are 0, anchored at x=0, so a vessel renders as fill x in [0, width).
- Walls (tag 0025): 0026 at (0,10) is 2x28 -> rows 10-11, x 0-27; 0027 at (0,20) is 2x22.
- Buttons (tag 0022, layer 1): 2x2 of color 9 at x 0-1, rows {8,12,18,22}.
- Button -> (shrinking vessel, growing vessel), from on_set_level (lines 1894-1935) with gravity
  x<0 (qhmwbtpcsk False, so ldbtzvbmoj is always true here):
    btn 8  : no vessel starts at y=8; A spans it; vessel at y=8+2*2=12 is B -> (B, A)
    btn 12 : B starts at 12; vessel ending at 12-2=10 is A              -> (A, B)
    btn 18 : no vessel starts at 18; B spans it; vessel at 18+4=22 is C -> (C, B)
    btn 22 : C starts at 22; vessel ending at 22-2=20 is B              -> (B, C)
- Transfer (xowbpvmzbd, lines 2030-2052): only if width(shrink) > 0 and
  width(grow) < ysoqxdegud(grow) = min(x+width of walls touching it) (no floors in this level):
  A: 28 (wall 0026), B: min(28, 22) = 22, C: 22 (wall 0027). Moves 2 cells of fill.
  Floaters (tag 0016) sitting at a vessel's surface (x == width) move with it.
- Floater 0017 at (4,26), rendered clockwise-rotated: [[14,4,-1],[14,4,4],[14,4,-1]].
- Target 0011 at (14,20), layer 1, renders 14 at x=14, rows 20-21 (other pixels transparent).
- Win (ielczunthe): floater x == target x == 14. That row is a level-up; the verifier excludes it.
"""

import numpy as np

SCALE = 2
N = 32
STEP_COUNTER = 50
BG, FILL, WALL, BTN, TGT = 3, 0, 5, 9, 14
HUD_ON, HUD_OFF = 7, 4
LANES = {"A": (0, 10), "B": (12, 20), "C": (22, 32)}
MEASURE_ROW = {"A": 4, "B": 15, "C": 30}  # rows with no button and no floater
WALLS = (((10, 12), 28), ((20, 22), 22))
BUTTON_TOPS = (8, 12, 18, 22)
MAPPING = {8: ("B", "A"), 12: ("A", "B"), 18: ("C", "B"), 22: ("B", "C")}
LIMIT = {"A": 28, "B": 22, "C": 22}
FLOATER_Y = 26
FLOATER = ((14, 4, -1), (14, 4, 4), (14, 4, -1))
TARGET_X, TARGET_ROWS = 14, (20, 21)


def _bar_count(steps):
    return int(round(64 * steps / STEP_COUNTER))


def _parse_hud(grid):
    row = [int(v) for v in grid[0]]
    n7 = 0
    while n7 < 64 and row[n7] == HUD_ON:
        n7 += 1
    if any(v != HUD_OFF for v in row[n7:]):
        return None
    for steps in range(STEP_COUNTER + 1):
        if _bar_count(steps) == n7:
            return steps
    return None


def _hud_row(steps):
    n7 = _bar_count(steps)
    return np.array([HUD_ON if x < n7 else HUD_OFF for x in range(64)])


def _render(widths, fx):
    lv = np.full((N, N), BG, dtype=np.int64)
    # layer 0 in the level's sprite-list order: floater, walls, vessels
    for dy, prow in enumerate(FLOATER):
        for dx, v in enumerate(prow):
            x = fx + dx
            if v >= 0 and 0 <= x < N:
                lv[FLOATER_Y + dy, x] = v
    for (y0, y1), xw in WALLS:
        lv[y0:y1, 0:xw] = WALL
    for name, (y0, y1) in LANES.items():
        lv[y0:y1, 0 : widths[name]] = FILL
    # layer 1: target then buttons
    for y in TARGET_ROWS:
        lv[y, TARGET_X] = TGT
    for y in BUTTON_TOPS:
        lv[y : y + 2, 0:2] = BTN
    return np.repeat(np.repeat(lv, SCALE, axis=0), SCALE, axis=1)


def _parse_board(grid):
    lv = grid[1::2, ::2]  # display row 2y+1 avoids the HUD row
    widths = {}
    for name, y in MEASURE_ROW.items():
        w = 0
        while w < N and int(lv[y, w]) == FILL:
            w += 1
        widths[name] = w
    fx = widths["C"]  # the floater rides the surface of vessel C
    return widths, fx


def engine(grid, action, data):
    g = np.asarray(grid)
    out = g.copy()
    if g.shape != (64, 64):
        return out
    steps = _parse_hud(g)
    if steps is None:
        return out
    new_steps = max(0, steps - 1) if int(action) == 6 else steps
    widths, fx = _parse_board(g)
    board = _render(widths, fx)
    recognised = np.array_equal(board[1:], g[1:])
    if recognised and int(action) == 6 and isinstance(data, dict):
        px, py = int(data.get("x", 0)), int(data.get("y", 0))
        gx = px // SCALE if px >= 0 else -1
        gy = py // SCALE if py >= 0 else -1
        if 0 <= gx < 2 and 0 <= gy < N:
            top = next((t for t in BUTTON_TOPS if t <= gy < t + 2), None)
            if top is not None:
                shrink, grow = MAPPING[top]
                if widths[shrink] > 0 and widths[grow] < LIMIT[grow]:
                    widths[shrink] -= 2
                    widths[grow] += 2
                    if shrink == "C":
                        fx -= 2
                    if grow == "C":
                        fx += 2
                    board = _render(widths, fx)
        out = board.copy()
    elif recognised:
        out = board.copy()
    out[0] = _hud_row(new_steps)
    return out.astype(g.dtype)


def is_level_complete(grid):
    g = np.asarray(grid)
    if g.shape != (64, 64):
        return False
    widths, fx = _parse_board(g)
    return bool(np.array_equal(_render(widths, fx)[1:], g[1:]) and fx == TARGET_X)
