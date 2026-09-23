"""Expert grid-only engine for g50t level 1 (positive control).

Uses ONLY (grid, action, data) plus constants derived from the public game source
(environment_files/g50t/5849a774/g50t.py): the level-1 sprite layout, the move step
jarvstobjt=6, the HUD rule tmwgfkaqxj, the ACTION5 rewind rule pmlawcgvcp.
No file reads, no simulator calls, no recorded grids, no input hashing.

Known unobservable state (cannot be read from the grid, so the engine must guess):
  * ucorwtereb % 2 -- the HUD bar moves one cell only on even action counts
    (g50t.py tmwgfkaqxj). HUD_POLICY picks a fixed guess.
  * the recorded move history areahjypvy (ACTION5 at the start cell is a no-op iff
    it is empty) and every ghost's replay path.
"""
import numpy as np

STEP = 6  # jarvstobjt
START = (13, 7)  # qftsebtxuc.set_position(13, 7) in level 1
PLATE = (37, 7)  # medyellngi.set_position(37, 7); 3x3 of color 8 at offset (2, 2)
DOOR_CLOSED = (13, 37)  # kjrcloicja at (13, 37), rotation 270, slides +x by STEP when pressed
GOAL_CELL = (43, 49)  # gilbljmfbc at (42, 48); win when player at goal + (1, 1)
# Lattice positions whose sprite centre lies on the track sprite vsrojdvivb (13, 7).
WALKABLE = {
    (13, 7), (13, 13), (13, 19), (13, 25), (13, 31), (13, 37), (13, 43), (13, 49),
    (19, 7), (19, 19), (19, 49), (25, 7), (25, 13), (25, 19), (25, 49), (31, 7),
    (31, 49), (37, 7), (37, 49), (43, 49),
}
# Door image as rendered at rotation 270 (7x7), rows top->bottom.
DOOR_IMG = np.array(
    [
        [5, 5, 5, 5, 5, 5, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 5, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 8],
        [5, 5, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 5, 5, 5, 5, 5, 5],
    ]
)
HUD_ROW = 63
HUD_POLICY = "never"  # fixed guess for the hidden step parity; chosen on visible rows only
A5_AT_START_POLICY = "advance"  # hidden: is the move history empty? visible row 8 says advance

PLAYER_COLOR, GHOST_COLOR, CENTER_COLOR, FLOOR = 9, 2, 5, 5


def _block(g, x, y):
    return g[y + 1 : y + 6, x + 1 : x + 6]


def _is_body(g, x, y, color):
    b = _block(g, x, y)
    if b.shape != (5, 5) or b[2, 2] != CENTER_COLOR:
        return False
    m = np.ones((5, 5), bool)
    m[2, 2] = False
    return bool((b[m] == color).all())


def _find(g, color):
    return [p for p in sorted(WALKABLE) if _is_body(g, p[0], p[1], color)]


def _restore_floor(out, x, y):
    out[y + 1 : y + 6, x + 1 : x + 6] = FLOOR
    if (x, y) == PLATE:
        out[y + 2 : y + 5, x + 2 : x + 5] = 8


def _draw_body(out, x, y, color):
    out[y + 1 : y + 6, x + 1 : x + 6] = color
    out[y + 3, x + 3] = CENTER_COLOR


def _door_closed(g):
    x, y = DOOR_CLOSED
    return bool((g[y + 1, x + 1 : x + 6] == 8).all())


def _draw_door(out, closed):
    x, y = DOOR_CLOSED
    if closed:
        out[y : y + 7, x + 7 : x + 13] = _door_background_open_region()
        out[y : y + 7, x : x + 7] = DOOR_IMG
    else:
        out[y : y + 7, x : x + 6] = FLOOR
        out[y : y + 7, x + 6 : x + 13] = DOOR_IMG


def _door_background_open_region():
    # What lies under the open door's extra 6 columns (x 20..25, y 37..43) when the door
    # is closed: background 0 except the wire casing awhgrxsdnu (rows 39..41 = 5,8,5).
    reg = np.zeros((7, 6), dtype=int)
    reg[2, :] = 5
    reg[3, :] = 8
    reg[4, :] = 5
    return reg


def _indicator_state(g):
    return 0 if g[1, 1] == PLAYER_COLOR else 1


def _draw_indicator(out, state):
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    if state == 0:
        out[1:4, 1:4] = np.where(ring == 1, 9, 0)
        out[1:4, 5:8] = 1
        out[5, 1:4] = 9
        out[5, 5:8] = 0
    else:
        out[1:4, 1:4] = np.where(ring == 1, 2, 0)
        out[1:4, 5:8] = np.where(ring == 1, 9, 0)
        out[5, 1:4] = 0
        out[5, 5:8] = 9


def _hud(out):
    if HUD_POLICY == "never":
        return
    row = out[HUD_ROW]
    ones = int((row == 1).sum())
    col = row.shape[0] - 1 - ones
    if col >= 0:
        row[col] = 1


def engine(grid, action, data):
    g = np.asarray(grid)
    out = g.copy()
    a = int(action)
    players = _find(g, PLAYER_COLOR)
    if not players:
        _hud(out)
        return out
    px, py = players[0]
    ghosts = _find(g, GHOST_COLOR)
    if a in (1, 2, 3, 4):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[a]
        tx, ty = px + dx * STEP, py + dy * STEP
        closed = _door_closed(g)
        blocked = (tx, ty) not in WALKABLE or (closed and (tx, ty) == DOOR_CLOSED)
        if not blocked:
            _restore_floor(out, px, py)
            for gx, gy in ghosts:
                _draw_body(out, gx, gy, GHOST_COLOR)
            _draw_body(out, tx, ty, PLAYER_COLOR)
            pressed_before = (px, py) == PLATE or PLATE in ghosts
            pressed_after = (tx, ty) == PLATE or PLATE in ghosts
            if pressed_before != pressed_after:
                _draw_door(out, closed=not pressed_after)
        _hud(out)
        return out
    if a == 5:
        at_start = (px, py) == START
        if at_start and A5_AT_START_POLICY != "advance":
            _hud(out)
            return out
        state = _indicator_state(g)
        _restore_floor(out, px, py)
        for gx, gy in ghosts:
            _restore_floor(out, gx, gy)
        if PLATE in ghosts or (px, py) == PLATE:
            if not _door_closed(g):
                _draw_door(out, closed=True)
        _draw_body(out, START[0], START[1], PLAYER_COLOR)
        _draw_indicator(out, 1 - state)
        _hud(out)
        return out
    _hud(out)
    return out


def is_level_complete(grid):
    g = np.asarray(grid)
    return _is_body(g, GOAL_CELL[0], GOAL_CELL[1], PLAYER_COLOR)
