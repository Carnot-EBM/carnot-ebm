"""Hand-written grid-only engine for ar25 level 0 (positive control).

Derived from environment_files/ar25/0c556536/ar25.py (Ar25.step, iywvsmblpj/toseucmfue
render_interface). Uses ONLY (grid, action, data) plus constants from that source.
No file reads, no simulator, no recorded grids.

Frame layout (21x21 board, scale 3, cell (x, y) -> pixels rows 3y..3y+2, cols 3x..3x+2):
  background 9, mirror 10, target 11, piece 5, reflection 4 (pbtdgroplk).
  Center pixel (3y+1, 3x+1) is the minimap overlay: piece cell -> 0 (selected),
  target -> 11, mirror -> 10 (priority 11 > 10/5 > 0 > 9).
  Column 63 is the step bar: rows 0..used-1 are 5, the rest 11 (StepCounter 64).

A7 (undo) pops a history stack that the frame does not show. PRIMARY POLICY: predict no
change (the last move is unknown from the grid). `A7_POLICY` can be set to a fixed guess
for the sensitivity report only; it is NOT tuned on held-out rows.
"""
import numpy as np

SCALE = 3
N = 21            # 63 // 3
BG, MIRROR, TARGET, PIECE, REFL, DOT = 9, 10, 11, 5, 4, 0
STEP_COUNTER = 64
BAR_USED, BAR_LEFT = 5, 11
A7_POLICY = None  # None = identity; or one of 'up','down','left','right' (sensitivity only)
_DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_NAMED = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}


def _parse(g):
    corner = g[0:3 * N:3, 0:3 * N:3]          # [y, x]
    center = g[1:3 * N:3, 1:3 * N:3]
    piece = {(x, y) for y in range(N) for x in range(N) if corner[y, x] == PIECE}
    targets = {(x, y) for y in range(N) for x in range(N) if center[y, x] == TARGET}
    mxs = [x for y in range(N) for x in range(N) if center[y, x] == MIRROR]
    mirror_x = max(set(mxs), key=mxs.count) if mxs else None
    col = g[:, 63]
    used = 0
    while used < 64 and col[used] == BAR_USED:
        used += 1
    return piece, targets, mirror_x, used


def _reflect(piece, mirror_x):
    if mirror_x is None:
        return set()
    out = set()
    for (x, y) in piece:
        rx = 2 * mirror_x - x
        if 0 <= rx < N and (rx, y) not in piece:
            out.add((rx, y))
    return out


def _render(g, piece, targets, mirror_x, used):
    new = g.copy()
    refl = _reflect(piece, mirror_x)
    for y in range(N):
        for x in range(N):
            if (x, y) in piece:
                c = PIECE
            elif (x, y) in refl:
                c = REFL
            elif (x, y) in targets:
                c = TARGET
            elif x == mirror_x:
                c = MIRROR
            else:
                c = BG
            new[3 * y:3 * y + 3, 3 * x:3 * x + 3] = c
            # minimap dot on the cell center
            if (x, y) in targets:
                d = TARGET
            elif x == mirror_x:
                d = MIRROR
            elif (x, y) in piece:
                d = DOT
            else:
                d = c
            new[3 * y + 1, 3 * x + 1] = d
    u = max(0, min(used, 64))
    new[:u, 63] = BAR_USED
    new[u:, 63] = BAR_LEFT
    return new


def _move(g, piece, targets, mirror_x, used, dx, dy):
    xs = [p[0] for p in piece]
    ys = [p[1] for p in piece]
    x0, y0 = min(xs) + dx, min(ys) + dy
    w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if x0 < 0 or x0 + w > N or y0 < 0 or y0 + h > N:
        return g.copy()                       # blocked: no move, no step charged
    moved = {(x + dx, y + dy) for (x, y) in piece}
    covered = moved | _reflect(moved, mirror_x)
    if targets and targets <= covered:
        return _render(g, moved, targets, mirror_x, used)   # level-up row (excluded by the verifier)
    return _render(g, moved, targets, mirror_x, used + 1)


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    if g.shape != (64, 64):
        return g
    piece, targets, mirror_x, used = _parse(g)
    a = int(action)
    if a in _DIRS and piece:
        return _move(g, piece, targets, mirror_x, used, *_DIRS[a])
    if a == 5:
        return _render(g, piece, targets, mirror_x, used + 1)
    if a == 7:
        if A7_POLICY is None or not piece:
            return g
        dx, dy = _NAMED[A7_POLICY]            # undo restores a prior position; no step charged
        return _render(g, {(x + dx, y + dy) for (x, y) in piece}, targets, mirror_x, used)
    return g                                  # A6 click: selection is unchanged on this level


def is_level_complete(grid):
    g = np.asarray(grid)
    piece, targets, mirror_x, _ = _parse(g)
    return bool(targets) and targets <= (piece | _reflect(piece, mirror_x))
