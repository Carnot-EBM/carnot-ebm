"""Expert grid-only engine for sp80 level 1 (positive control).

Uses ONLY its three arguments plus constants read from the public source
environment_files/sp80/589a99af/sp80.py. It reads no files, calls no simulator,
and embeds no recorded grids.

Constants from the source (level 1):
  - 16x16 logical grid drawn at scale 4 onto 64x64, no letterbox (camera.py).
  - display row 0 is a step bar: 14 for x < round(64*steps/30), else 0 (sklgkadoxw).
  - "steps" = 30 on level 1; every non-RESET action costs 1 step (ytqycovhld(1)).
  - colors: 12 background, 9 selected platform, 8 unselected platform,
    15 unselected diverter, 11 cup, 6 water source, 4 source marker, 1 floor bar.
  - move rules (step + husluhmboo): new y < 3 is refused; a 1-cell margin around
    every cup is refused; overlapping any non-platform sprite or the border is refused;
    overlapping only platforms is allowed.
  - ACTION5 (spill): a non-level-up outcome always ends in the rest state, so only the
    bar changes (plus re-selection of the platform nearest the origin after a failed
    spill; level 1 has one platform, so nothing visible changes).
  - ACTION6 (click): selects the platform whose bounding box holds the clicked cell.
"""
import numpy as np

SCALE = 4
N = 16
STEPS_MAX = 30  # level 1 "steps" in sp80.py levels[0].data
BG = 12
SEL = 9
PLATFORM = 8
DIVERTER = 15
CUP = 11
BLOCKING = {1, 4, 6, 11, 13, 14}  # floor bar, source marker, water, cup, filled cup, hit bar


def _logical(grid):
    # sample the centre of each 4x4 block; display row 0 (the bar) is never sampled
    return grid[SCALE // 2::SCALE, SCALE // 2::SCALE][:N, :N].copy()


def _bar_after_step(grid):
    row = grid[0]
    w = int((row == 14).sum())
    steps = None
    for s in range(STEPS_MAX + 1):
        if round(64 * s / STEPS_MAX) == w:
            steps = s
            break
    if steps is None:  # unexpected bar; leave it alone
        return row.copy()
    s2 = max(0, steps - 1)
    w2 = round(64 * s2 / STEPS_MAX)
    out = np.zeros(64, dtype=grid.dtype)
    out[:w2] = 14
    return out


def _components(L, colors):
    """Connected components (4-neighbour) of cells whose value is in `colors`."""
    seen = np.zeros(L.shape, dtype=bool)
    comps = []
    for y in range(L.shape[0]):
        for x in range(L.shape[1]):
            if seen[y, x] or L[y, x] not in colors:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < L.shape[0] and 0 <= nx < L.shape[1] and not seen[ny, nx] \
                            and L[ny, nx] in colors:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _bbox(cells):
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1  # x, y, w, h


def _paint(grid, L_new, L_old):
    out = grid.copy()
    diff = np.argwhere(L_new != L_old)
    for y, x in diff:
        out[y * SCALE:(y + 1) * SCALE, x * SCALE:(x + 1) * SCALE] = L_new[y, x]
    return out


def _cup_boxes(L):
    return [_bbox(c) for c in _components(L, {CUP})]


def engine(grid, action, data):
    grid = np.asarray(grid)
    L = _logical(grid)
    L_new = L.copy()
    action = int(action)

    sel_comps = _components(L, {SEL})
    sel = sel_comps[0] if sel_comps else None

    if action in (1, 2, 3, 4) and sel is not None:
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
        x, y, pw, ph = _bbox(sel)
        nx, ny = x + dx, y + dy
        ok = ny >= 3
        if ok:
            for rx, ry, rw, rh in _cup_boxes(L):
                if nx < rx + rw + 1 and nx + pw > rx - 1 and ny < ry + rh + 1 and ny + ph > ry - 1:
                    ok = False
                    break
        if ok:
            sel_set = set(sel)
            for cy, cx in sel:
                ty, tx = cy + dy, cx + dx
                if not (0 <= ty < N and 0 <= tx < N):
                    ok = False  # border sprite bodekplurlf16 surrounds the grid
                    break
                if (ty, tx) in sel_set:
                    continue
                if int(L[ty, tx]) in BLOCKING:
                    ok = False
                    break
        if ok:
            for cy, cx in sel:
                L_new[cy, cx] = BG
            for cy, cx in sel:
                L_new[cy + dy, cx + dx] = SEL
    elif action == 6 and isinstance(data, dict):
        gx, gy = int(data.get("x", 0)) // SCALE, int(data.get("y", 0)) // SCALE
        if 0 <= gx < N and 0 <= gy < N:
            for comp in _components(L, {PLATFORM, DIVERTER}):
                bx, by, bw, bh = _bbox(comp)
                if bx <= gx < bx + bw and by <= gy < by + bh:
                    if sel is not None:
                        for cy, cx in sel:
                            L_new[cy, cx] = PLATFORM
                    for cy, cx in comp:
                        L_new[cy, cx] = SEL
                    break
    elif action == 5:
        # failed-spill reset re-selects the platform nearest the origin (min x^2+y^2)
        plats = _components(L, {SEL, PLATFORM})
        if len(plats) > 1:
            best = min(plats, key=lambda c: _bbox(c)[0] ** 2 + _bbox(c)[1] ** 2)
            for comp in plats:
                for cy, cx in comp:
                    L_new[cy, cx] = SEL if comp is best else PLATFORM

    out = _paint(grid, L_new, L)
    out[0] = _bar_after_step(grid)
    return out


def is_level_complete(grid):
    # Level completion happens inside ACTION5 and is never a rest state; not modelled.
    return False
