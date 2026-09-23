"""Expert grid-only engine for wa30 (positive control).

Uses ONLY (grid, action, data) plus constants read from the public game source
environment_files/wa30/ee6fef47/wa30.py. No file reads, no simulator, no lookup.

Mechanics taken from the source:
  - 4-pixel lattice (celomdfhbh = 4). ACTION1..4 move the player by 4 (up/down/left/right).
  - Player sprite wppuejnwhl: row 0 = 0, rows 1-3 = 14. Rotation 0/90/180/270 puts the
    black edge top/right/bottom/left (arcengine rotates clockwise).
  - Not carrying: rotation is set to the move direction even if the move is blocked.
  - Collision uses TOP-LEFT positions of collidable sprites plus an out-of-board ring.
  - Box pktgsotzmw: ring colour 4, centre 9. Ring colour after each step (zzppkjnqgk):
    grabbed by player -> 0; player faces it -> 3; otherwise 4 (5 if grabbed by an AI).
  - ACTION5: carrying -> release; else grab the faced box.
  - Carrying: rotation is kept, box keeps its offset, both move only if both targets free.
  - Step bar on row 63: round(64 * steps_left / StepCounter) cells of 7, rest 4.
    steps_left is NOT visible, so the bar change is a guess (see BAR_POLICY).
"""
import numpy as np

L = 4
BG = 1
PLAYER = 14
BOX_CENTRE = 9
BOX_RINGS = (0, 3, 4, 5)
ZONE_EDGE = 9
ZONE_FILL = 2
BAR_ROW = 63
BAR_ON = 7
BAR_OFF = 4
# "hold": never decrement the bar. Each bar length covers 2-4 step counts, and only
# the lowest one decrements, so "hold" is the most likely outcome for every bar length.
BAR_POLICY = "hold"
DIRS = {1: (0, -L), 2: (0, L), 3: (-L, 0), 4: (L, 0)}
ROT_OF = {1: 0, 2: 180, 3: 270, 4: 90}
FACE = {0: (0, -L), 90: (L, 0), 180: (0, L), 270: (-L, 0)}


def _cells(g):
    """Yield top-left (x, y) of every 4x4 lattice cell in the play area."""
    h, w = g.shape
    for y in range(0, h, L):
        for x in range(0, w, L):
            yield x, y


def _block(g, x, y):
    return g[y : y + L, x : x + L]


def _find_player(g):
    ys, xs = np.where(g[:BAR_ROW] == PLAYER)
    if len(ys) == 0:
        return None
    x = (int(xs.min()) // L) * L
    y = (int(ys.min()) // L) * L
    b = _block(g, x, y)
    rows = b.shape[0]
    if np.all(b[0, :] == 0):
        rot = 0
    elif np.all(b[:, -1] == 0):
        rot = 90
    elif np.all(b[:, 0] == 0):
        rot = 270
    else:
        rot = 180  # bottom edge black, or hidden under the bar row
    return x, y, rot


def _is_box(b):
    if b.shape != (L, L):
        return False
    if not np.all(b[1:3, 1:3] == BOX_CENTRE):
        return False
    ring = np.concatenate([b[0, :], b[-1, :], b[1:3, 0], b[1:3, -1]])
    return len(set(ring.tolist())) == 1 and int(ring[0]) in BOX_RINGS


def _find_boxes(g, player):
    boxes = {}
    for x, y in _cells(g):
        if y + L > BAR_ROW:
            continue
        if player and (x, y) == (player[0], player[1]):
            continue
        b = _block(g, x, y)
        if _is_box(b):
            boxes[(x, y)] = int(b[0, 0])
    return boxes


def _zone_member(g, x, y, occupied):
    """Is the covered lattice cell (x, y) part of a 9-edged / 2-filled zone?

    A zone cell's visible neighbour shows fill (2) on the edge facing the covered cell
    when the zone continues through it; a border edge (9) means the zone stops there.
    """
    h, w = g.shape
    for dx, dy in ((L, 0), (-L, 0), (0, L), (0, -L)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in occupied:
            continue
        nb = _block(g, nx, ny)
        if nb.shape != (L, L) or ZONE_FILL not in nb:
            continue
        if dx > 0:
            edge = nb[1:3, 0]
        elif dx < 0:
            edge = nb[1:3, -1]
        elif dy > 0:
            edge = nb[0, 1:3]
        else:
            edge = nb[-1, 1:3]
        if np.all(edge == ZONE_FILL):
            return True
    return False


def _zone_pixels(g, x, y, occupied):
    """Rebuild a zone cell: 9 on every side whose neighbour is not zone, else 2."""
    h, w = g.shape
    out = np.full((L, L), ZONE_FILL, dtype=g.dtype)

    def nb_zone(nx, ny):
        if not (0 <= nx < w and 0 <= ny < h):
            return False
        if (nx, ny) in occupied:
            return _zone_member(g, nx, ny, occupied)
        nb = _block(g, nx, ny)
        return nb.shape == (L, L) and ZONE_FILL in nb

    if not nb_zone(x, y - L):
        out[0, :] = ZONE_EDGE
    if not nb_zone(x, y + L):
        out[-1, :] = ZONE_EDGE
    if not nb_zone(x - L, y):
        out[:, 0] = ZONE_EDGE
    if not nb_zone(x + L, y):
        out[:, -1] = ZONE_EDGE
    return out


def _underlay(g, x, y, occupied):
    if _zone_member(g, x, y, occupied):
        return _zone_pixels(g, x, y, occupied)
    return np.full((L, L), BG, dtype=g.dtype)


def _player_pixels(rot, dtype):
    base = np.array([[0] * 4] + [[PLAYER] * 4] * 3, dtype=dtype)
    k = int((-rot % 360) / 90)
    return np.rot90(base, k=k) if k else base


def _box_pixels(ring, dtype):
    b = np.full((L, L), ring, dtype=dtype)
    b[1:3, 1:3] = BOX_CENTRE
    return b


def _paint(out, x, y, pix):
    h, w = out.shape
    y1 = min(y + L, BAR_ROW)  # the bar row is drawn last, on top of everything
    if y1 > y:
        out[y:y1, x : x + L] = pix[: y1 - y]


def _free(pos, solid):
    x, y = pos
    if x < 0 or y < 0 or x > 64 - L or y > 64 - L:
        return False
    return pos not in solid


def engine(grid, action, data):
    g = np.asarray(grid)
    out = g.copy()
    player = _find_player(g)
    if player is None or action not in (1, 2, 3, 4, 5):
        return out
    px, py, rot = player
    boxes = _find_boxes(g, player)
    carried = next((p for p, c in boxes.items() if c == 0), None)
    occupied = set(boxes) | {(px, py)}
    solid = set(occupied)  # every movable sprite here is collidable (layer-1 sprites)

    npx, npy, nrot = px, py, rot
    box_moves = {}
    new_carried = carried
    if action in DIRS:
        dx, dy = DIRS[action]
        if carried is None:
            nrot = ROT_OF[action]
            if _free((px + dx, py + dy), solid):
                npx, npy = px + dx, py + dy
        else:
            ox, oy = carried[0] - px, carried[1] - py
            v = (px + dx, py + dy)
            t = (v[0] + ox, v[1] + oy)
            v_ok = (v not in solid or v == carried) and _free(v, set())
            t_ok = (t not in solid or t == (px, py)) and _free(t, set())
            if v_ok and t_ok:
                npx, npy = v
                box_moves[carried] = t
                new_carried = t
    else:  # ACTION5
        if carried is not None:
            new_carried = None
        else:
            fx, fy = FACE[rot]
            target = (px + fx, py + fy)
            if target in boxes:
                new_carried = target

    # Final box positions and ring colours.
    final_boxes = {}
    for p in boxes:
        final_boxes[box_moves.get(p, p)] = p
    fx, fy = FACE[nrot]
    faced = (npx + fx, npy + fy)
    new_ring = {}
    for p in final_boxes:
        if p == new_carried:
            new_ring[p] = 0
        elif p == faced:
            new_ring[p] = 3
        else:
            old = boxes[final_boxes[p]]
            new_ring[p] = 5 if old == 5 else 4

    # Erase cells that were vacated, then redraw movable sprites.
    old_cells = {(px, py)} | set(box_moves)
    new_cells = {(npx, npy)} | set(box_moves.values())
    for cx, cy in old_cells - new_cells:
        _paint(out, cx, cy, _underlay(g, cx, cy, occupied))
    for p, ring in new_ring.items():
        _paint(out, p[0], p[1], _box_pixels(ring, out.dtype))
    _paint(out, npx, npy, _player_pixels(nrot, out.dtype))

    # Step bar: steps_left is hidden; BAR_POLICY decides.
    if BAR_POLICY == "decrement":
        k = int((g[BAR_ROW] == BAR_ON).sum())
        out[BAR_ROW, :] = BAR_OFF
        out[BAR_ROW, : max(0, k - 1)] = BAR_ON
    return out


def is_level_complete(grid):
    return False
