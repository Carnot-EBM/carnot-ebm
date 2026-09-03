import numpy as np

# ---------------------------------------------------------------------------
# World model for ARC-AGI-3 game 'r11l'
#
# Observed mechanics (induced from the transition log):
#  * Column 0 is a progress bar: on every action the next cell below the
#    contiguous run of 5s starting at row 0 turns to 5.
#  * The board contains a diagonal "track" made of colour-1 cells running
#    from top-left towards bottom-right inside the big colour-5 field.
#  * Three object types live on/around the track:
#      F-gem : 20-cell diamond of 15s with a 6 centre        (the mover)
#      G-gem : 12-cell diamond of 3s  with a 15 centre       (relay gem)
#      ring  : 12-cell ring of 0s     with a 15 centre       (socket / trail)
#  * Clicking an F-gem moves its centre roughly halfway (per axis, floored,
#    min 1 step) toward the nearest G-gem centre.  All existing rings are
#    erased, a fresh ring is left at the old centre, and the track segment
#    directly up-left behind the old position (consecutive 1-cells along the
#    (-1,-1) diagonal starting two cells up-left) is wiped back to 5.
#  * Clicking a G-gem converts it into a ring and spawns a new G-gem at the
#    up-left end of the remaining track (min r+c cell of colour 1), shifted
#    by (-2,-2); if no track remains it respawns at the level's start socket.
# ---------------------------------------------------------------------------

RING_OFFS = [(-2, 0), (-1, -1), (-1, 0), (-1, 1),
             (0, -2), (0, -1), (0, 1), (0, 2),
             (1, -1), (1, 0), (1, 1),
             (2, 0)]

F_OFFS = [(-2, -1), (-2, 0), (-2, 1),
          (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
          (0, -2), (0, -1), (0, 1), (0, 2),
          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
          (2, -1), (2, 0), (2, 1)]

G_FALLBACK_CENTER = (36, 7)   # top-left socket of this level layout


def _inb(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _find_f_centers(g):
    """Centres of F-gems: colour-6 cells surrounded by the 15-diamond."""
    out = []
    for r in range(g.shape[0]):
        row = g[r]
        for c in np.flatnonzero(row == 6):
            cnt = sum(1 for dr, dc in F_OFFS if _inb(r + dr, c + dc, *g.shape)
                      and g[r + dr, c + dc] == 15)
            if cnt >= 8:
                out.append((int(r), int(c)))
    return out


def _count_around(g, r, c, val):
    return sum(1 for dr, dc in RING_OFFS
               if _inb(r + dr, c + dc, *g.shape) and g[r + dr, c + dc] == val)


def _find_g_centers(g):
    """Centres of G-gems: a 15 with mostly 3s on the ring offsets."""
    out = []
    for r in range(g.shape[0]):
        for c in np.flatnonzero(g[r] == 15):
            if c == 0:
                continue
            if _count_around(g, r, c, 3) >= 7:
                out.append((int(r), int(c)))
    return out


def _find_rings(g, exclude=()):
    """Centres of empty rings: a 15 with mostly 0s on the ring offsets."""
    ex = set(exclude)
    out = []
    for r in range(g.shape[0]):
        for c in np.flatnonzero(g[r] == 15):
            if c == 0 or (r, c) in ex:
                continue
            if _count_around(g, r, c, 0) >= 8:
                out.append((int(r), int(c)))
    return out


def _draw_ring(g, cr, cc):
    g[cr, cc] = 15
    for dr, dc in RING_OFFS:
        if _inb(cr + dr, cc + dc, *g.shape):
            g[cr + dr, cc + dc] = 0


def _erase_ring(g, cr, cc):
    for dr, dc in RING_OFFS:
        if _inb(cr + dr, cc + dc, *g.shape):
            g[cr + dr, cc + dc] = 5
    if _inb(cr, cc, *g.shape):
        g[cr, cc] = 5


def _draw_f(g, cr, cc):
    g[cr, cc] = 6
    for dr, dc in F_OFFS:
        if _inb(cr + dr, cc + dc, *g.shape):
            g[cr + dr, cc + dc] = 15


def _erase_f_footprint(g, cr, cc):
    for dr, dc in F_OFFS:
        if _inb(cr + dr, cc + dc, *g.shape):
            g[cr + dr, cc + dc] = 5
    if _inb(cr, cc, *g.shape):
        g[cr, cc] = 5


def _draw_g(g, cr, cc):
    g[cr, cc] = 15
    for dr, dc in RING_OFFS:
        if _inb(cr + dr, cc + dc, *g.shape):
            g[cr + dr, cc + dc] = 3


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    # ---- progress bar in column 0 (one cell per action) -------------------
    k = 0
    while k < h and g[k, 0] == 5:
        k += 1
    if k < h:
        g[k, 0] = 5

    if action != 6 or not isinstance(data, dict):
        return g

    try:
        r = int(data.get('y', -1))
        c = int(data.get('x', -1))
    except Exception:
        return g
    if not _inb(r, c, h, w):
        return g

    f_centers = _find_f_centers(g)
    g_centers = _find_g_centers(g)
    rings = _find_rings(g, exclude=set(f_centers) | set(g_centers))

    def near(center, rad):
        cr, cc = center
        return abs(cr - r) <= rad and abs(cc - c) <= rad

    # ---- click on an F-gem: ease it halfway toward the nearest G-gem ------
    for fc in f_centers:
        if near(fc, 2):
            fr, fcc = fc
            target = None
            best = None
            for gc in g_centers:
                d = (gc[0] - fr) ** 2 + (gc[1] - fcc) ** 2
                if best is None or d < best:
                    best, target = d, gc
            if target is not None:
                tr, tc = target
                dr, dc = tr - fr, tc - fcc
                sr = ((abs(dr) + 1) // 2) * (1 if dr > 0 else -1 if dr < 0 else 0)
                sc = ((abs(dc) + 1) // 2) * (1 if dc > 0 else -1 if dc < 0 else 0)
                nr, nc = fr + sr, fcc + sc
                if _inb(nr, nc, h, w):
                    # wipe all existing rings
                    for rr in rings:
                        _erase_ring(g, *rr)
                    # clear old footprint, leave a ring at the old centre
                    _erase_f_footprint(g, fr, fcc)
                    _draw_ring(g, fr, fcc)
                    # erase the track segment directly up-left behind us
                    br, bc = fr - 2, fcc - 2
                    while _inb(br, bc, h, w) and g[br, bc] == 1:
                        g[br, bc] = 5
                        br -= 1
                        bc -= 1
                    # draw the moved gem on top
                    _draw_f(g, nr, nc)
            return g

    # ---- click on a G-gem: become a ring, respawn a G at the track start --
    for gc in g_centers:
        if near(gc, 2):
            gr, gcc = gc
            for rr in rings:
                _erase_ring(g, *rr)
            # convert this gem into an empty ring (3s -> 0s, keep 15 centre)
            for dr, dc in RING_OFFS:
                if _inb(gr + dr, gcc + dc, h, w):
                    g[gr + dr, gcc + dc] = 0
            g[gr, gcc] = 15
            # find up-left end of remaining track
            ones = np.argwhere(g == 1)
            spawn = None
            if len(ones):
                s = int(np.min(ones[:, 0] + ones[:, 1]))
                cand = ones[ones[:, 0] + ones[:, 1] == s]
                sr, sc = int(cand[0, 0]), int(cand[0, 1])
                spawn = (sr - 2, sc - 2)
            else:
                spawn = G_FALLBACK_CENTER
            if _inb(spawn[0], spawn[1], h, w):
                _draw_g(g, *spawn)
            return g

    return g


def is_level_complete(grid):
    # No explicit win frame was observed; the only monotonic completion
    # indicator in the data is the column-0 progress bar filling top->bottom.
    try:
        col = np.asarray(grid)[:, 0]
    except Exception:
        return False
    return bool(col.size > 0 and np.all(col == 5))