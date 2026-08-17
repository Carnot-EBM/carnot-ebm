import numpy as np


def _player_mask(g):
    """Color-5 blob = the controlled object (exclude floor row & counter column)."""
    m = (g == 5)
    m[g.shape[0] - 1, :] = False
    m[:, g.shape[1] - 1] = False
    return m


def _twin_mask(g):
    """Color-4 blob = mirror twin of the player."""
    return (g == 4)


def _shift(mask, dr, dc, g, allowed):
    """Rigidly shift a mask by (dr,dc); keep only cells landing on `allowed` ground."""
    H, W = g.shape
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        ny, nx = y + dr, x + dc
        if 0 <= ny < H and 0 <= nx < W and allowed[ny, nx]:
            out[ny, nx] = True
    return out


def engine(grid, action, data):
    g = np.asarray(grid)
    H, W = g.shape
    out = g.copy()

    DIRS = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}
    moved = action in DIRS

    if moved:
        p = _player_mask(g)
        t = _twin_mask(g)
        dr, dc = DIRS[action]
        # twin mirrors horizontally across the central wall column
        tr, tc = dr, -dc

        # destinations may be background or cells vacated by either mover this step
        free5 = (g == 9) | p | t
        free4 = (g == 9) | p | t

        n5 = _shift(p, dr, dc, g, free5)
        n4 = _shift(t, tr, tc, g, free4)

        # erase old positions, then place new ones (movers overwrite soft matter,
        # but never walls(10)/floor/counter-column since those are not "free")
        out[p] = 9
        out[t] = 9
        out[n4] = 4
        out[n5] = 5

        # move counter: right-edge column grows downward one cell per move
        col = W - 1
        r = 0
        while r < H and g[r, col] == 5:
            r += 1
        if r < H and g[r, col] == 11:
            out[r, col] = 5

    return out


def is_level_complete(grid):
    g = np.asarray(grid)
    H, W = g.shape

    # dock template = the color-11 J structure in the play area (not the edge column)
    tmpl = (g == 11).copy()
    tmpl[:, W - 1] = False
    if not tmpl.any():
        return False
    ys, xs = np.nonzero(tmpl)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    # mirror axis = center of the color-10 wall
    wcols = np.nonzero((g == 10).any(axis=0))[0]
    center = int(round(float(wcols.mean()))) if len(wcols) else W // 2 - 1
    mir = lambda c: int(2 * center - c)

    # player must be exactly docked at the mirrored target slot
    p = _player_mask(g)
    if not p.any():
        return False
    py, px = np.nonzero(p)
    pr0, pr1, pc0, pc1 = int(py.min()), int(py.max()), int(px.min()), int(px.max())
    ok_player = (pr0, pr1) == (r0, r1) and (pc0, pc1) == (mir(c1), mir(c0))
    if not ok_player:
        return False

    # twin must have reached its own dock zone (it may keep holes where template
    # cells survived, so only require it to sit inside the template footprint)
    t = _twin_mask(g)
    if not t.any():
        return False
    ty, tx = np.nonzero(t)
    tr0, tr1, tc0, tc1 = int(ty.min()), int(ty.max()), int(tx.min()), int(tx.max())
    ok_twin = (tr0 >= r0) and (tr1 <= r1) and (tc0 >= c0) and (tc1 <= c1)
    return bool(ok_twin)