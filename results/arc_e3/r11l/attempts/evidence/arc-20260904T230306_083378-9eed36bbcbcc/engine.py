import numpy as np


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)

    try:
        act = int(action)
    except Exception:
        return g

    # Only ACTION6 was observed to have an effect in the provided transition.
    if act != 6:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    x = y = None
    if isinstance(data, dict):
        x = data.get("x", data.get("px"))
        y = data.get("y", data.get("py"))

    try:
        c = int(round(float(x)))
        r = int(round(float(y)))
    except Exception:
        return g

    if not (0 <= r < H and 0 <= c < W):
        return g

    # The clicked object in the observed transition is a rounded-square core:
    # a central color-6 pixel surrounded by a 5x5 shell of color 15/12 with
    # the four corners missing.  Clicking any cell of such a core sets the
    # top-left progress/start flag to background color 5.
    mask_offsets = [
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if not (abs(dr) == 2 and abs(dc) == 2)
    ]
    shell_offsets = [off for off in mask_offsets if off != (0, 0)]

    clickable = False

    for cr in range(H):
        for cc in range(W):
            if g[cr, cc] != 6:
                continue

            ok = True
            for dr, dc in shell_offsets:
                rr = cr + dr
                c2 = cc + dc
                if rr < 0 or rr >= H or c2 < 0 or c2 >= W:
                    ok = False
                    break
                v = g[rr, c2]
                if v != 15 and v != 12:
                    ok = False
                    break

            if not ok:
                continue

            for dr, dc in mask_offsets:
                if cr + dr == r and cc + dc == c:
                    clickable = True
                    break

            if clickable:
                break

        if clickable:
            break

    if clickable:
        g[0, 0] = 5

    return g


def is_level_complete(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if H == 0 or W == 0:
        return False

    # The observed opening/current boards are not complete.  A reasonable
    # terminal condition inferred from the object vocabulary is that either
    # all central color-6 cores have been removed, or all remaining objective
    # objects (color 3 / color 12 components) have been cleared, with the
    # level-start flag active at the top-left corner.
    started = bool(g[0, 0] == 5)
    has_core = bool(np.any(g == 6))
    has_objectives = bool(np.any(g == 3) or np.any(g == 12))

    return started and ((not has_core) or (not has_objectives))