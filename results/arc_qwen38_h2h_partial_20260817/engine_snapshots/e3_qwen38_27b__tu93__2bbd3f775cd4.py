import numpy as np


def engine(grid, action, data):
    g = np.array(grid, copy=True)
    H, W = g.shape

    # Terminal win state: do not evolve further.
    if is_level_complete(g):
        return g

    # Locate the 3x3 player sprite (colors 9 and 4).
    mask = (g == 9) | (g == 4)
    if not mask.any():
        return g

    ys, xs = np.nonzero(mask)
    r = int(ys.min())
    c = int(xs.min())

    dr = dc = 0
    if action == 1:       # up
        dr = -6
    elif action == 2:     # down
        dr = 6
    elif action == 3:     # left
        dc = -6
    elif action == 4:     # right
        dc = 6
    else:
        return g

    nr = r + dr
    nc = c + dc
    ir = r + dr // 2
    ic = c + dc // 2

    # Bounds for destination and intermediate 3x3 tile.
    if nr < 0 or nc < 0 or nr + 2 >= H or nc + 2 >= W:
        return g
    if ir < 0 or ic < 0 or ir + 2 >= H or ic + 2 >= W:
        return g

    # Color 5 is solid background/wall. The jumped-over tile and landing tile
    # must both be non-solid.
    if int(g[ir + 1, ic + 1]) == 5 or int(g[nr + 1, nc + 1]) == 5:
        return g

    # Reaching a goal tile (color 14) completes the level.
    winning = bool((g[nr:nr + 3, nc:nc + 3] == 14).any())

    # Current fuel/timer bar length: contiguous prefix of color 6 on last row.
    L = 0
    for v in g[-1]:
        if int(v) == 6:
            L += 1
        else:
            break

    # Clear old player tile; leaving it as empty 0 matches observed deltas.
    r0 = max(0, r)
    c0 = max(0, c)
    r1 = min(H, r + 3)
    c1 = min(W, c + 3)
    g[r0:r1, c0:c1] = 0

    # Place new player sprite at destination.
    nr0 = max(0, nr)
    nc0 = max(0, nc)
    nr1 = min(H, nr + 3)
    nc1 = min(W, nc + 3)
    g[nr0:nr1, nc0:nc1] = 9

    # Indicator cell shows movement direction. On a winning move, leave one
    # goal-colored cell inside the sprite so is_level_complete can detect it.
    if action == 1:
        pr, pc = nr, nc + 1
    elif action == 2:
        pr, pc = nr + 2, nc + 1
    elif action == 3:
        pr, pc = nr + 1, nc
    else:
        pr, pc = nr + 1, nc + 2

    if 0 <= pr < H and 0 <= pc < W:
        g[pr, pc] = 14 if winning else 4

    if winning:
        # Level complete: reset the bottom bar to full.
        g[-1, :] = 6
    else:
        # Observed fuel rule: normally shrink by 1; when current length mod 5
        # is 4, shrink by 2.
        cost = 0
        if L > 0:
            cost = 2 if (L % 5 == 4) else 1

        for k in range(cost):
            idx = L - 1 - k
            if idx >= 0:
                g[-1, idx] = 0

    return g


def is_level_complete(grid):
    H, W = grid.shape
    if H < 3 or W < 3:
        return False

    g = np.asarray(grid)

    # A completed state produced by engine() has a player tile that contains
    # both body color 9 and goal color 14. Normal states keep these colors in
    # separate aligned 3x3 tiles.
    if not ((g == 9).any() and (g == 14).any()):
        return False

    for r in range(0, H - 2, 3):
        for c in range(0, W - 2, 3):
            block = g[r:r + 3, c:c + 3]
            if (block == 14).any() and (block == 9).any():
                return True

    return False