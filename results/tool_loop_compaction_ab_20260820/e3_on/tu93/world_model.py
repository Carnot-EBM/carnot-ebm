import numpy as np

def _find_player(grid):
    mask = (grid == 9) | (grid == 4)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0 = int(np.argmax(rows)); r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r0, c0, r1, c1

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    box = _find_player(g)
    if box is None:
        return g
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    step = 2 * h
    dr, dc = 0, 0
    if action == 1:
        dr = -1
    elif action == 2:
        dr = 1
    elif action == 3:
        dc = -1
    elif action == 4:
        dc = 1
    else:
        return g
    nr0, nc0 = r0 + dr * step, c0 + dc * step
    nr1, nc1 = r1 + dr * step, c1 + dc * step
    H, W = g.shape
    if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
        return g
    dest = g[nr0:nr1 + 1, nc0:nc1 + 1]
    if np.any((dest == 5) | (dest == 6) | (dest == 9) | (dest == 4)):
        return g
    g[r0:r1 + 1, c0:c1 + 1] = 0
    g[nr0:nr1 + 1, nc0:nc1 + 1] = 9
    # facing marker (4) at edge-center in movement direction
    if action == 1:
        g[nr0, (nc0 + nc1) // 2] = 4
    elif action == 2:
        g[nr1, (nc0 + nc1) // 2] = 4
    elif action == 3:
        g[(nr0 + nr1) // 2, nc0] = 4
    elif action == 4:
        g[(nr0 + nr1) // 2, nc1] = 4
    # bottom bar (color 6) erodes from the right by 1 cell per action
    last = g[-1]
    if (last == 6).any():
        idx = np.where(last == 6)[0]
        last[idx[-1]] = 0
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    return not ((g == 9) | (g == 4)).any()
