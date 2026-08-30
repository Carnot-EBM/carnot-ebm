import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    # Find the main object: a 5x5 block with top 2 rows = 12, bottom 3 rows = 9
    # Search for the top-left of such a block
    found = None
    for r in range(H - 4):
        for c in range(W - 4):
            ok = True
            for i in range(2):
                if not np.all(g[r + i, c:c + 5] == 12):
                    ok = False
                    break
            if ok:
                for i in range(2, 5):
                    if not np.all(g[r + i, c:c + 5] == 9):
                        ok = False
                        break
            if ok:
                found = (r, c)
                break
        if found:
            break
    if found is None:
        return g
    r0, c0 = found
    # Determine move direction
    dr, dc = 0, 0
    if action == 1:
        dr, dc = -5, 0
    elif action == 3:
        dr, dc = 0, -5
    elif action == 2:
        dr, dc = 5, 0
    elif action == 4:
        dr, dc = 0, 5
    else:
        return g
    r1, c1 = r0 + dr, c0 + dc
    # Check bounds
    if r1 < 0 or c1 < 0 or r1 + 5 > H or c1 + 5 > W:
        return g
    # Capture object pattern
    obj = g[r0:r0 + 5, c0:c0 + 5].copy()
    # Clear old location (set to 3)
    g[r0:r0 + 5, c0:c0 + 5] = 3
    # Place object at new location
    g[r1:r1 + 5, c1:c1 + 5] = obj
    return g

def is_level_complete(grid):
    return False
