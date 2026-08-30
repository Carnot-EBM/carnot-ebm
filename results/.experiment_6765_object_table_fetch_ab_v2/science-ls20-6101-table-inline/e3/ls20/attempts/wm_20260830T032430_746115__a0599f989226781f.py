import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    # find the active 5x5 block: top 2 rows color 12, bottom 3 rows color 9
    best = None
    for r in range(H - 4):
        for c in range(W - 4):
            if (g[r, c:c+5].tolist() == [12]*5 and
                g[r+1, c:c+5].tolist() == [12]*5 and
                g[r+2, c:c+5].tolist() == [9]*5 and
                g[r+3, c:c+5].tolist() == [9]*5 and
                g[r+4, c:c+5].tolist() == [9]*5):
                best = (r, c)
                break
        if best:
            break
    if best is None:
        return g
    r, c = best
    # determine move direction
    dr, dc = 0, 0
    if action == 1:
        dr, dc = -5, 0
    elif action == 3:
        dr, dc = 0, -5
    elif action == 2:
        dr, dc = 5, 0
    elif action == 4:
        dr, dc = 0, 5
    nr, nc = r + dr, c + dc
    if nr < 0 or nc < 0 or nr + 5 > H or nc + 5 > W:
        return g
    # clear old block to 3
    g[r:r+5, c:c+5] = 3
    # place new block
    g[nr:nr+2, nc:nc+5] = 12
    g[nr+2:nr+5, nc:nc+5] = 9
    # decrement the 11 bar: leftmost column containing 11 -> set those 11s to 3
    mask = (g == 11)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) > 0:
        col = cols[0]
        g[mask[:, col]] = 3
    return g

def is_level_complete(grid):
    return False
