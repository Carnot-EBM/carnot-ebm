import numpy as np

def _find_block(grid, color):
    # find the 5x5 block of `color` (the player). Return (r0,c0) top-left or None.
    H, W = grid.shape
    mask = (grid == color)
    # find rows/cols where a 5x5 all-color square exists
    for r in range(H - 4):
        for c in range(W - 4):
            if mask[r:r+5, c:c+5].all():
                return (r, c)
    return None

def engine(grid, action, data):
    g = grid.astype(int).copy()
    H, W = g.shape
    # locate the 9 player block (5x5)
    pos = _find_block(g, 9)
    if pos is None:
        return g
    r0, c0 = pos
    # determine move direction
    if action == 2:
        dr, dc = 6, 0
    elif action == 4:
        dr, dc = 0, 6
    else:
        return g
    nr0, nc0 = r0 + dr, c0 + dc
    # bounds check
    if nr0 + 5 > H or nc0 + 5 > W:
        return g
    # clear old position to 5
    g[r0:r0+5, c0:c0+5] = 5
    # place new 9 block
    g[nr0:nr0+5, nc0:nc0+5] = 9
    # trail: a 2 block appears adjacent in the "other" axis at the old row/col band
    # For ACTION2 (down move): 2 block at old rows, cols c0+6..c0+10
    # For ACTION4 (right move): 2 block at rows r0+6..r0+10, old cols
    if action == 2:
        tr, tc = r0, c0 + 6
    else:
        tr, tc = r0 + 6, c0
    if tr + 5 <= H and tc + 5 <= W:
        g[tr:tr+5, tc:tc+5] = 2
    # bottom counter: r63, decrement a 9->1 at rightmost? handle later
    return g

def is_level_complete(grid):
    return False
