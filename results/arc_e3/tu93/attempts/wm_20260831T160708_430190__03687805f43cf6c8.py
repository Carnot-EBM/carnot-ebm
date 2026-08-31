import numpy as np

def _block_pos(grid):
    # find the 3x3 block containing a 4 (the special "player" block, 9s with 4 center)
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 4:
                # center of the 3x3 block
                r0, c0 = r - 1, c - 1
                return r0, c0
    return None

def engine(grid, action, data):
    g = grid.astype(int).copy()
    H, W = g.shape
    pos = _block_pos(g)
    if pos is None:
        return g
    r0, c0 = pos
    # block grid: blocks are 3x3 starting at row 15, col 15, step 3
    # map to block coords
    br, bc = (r0 - 15) // 3, (c0 - 15) // 3
    # actions: 1=up, 2=down, 3=left, 4=right (block movement)
    dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
    nbr, nbc = br + dr, bc + dc
    # bounds: 11x11 block grid
    if 0 <= nbr < 11 and 0 <= nbc < 11:
        nr0, nc0 = 15 + nbr * 3, 15 + nbc * 3
        # move the 3x3 block
        block = g[r0:r0+3, c0:c0+3].copy()
        g[r0:r0+3, c0:c0+3] = 0
        g[nr0:nr0+3, nc0:nc0+3] = block
    return g

def is_level_complete(grid):
    return False
