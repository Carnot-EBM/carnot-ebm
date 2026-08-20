import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    # Find the player: a 6x6 block of color 6 (the moving object)
    # Identify all 6x6 all-6 blocks
    best = None
    for r in range(H-5):
        for c in range(W-5):
            block = g[r:r+6, c:c+6]
            if np.all(block == 6):
                # candidate player top-left
                if best is None:
                    best = (r, c)
    if best is None:
        return g
    r0, c0 = best
    # The player occupies rows r0..r0+5, cols c0..c0+5 (6x6 of 6s)
    # plus a "face" of 1s/2s to the right at rows r0+2..r0+3, cols c0+5..c0+11
    # Build the player sprite from current grid
    sprite = g[r0:r0+6, c0:c0+12].copy()  # 6 rows x 12 cols
    # Determine background color (what's around) - use 4 (blue) as default
    bg = 4
    if action == 1:  # move up
        dr, dc = -6, 0
    elif action == 2:  # move down
        dr, dc = 6, 0
    elif action == 3:  # move left
        dr, dc = 0, -6
    elif action == 4:  # move right
        dr, dc = 0, 6
    else:
        return g
    nr0, nc0 = r0+dr, c0+dc
    if nr0 < 0 or nc0 < 0 or nr0+6 > H or nc0+12 > W:
        return g
    # Clear old sprite area (set to bg)
    g[r0:r0+6, c0:c0+12] = bg
    # Place new sprite
    g[nr0:nr0+6, nc0:nc0+12] = sprite
    return g

def is_level_complete(grid):
    return False
