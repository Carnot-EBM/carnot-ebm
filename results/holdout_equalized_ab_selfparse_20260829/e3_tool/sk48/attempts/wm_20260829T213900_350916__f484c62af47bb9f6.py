import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    # Find the player: 6x6 block of color 6 (border)
    # locate top-left of a 6x6 all-6 block
    def find_player(g):
        for r in range(H-5):
            for c in range(W-5):
                if np.all(g[r:r+6, c:c+6] == 6):
                    return r, c
        return None
    p = find_player(g)
    if p is None:
        return g
    r0, c0 = p
    # extract player sprite (6x6)
    sprite = g[r0:r0+6, c0:c0+6].copy()
    # clear player region
    g[r0:r0+6, c0:c0+6] = 5
    # movement: action 1 = up, 2 = down, 3 = left, 4 = right (guess)
    dr, dc = 0, 0
    if action == 1:
        dr = -1
    elif action == 2:
        dr = 1
    elif action == 3:
        dc = -1
    elif action == 4:
        dc = 1
    nr, nc = r0+dr, c0+dc
    if 0 <= nr and nr+6 <= H and 0 <= nc and nc+6 <= W:
        g[nr:nr+6, nc:nc+6] = sprite
    return g

def is_level_complete(grid):
    return False
