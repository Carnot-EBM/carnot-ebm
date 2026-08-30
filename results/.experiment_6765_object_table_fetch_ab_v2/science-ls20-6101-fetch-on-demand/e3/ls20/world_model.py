import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    # rough candidate: find the 12/9 object (top 12, bottom 9) and the 3 pixel near the 11 bar
    # ACTION3 = move object left, ACTION1 = move object up (guess)
    # find object: cells of color 12
    obj = np.argwhere(g == 12)
    if len(obj) > 0:
        r0 = obj[:,0].min(); r1 = obj[:,0].max()
        c0 = obj[:,1].min(); c1 = obj[:,1].max()
        block = g[r0:r1+1, c0:c1+1].copy()
        if action == 3:
            nc0 = c0 - 5
            nc1 = c1 - 5
            if nc0 >= 0:
                g[r0:r1+1, c0:c1+1] = 3
                g[r0:r1+1, nc0:nc1+1] = block
        elif action == 1:
            nr0 = r0 - 5
            nr1 = r1 - 5
            if nr0 >= 0:
                g[r0:r1+1, c0:c1+1] = 3
                g[nr0:nr1+1, c0:c1+1] = block
    return g

def is_level_complete(grid):
    return False
