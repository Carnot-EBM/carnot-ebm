import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    if action != 6 or not data:
        return g
    r = int(data.get('y', 0)); c = int(data.get('x', 0))
    H, W = g.shape
    if not (0 <= r < H and 0 <= c < W):
        return g
    # rough: dissolve a 15-diamond centered at click, spawn new one down-right, trail of 5s up-left
    # plus shape offsets
    plus = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,-1),(1,-1),(-1,1)]
    # find 15 diamond centered at (r,c)
    if g[r,c] == 6:
        for dr,dc in plus:
            rr,cc = r+dr,c+dc
            if 0<=rr<H and 0<=cc<W and g[rr,cc]==15:
                g[rr,cc]=0
        g[r,c]=15
        # spawn new diamond down-right
        nr,nc = r+6,c+5
        for dr,dc in plus:
            rr,cc = nr+dr,nc+dc
            if 0<=rr<H and 0<=cc<W:
                g[rr,cc]=15
        g[nr,nc]=6
    return g

def is_level_complete(grid):
    return False
