import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 6 and data is not None:
        px, py = int(data.get('x', 0)), int(data.get('y', 0))
        # logical coords
        r, c = py, px
        # find the 5x5 block of 15s (with 6 center) whose center is clicked
        # search for a 5x5 region of mostly 15 with a 6 in center near (r,c)
        target = None
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                cr, cc = r + dr, c + dc
                if 2 <= cr < H - 2 and 2 <= cc < W - 2:
                    sub = g[cr-2:cr+3, cc-2:cc+3]
                    if sub[2, 2] == 6 and (sub == 15).sum() >= 20:
                        target = (cr, cc)
                        break
            if target:
                break
        if target:
            cr, cc = target
            # move block down-right by (6,5)
            nr, nc = cr + 6, cc + 5
            if 2 <= nr < H - 2 and 2 <= nc < W - 2:
                block = g[cr-2:cr+3, cc-2:cc+3].copy()
                # clear old
                g[cr-2:cr+3, cc-2:cc+3] = 5
                # place new
                g[nr-2:nr+3, nc-2:nc+3] = block
    return g

def is_level_complete(grid):
    return False
