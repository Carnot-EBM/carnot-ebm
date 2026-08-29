import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    if action == 4:
        # find the 0 center of the 9 cross
        z = np.argwhere(g == 0)
        if len(z) == 1:
            r, c = int(z[0][0]), int(z[0][1])
            # move vertical 9 line (col c) to col c+3, and 0 to (r, c+3)
            # horizontal line row r shifts right by 3
            # vertical line: all 9 in column c (excluding center)
            for rr in range(H):
                if rr != r and g[rr, c] == 9:
                    g[rr, c] = 5
                    if c + 3 < W:
                        g[rr, c + 3] = 9
            # horizontal line: all 9 in row r (excluding center)
            for cc in range(W):
                if cc != c and g[r, cc] == 9:
                    g[r, cc] = 5
                    if cc + 3 < W:
                        g[r, cc + 3] = 9
            # move center 0
            g[r, c] = 5
            if c + 3 < W:
                g[r, c + 3] = 0
    elif action == 1:
        # move 11 structure up by 3
        mask = (g == 11)
        ys, xs = np.where(mask)
        if len(ys) > 0:
            newg = g.copy()
            for rr, cc in zip(ys, xs):
                newg[rr, cc] = 5
            for rr, cc in zip(ys, xs):
                if rr - 3 >= 0:
                    newg[rr - 3, cc] = 11
            g = newg
    return g

def is_level_complete(grid):
    return False
