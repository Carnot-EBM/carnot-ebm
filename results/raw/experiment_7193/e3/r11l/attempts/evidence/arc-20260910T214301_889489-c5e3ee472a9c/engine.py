import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or data is None:
        return g
    px, py = int(data['x']), int(data['y'])
    c, r = px, py
    if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
        return g
    if g[r, c] != 6:
        # click on empty/background: fill column 0 top-down with 5
        for rr in range(g.shape[0]):
            if g[rr, 0] != 5:
                g[rr, 0] = 5
                break
        return g
    # move the 5x5 fat-diamond (15s + center 6) by (+1,+1) if all target cells are empty (5)
    offs = [(-2,-1),(-2,0),(-2,1),
            (-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
            (0,-2),(0,-1),(0,1),(0,2),
            (1,-2),(1,-1),(1,0),(1,1),(1,2),
            (2,-1),(2,0),(2,1)]
    ok = True
    for dr, dc in offs:
        rr, cc = r+dr, c+dc
        if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]) or g[rr, cc] != 5:
            ok = False
            break
    if ok:
        for dr, dc in offs:
            g[r+dr, c+dc] = 15
        g[r+1, c+1] = 6
        g[r, c] = 5
    return g

def is_level_complete(grid):
    return False