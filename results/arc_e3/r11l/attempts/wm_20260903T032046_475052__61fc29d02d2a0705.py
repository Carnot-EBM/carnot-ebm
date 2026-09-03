import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or not isinstance(data, dict):
        return g
    x = int(data.get('x', 0)); y = int(data.get('y', 0))
    # progress bar in column 0: fills top->bottom, wraps at 18
    k = int(np.sum(g[:, 0] == 5))
    if k >= 18:
        k = 0
    g[k, 0] = 5
    # click toggles the 3x3 diamond (centered on click) between 5 and 3
    r, c = y, x
    if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
        cur = g[r, c]
        if cur == 5:
            new = 3
        elif cur == 3:
            new = 5
        else:
            new = cur
        if new != cur:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                        if g[rr, cc] in (5, 3):
                            g[rr, cc] = new
    return g

def is_level_complete(grid):
    return False