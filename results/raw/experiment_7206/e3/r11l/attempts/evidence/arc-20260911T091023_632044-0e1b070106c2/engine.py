import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action != 6 or data is None:
        return g
    px, py = data['x'], data['y']
    # Find the magenta (6) pixel
    mag = np.argwhere(g == 6)
    if len(mag) == 0:
        return g
    mr, mc = mag[0]
    # Move magenta toward click direction with decreasing step
    dr = np.sign(py - mr)
    dc = np.sign(px - mc)
    # Determine step size based on distance
    dist = max(abs(py - mr), abs(px - mc))
    step = max(1, dist // 3)
    nr = mr + dr * step
    nc = mc + dc * step
    nr = np.clip(nr, 0, g.shape[0]-1)
    nc = np.clip(nc, 0, g.shape[1]-1)
    # Leave trail: convert cells along path to 5
    r, c = mr, mc
    while (r, c) != (nr, nc):
        g[r, c] = 5
        r += dr
        c += dc
    g[nr, nc] = 6
    g[mr, mc] = 5
    # Increment column 0 counter
    zeros = np.argwhere(g[:, 0] == 0)
    if len(zeros) > 0:
        g[zeros[0][0], 0] = 5
    return g

def is_level_complete(grid):
    return False