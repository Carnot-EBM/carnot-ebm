import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 4:
        # move the 9-snake (with its 0 hole) right by 3
        mask9 = (g == 9)
        mask0 = (g == 0)
        g[mask9] = 5
        g[mask0] = 5
        if W >= 3:
            g[:, 3:] = np.where(mask9[:, :-3], 9, g[:, 3:])
            g[:, 3:] = np.where(mask0[:, :-3], 0, g[:, 3:])
    elif action == 5:
        # fill the 0 hole with 9
        g[g == 0] = 9
    elif action == 1:
        # move the 11 wall up by 3
        mask11 = (g == 11)
        g[mask11] = 5
        if H >= 3:
            g[3:, :] = np.where(mask11[:-3, :], 11, g[3:, :])
    return g

def is_level_complete(grid):
    return False