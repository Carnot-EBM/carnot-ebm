import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    if action == 6 and data is not None:
        # click: move player (color 4) left by 4, carrying the 11 part
        # find player cells (color 4)
        ys, xs = np.where(g == 4)
        if len(ys) > 0:
            # shift all 4 and 11 cells left by 4
            # collect the sprite: 4 and 11 cells
            mask = (g == 4) | (g == 11)
            # build new grid
            ng = g.copy()
            # clear old positions
            ng[mask] = 0
            # place new positions shifted left by 4
            ny, nx = ys - 0, xs - 4
            valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
            for r, c, v in zip(ys[valid], nx[valid], g[ys[valid], xs[valid]]):
                ng[r, c] = v
            g = ng
    return g

def is_level_complete(grid):
    return False
