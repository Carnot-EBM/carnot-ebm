import numpy as np

def _find_block(grid, color):
    # find bounding box of all cells of given color
    mask = (grid == color)
    if not mask.any():
        return None
    rs = np.where(mask.any(axis=1))[0]
    cs = np.where(mask.any(axis=0))[0]
    return (rs.min(), rs.max(), cs.min(), cs.max())

def engine(grid, action, data):
    g = grid.copy()
    if action == 4:
        # move the 9-block right by 4
        bb = _find_block(g, 9)
        if bb is not None:
            r0, r1, c0, c1 = bb
            # shift right by 4
            new_c0 = c0 + 4
            new_c1 = c1 + 4
            # clear old
            g[r0:r1+1, c0:c1+1] = 12
            # set new (clipped)
            nc0 = max(0, new_c0)
            nc1 = min(g.shape[1]-1, new_c1)
            if nc1 >= nc0:
                g[r0:r1+1, nc0:nc1+1] = 9
        # erode 14 bar from right by 2
        if (g == 14).any():
            # find rightmost 14 in row 0
            row = 0
            cols = np.where(g[row] == 14)[0]
            if len(cols) >= 2:
                g[row, cols[-2]] = 0
                g[row, cols[-1]] = 0
    return g

def is_level_complete(grid):
    return False
