import numpy as np

def engine(grid, action, data):
    g = grid.astype(int).copy()
    H, W = g.shape
    if action != 6 or data is None:
        return g
    px, py = int(data.get('x', 0)), int(data.get('y', 0))
    # Move the 4/11 object left by 4 (rough hypothesis)
    # Identify object: color 4 and 11 components in lower region
    # Simple: find all 4 and 11 cells, shift left by 4
    mask = np.zeros((H, W), dtype=bool)
    mask[g == 4] = True
    mask[g == 11] = True
    # only the lower object (rows > 40)
    mask[:40, :] = False
    newg = g.copy()
    # clear old
    newg[mask] = 0
    # shift left by 4
    shifted = np.zeros_like(mask)
    shifted[:, 4:] = mask[:, :-4]
    # place
    newg[shifted] = g[mask]
    # 3-region boundary: rows 1-27 cols 32-35 -> 3 ; rows 32-43 cols 48-51 -> 0
    newg[1:28, 32:36] = 3
    newg[32:44, 48:52] = 0
    # top-right cell
    newg[0, W-1] = 4
    return newg

def is_level_complete(grid):
    return False
