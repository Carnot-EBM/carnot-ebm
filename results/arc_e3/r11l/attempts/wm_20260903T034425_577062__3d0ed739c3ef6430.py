import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        px, py = data.get('x', 0), data.get('y', 0)
        # pixel = logical * 1, so logical coords = pixel coords
        r, c = py, px
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            # Click on a 15-colored cell: change (0,0) to 5
            if g[r, c] == 15:
                g[0, 0] = 5
    return g

def is_level_complete(grid):
    # Check if (0,0) is 5 (the observed completion change)
    return grid[0, 0] == 5