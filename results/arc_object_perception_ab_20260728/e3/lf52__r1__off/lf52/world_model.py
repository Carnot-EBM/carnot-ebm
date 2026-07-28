import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if r == py and c == px:
                    grid[r, c] = 1
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 10)

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return False
    return np.all(g[1:, 1:] == g[:-1, :-1])
