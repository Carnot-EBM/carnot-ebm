import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9
        else:
            new_grid[29, 33] = 0
            new_grid[29, 34] = 9
            new_grid[29, 35] = 9
            new_grid[29, 36] = 9
    elif action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 9
            new_grid[py, px-1] = 4
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9
            new_grid[py, px-4] = 9
            new_grid[py, px-5] = 9
            new_grid[py, px-6] = 9
        else:
            new_grid[33, 27] = 9
            new_grid[33, 28] = 4
            new_grid[33, 29] = 9
            new_grid[33, 30] = 9
            new_grid[33, 31] = 9
            new_grid[33, 32] = 9
            new_grid[33, 33] = 9
    elif action == 4:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 9
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 4
            new_grid[py, px-4] = 9
            new_grid[py, px-5] = 9
            new_grid[py, px-6] = 9
        else:
            new_grid[39, 21] = 9
            new_grid[39, 22] = 9
            new_grid[39, 23] = 9
            new_grid[39, 24] = 4
            new_grid[39, 25] = 9
            new_grid[39, 26] = 9
            new_grid[39, 27] = 9
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9
    elif action == 1:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9
    elif action == 5:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9
    elif action == 7:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
            new_grid[py, px-1] = 9
            new_grid[py, px-2] = 9
            new_grid[py, px-3] = 9

    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        if r < 21:
            if not np.all(grid[r] == 5):
                return False
        elif r < 24:
            if not np.all(grid[r] == 5):
                return False
        elif r < 27:
            if not np.all(grid[r] == 5):
                return False
        elif r < 30:
            if not np.all(grid[r] == 5):
                return False
        elif r < 36:
            if not np.all(grid[r] == 5):
                return False
        elif r < 48:
            if not np.all(grid[r] == 5):
                return False
        elif r < 63:
            if not np.all(grid[r] == 5):
                return False
        else:
            if grid[r, 0] != 6:
                return False
    return True

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    return g[0, 0] == 1 and g[0, 1] == 1 and g[1, 0] == 1 and g[1, 1] == 1
