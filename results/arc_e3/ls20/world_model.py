import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        if data.get('x') is not None:
            px, py = data['x'], data['y']
            h, w = grid.shape
            if 0 <= px < h and 0 <= py < w:
                grid[px, py] = 3
            return grid
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 3:
                    grid[r, c] = 0
        return grid
    return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    if grid.dtype != object:
        return False
    if not np.all([isinstance(c, str) for row in grid for c in row]):
        return False
    if not np.all([len(c) == 1 for row in grid for c in row]):
        return False
    if not np.all([c in '0123456789' for row in grid for c in row]):
        return False
    if np.all([c == '0' for row in grid for c in row]):
        return False
    if np.all([c == '9' for row in grid for c in row]):
        return False
    if np.all([c == '1' for row in grid for c in row]):
        return False
    if np.all([c == '2' for row in grid for c in row]):
        return False
    if np.all([c == '3' for row in grid for c in row]):
        return False
    if np.all([c == '4' for row in grid for c in row]):
        return False
    if np.all([c == '5' for row in grid for c in row]):
        return False
    if np.all([c == '6' for row in grid for c in row]):
        return False
    if np.all([c == '7' for row in grid for c in row]):
        return False
    if np.all([c == '8' for row in grid for c in row]):
        return False
    if np.all([c == '9' for row in grid for c in row]):
        return False
    return True
