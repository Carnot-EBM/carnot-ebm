import numpy as np

def _find_player(grid):
    # player = 3x3 block of color 9 with a color-4 center
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if grid[r, c] == 9 and grid[r + 1, c + 1] == 4 and grid[r + 2, c] == 9:
                return (r, c)
    return None

def _step(grid, dr, dc):
    p = _find_player(grid)
    if p is None:
        return grid
    r, c = p
    nr, nc = r + dr, c + dc
    if nr < 0 or nc < 0 or nr + 2 >= grid.shape[0] or nc + 2 >= grid.shape[1]:
        return grid
    out = grid.copy()
    # clear old
    out[r:r + 3, c:c + 3] = 0
    # draw new
    out[nr:nr + 3, nc:nc + 3] = 9
    out[nr + 1, nc + 1] = 4
    return out

def engine(grid, action, data):
    grid = np.array(grid, dtype=int, copy=True)
    if action == 2:
        return _step(grid, 6, 0)
    elif action == 3:
        return _step(grid, 0, -6)
    elif action == 4:
        return _step(grid, 0, 6)
    elif action == 1:
        return _step(grid, -6, 0)
    return grid

def is_level_complete(grid):
    return False
