import numpy as np

def _find_player(grid):
    # player = 3x3 block of color 9
    H, W = grid.shape
    for r in range(H - 2):
        for c in range(W - 2):
            block = grid[r:r+3, c:c+3]
            if np.all(block == 9):
                return (r, c)
    return None

def _move_player(grid, dr, dc):
    p = _find_player(grid)
    if p is None:
        return grid
    r0, c0 = p
    r1, c1 = r0 + dr, c0 + dc
    H, W = grid.shape
    if r1 < 0 or c1 < 0 or r1 + 2 >= H or c1 + 2 >= W:
        return grid
    new = grid.copy()
    # clear old
    new[r0:r0+3, c0:c0+3] = 0
    # copy new
    new[r1:r1+3, c1:c1+3] = grid[r0:r0+3, c0:c0+3]
    return new

def _deplete_bar(grid, n):
    new = grid.copy()
    H, W = grid.shape
    row = H - 1
    # deplete from the right
    for i in range(n):
        c = W - 1 - i
        if c >= 0:
            new[row, c] = 0
    return new

def engine(grid, action, data):
    grid = np.array(grid, dtype=int)
    new = grid.copy()
    if action == 1:
        new = _move_player(new, -6, 0)
    elif action == 2:
        new = _move_player(new, 6, 0)
    elif action == 3:
        new = _move_player(new, 0, -6)
    elif action == 4:
        new = _move_player(new, 0, 6)
    # deplete bar by 1
    new = _deplete_bar(new, 1)
    return new

def is_level_complete(grid):
    return False
