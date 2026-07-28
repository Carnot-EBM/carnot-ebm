import numpy as np

def engine(grid: np.ndarray, action: int, data: dict) -> np.ndarray:
    if action == 6:
        px = data['x']
        py = data['y']
        grid = grid.copy()
        grid[py, px] = 15
        grid[py, px + 1] = 15
        grid[py, px + 2] = 15
        grid[py + 1, px] = 5
        grid[py + 1, px + 1] = 5
        grid[py + 1, px + 2] = 5
        grid[py + 3, px + 60] = 5
        return grid
    return grid

def is_level_complete(grid: np.ndarray) -> bool:
    if grid[63, 0] != 0:
        return False
    for r in range(64):
        if grid[r, 0] != 5:
            return False
    for r in range(23, 32):
        if grid[r, 0] != 5 or grid[r, 30] != 9 or grid[r, 35] != 5:
            return False
    for r in range(37, 42):
        if grid[r, 0] != 5 or grid[r, 10] != 10 or grid[r, 22] != 5:
            return False
    for r in range(40, 42):
        if grid[r, 0] != 5 or grid[r, 37] != 10 or grid[r, 26] != 5:
            return False
    for r in range(41, 42):
        if grid[r, 0] != 5 or grid[r, 16] != 10 or grid[r, 47] != 5:
            return False
    for r in range(54, 58):
        if grid[r, 0] != 5 or grid[r, 49] != 10 or grid[r, 14] != 5:
            return False
    for r in range(56, 58):
        if grid[r, 0] != 5 or grid[r, 47] != 10 or grid[r, 16] != 5:
            return False
    return True