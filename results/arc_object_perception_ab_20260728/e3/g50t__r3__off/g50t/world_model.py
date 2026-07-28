import numpy as np

def engine(grid: np.ndarray, action: int, data: dict) -> np.ndarray:
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 5:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
    elif action == 7:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= px < W and 0 <= py < H:
            new_grid[py, px] = 9
            
    return new_grid

def is_level_complete(grid: np.ndarray) -> bool:
    return np.array_equal(grid, np.zeros((64, 64), dtype=int))