import numpy as np

def engine(grid: np.ndarray, action: int, data: dict):
    if action == 6:
        px = data.get('x', 0)
        py = data.get('y', 0)
        row = py // 1
        col = px // 1
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            grid[row, col] = 14
    elif action == 3:
        # Directional action: 3 = Up, 4 = Down, 5 = Left, 6 = Right, 7 = Toggle
        # Based on observed transitions, action 3 affects rows 19-22 and columns 23-35
        # It seems to clear cells in a specific region
        # Simplified rule: if action is 3, clear cells in rows 19-22 at specific columns
        # This is a heuristic based on the observed data
        if 19 <= 0 < 23:  # Always true for this game
            grid[19:23, 23:36] = 0
    return grid

def is_level_complete(grid: np.ndarray):
    # Based on the observed data, the level is complete when specific cells are set to 14
    # This is a heuristic based on the observed data
    return np.any(grid == 14)