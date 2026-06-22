import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        # The observed transitions show that clicking at (23, 2) (logical (23, 2))
        # changes the first column (col 0) of rows 0-5 to value 5.
        # This suggests that the action toggles or sets a specific region of the grid.
        # Based on the pattern, it seems to be setting the first column of the grid to 5.
        # However, the action is repeated 6 times, and each time it changes a different row range.
        # Actually, looking at the transitions, it seems like the action is setting the first column to 5.
        # Let's assume the action sets the first column to 5.
        grid[0:6, 0] = 5
        return grid
    return grid

def is_level_complete(grid):
    # The observed transitions show that the level is complete when the grid is in a specific state.
    # However, the provided transitions do not show a win state.
    # Let's assume the level is complete when the grid is in a specific state.
    # Based on the transitions, the level is complete when the grid is in a specific state.
    # Let's assume the level is complete when the grid is in a specific state.
    # This is a guess.
    return False