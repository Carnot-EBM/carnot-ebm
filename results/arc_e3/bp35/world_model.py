import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'] // 1, data['y'] // 1
        # Clicking on a 0 cell toggles it to 15 (black)
        if grid[py, px] == 0:
            grid[py, px] = 15
        else:
            grid[py, px] = 0
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, a win state is not explicitly defined
    # However, we can infer that if the grid has changed significantly, it might be a win
    # For now, return False as a placeholder
    return False