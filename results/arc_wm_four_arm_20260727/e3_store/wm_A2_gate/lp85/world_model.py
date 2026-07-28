import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        grid = grid.copy()
        
        # Check bounds
        if px < 0 or px >= w or py < 0 or py >= h:
            return grid
            
        # Check if empty
        if grid[py][px] == 0:
            return grid
            
        # If not empty, it's a toggle action
        # Toggle the cell
        grid[py][px] = 0 if grid[py][px] != 0 else 1
        
        return grid
    else:
        # Directional actions (1-5)
        # These don't affect the grid based on observed data
        return grid

def is_level_complete(grid):
    # Based on the game logic, the level is complete when all targets are collected
    # Since we don't have explicit target markers, we check if the grid is in a stable state
    # or if there are no more movable elements
    # For this game, we consider it complete if there are no non-zero elements left
    # or if the grid matches a specific win pattern
    # Given the limited information, we'll assume completion when the grid is all zeros
    # or when no changes occur in a transition
    return np.all(grid == 0) or np.all(grid == 1)