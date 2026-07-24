import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Action 3 is a click that toggles a 3x3 area centered at (px, py)
        # The observed changes show 3x3 blocks being toggled to color 5
        for r in range(max(0, py - 1), min(H, py + 2)):
            for c in range(max(0, px - 1), min(W, px + 2)):
                new_grid[r, c] = 5
                
    elif action == 2:
        if data is None:
            return new_grid
        # Action 2 is a click that toggles a single cell
        # The observed changes show single cells being toggled to color 5
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
    return new_grid

def is_level_complete(grid):
    # Check if the grid is complete based on the win state pattern
    # The win state has specific patterns of colors
    # Based on the initial grid and transitions, the win state is reached when
    # the grid matches a specific configuration
    # Since we don't have explicit win state data, we check for a common pattern
    # In this game, the win state is typically when all cells are filled or match a specific pattern
    # Based on the observed transitions, the win state seems to be when the grid is fully filled with color 5
    # However, looking at the initial grid, it's not fully filled
    # Let's check if the grid matches the win state pattern from the initial grid
    # The win state is likely when the grid is fully filled with color 5
    # But based on the initial grid, it's not fully filled
    # Let's check if the grid is fully filled with color 5
    return np.all(grid == 5)