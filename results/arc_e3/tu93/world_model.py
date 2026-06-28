import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Clear bottom-right corner (63x63) and nearby cells
        # Based on observed changes: r63c63:0, r63c61:0,0, r63c60:0, r63c59:0, r63c58:0, r63c56:0,0
        # This appears to be a pattern of clearing the bottom row
        grid[63, :] = 0
    elif action == 2:
        # Action 2: Clear bottom row
        grid[63, :] = 0
    elif action == 3:
        # Action 3: Clear bottom row
        grid[63, :] = 0
    elif action == 4:
        # Action 4: Clear bottom row
        grid[63, :] = 0
    elif action == 5:
        # Action 5: Clear bottom row
        grid[63, :] = 0
    elif action == 6:
        # Action 6: Click at pixel coordinates (px, py)
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Convert pixel to logical coordinates
            row, col = py // 1, px // 1
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                grid[row, col] = 0
    elif action == 7:
        # Action 7: Clear bottom row
        grid[63, :] = 0
    
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the initial grid, a win state would have all cells cleared
    # or a specific pattern. Since we don't have explicit win state data,
    # we check if the grid is all zeros (fully cleared)
    return np.all(grid == 0)