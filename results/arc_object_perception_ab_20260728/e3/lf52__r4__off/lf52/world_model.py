import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        # Find the nearest 10x block in the same row
        row = py // 10
        col = px // 10
        # Determine the target column based on the action
        if col == 0:
            target_col = 17
        elif col == 1:
            target_col = 29
        elif col == 2:
            target_col = 41
        elif col == 3:
            target_col = 53
        elif col == 4:
            target_col = 65
        else:
            target_col = 17
        
        # Create a new grid
        new_grid = grid.copy()
        
        # Apply the changes based on the action
        # This is a simplified version of the observed transitions
        # In a real scenario, this would be more complex
        # For now, we just toggle the cells around the target column
        for r in range(H):
            for c in range(W):
                if r == row and c == target_col:
                    new_grid[r, c] = 5
                elif r == row and c == target_col + 1:
                    new_grid[r, c] = 5
                elif r == row and c == target_col - 1:
                    new_grid[r, c] = 5
        
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid matches the win state pattern
    # This is a simplified version of the observed win state
    # In a real scenario, this would be more complex
    # For now, we just check if the grid has the correct number of 5s
    count_5 = np.sum(grid == 5)
    return count_5 > 1000