import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        
        # Check if the clicked cell is a 1 (background)
        if new_grid[py, px] == 1:
            # Find the nearest 3 or 11 in the same row
            row = py
            col = px
            
            # Search left for 3 or 11
            for c in range(col - 1, -1, -1):
                if new_grid[row, c] in [3, 11]:
                    # Found a 3 or 11, check if it's a 11
                    if new_grid[row, c] == 11:
                        # Move the 11 to the clicked position
                        new_grid[row, c] = 1
                        new_grid[row, px] = 11
                    else:
                        # Move the 3 to the clicked position
                        new_grid[row, c] = 1
                        new_grid[row, px] = 3
                    break
            
            # Search right for 3 or 11
            for c in range(col + 1, new_grid.shape[1]):
                if new_grid[row, c] in [3, 11]:
                    # Found a 3 or 11, check if it's a 11
                    if new_grid[row, c] == 11:
                        # Move the 11 to the clicked position
                        new_grid[row, c] = 1
                        new_grid[row, px] = 11
                    else:
                        # Move the 3 to the clicked position
                        new_grid[row, c] = 1
                        new_grid[row, px] = 3
                    break
        
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    # Check if there are any 11s left in the grid
    if np.any(grid == 11):
        return False
    return True