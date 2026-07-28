import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        
        # Check if the clicked cell is a 1 (empty space)
        if new_grid[py, px] == 1:
            # Determine the direction of movement based on the position of the '3' block
            # The '3' block is typically at the top of the grid
            # We look for the nearest '3' block in the same column as the clicked cell
            # and move it towards the clicked cell
            
            # Find the nearest '3' block in the same column
            col = px
            found_3 = False
            for r in range(h):
                if new_grid[r, col] == 3:
                    found_3 = True
                    break
            
            if found_3:
                # Move the '3' block towards the clicked cell
                # The '3' block is at the top, so we move it down
                # We need to find the extent of the '3' block in the column
                start_r = py
                end_r = py
                for r in range(py, -1, -1):
                    if new_grid[r, col] == 3:
                        start_r = r
                    else:
                        break
                
                for r in range(py, -1, -1):
                    if new_grid[r, col] == 3:
                        end_r = r
                        break
                
                # Move the '3' block down
                for r in range(start_r, end_r + 1):
                    new_grid[r, col] = 1
                    new_grid[r + 1, col] = 3
                
                # Check if the '3' block has reached the bottom
                if new_grid[end_r + 1, col] == 1:
                    new_grid[end_r + 1, col] = 3
                    new_grid[end_r, col] = 1
        
        return new_grid
    else:
        # For other actions, return the grid unchanged
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if all cells are filled with 3s
    return np.all(grid == 3)