import numpy as np

def engine(grid, action, data):
    if action == 7:
        return grid.copy()
    
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        
        # Check if the click is on a 0 cell (empty space)
        if grid[logical_y, logical_x] == 0:
            # Create a new grid with the clicked cell and its neighbors set to 15
            new_grid = grid.copy()
            new_grid[logical_y, logical_x] = 15
            
            # Set neighbors to 15 if they are 0
            if logical_y > 0 and grid[logical_y - 1, logical_x] == 0:
                new_grid[logical_y - 1, logical_x] = 15
            if logical_y < grid.shape[0] - 1 and grid[logical_y + 1, logical_x] == 0:
                new_grid[logical_y + 1, logical_x] = 15
            if logical_x > 0 and grid[logical_y, logical_x - 1] == 0:
                new_grid[logical_y, logical_x - 1] = 15
            if logical_x < grid.shape[1] - 1 and grid[logical_y, logical_x + 1] == 0:
                new_grid[logical_y, logical_y + 1] = 15
                
            return new_grid
        
        return grid.copy()
    
    return grid.copy()

def is_level_complete(grid):
    # Check if the grid is full of 15s
    return np.all(grid == 15)