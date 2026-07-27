import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        return new_grid
    
    if action == 2:
        # Action 2: Fill specific regions with color 13 (blue)
        # Based on observed changes, this action fills a large rectangular region
        # and some specific cells with color 0 (black)
        # The pattern suggests filling a large area in the middle
        # and creating a specific shape
        
        # Fill the large region (rows 11-31, cols 11-23) with color 13
        # This is inferred from the changes in rows 11-31
        for r in range(11, 32):
            for c in range(11, 24):
                if new_grid[r, c] != 13:
                    new_grid[r, c] = 13
        
        # Fill specific cells with color 0
        # Based on the changes, specific cells are filled with 0
        # This is inferred from the changes in rows 14-31
        for r in range(14, 32):
            for c in range(14, 24):
                if new_grid[r, c] != 0:
                    new_grid[r, c] = 0
        
        return new_grid
    
    if action == 3:
        # Action 3: Fill specific regions with color 13 (blue)
        # Based on observed changes, this action fills a large rectangular region
        # and some specific cells with color 0 (black)
        # The pattern suggests filling a large area in the middle
        # and creating a specific shape
        
        # Fill the large region (rows 29-49, cols 35-47) with color 13
        # This is inferred from the changes in rows 29-49
        for r in range(29, 50):
            for c in range(35, 48):
                if new_grid[r, c] != 13:
                    new_grid[r, c] = 13
        
        # Fill specific cells with color 0
        # Based on the changes, specific cells are filled with 0
        # This is inferred from the changes in rows 29-49
        for r in range(29, 50):
            for c in range(35, 48):
                if new_grid[r, c] != 0:
                    new_grid[r, c] = 0
        
        return new_grid
    
    if action == 5:
        # Action 5: Fill specific regions with color 13 (blue)
        # Based on observed changes, this action fills a large rectangular region
        # and some specific cells with color 0 (black)
        # The pattern suggests filling a large area in the middle
        # and creating a specific shape
        
        # Fill the large region (rows 14-31, cols 11-26) with color 13
        # This is inferred from the changes in rows 14-31
        for r in range(14, 32):
            for c in range(11, 27):
                if new_grid[r, c] != 13:
                    new_grid[r, c] = 13
        
        # Fill specific cells with color 0
        # Based on the changes, specific cells are filled with 0
        # This is inferred from the changes in rows 14-31
        for r in range(14, 32):
            for c in range(11, 27):
                if new_grid[r, c] != 0:
                    new_grid[r, c] = 0
        
        return new_grid
    
    if action == 6:
        # Action 6: Click action with data
        # Based on observed changes, this action fills a large rectangular region
        # and some specific cells with color 0 (black)
        # The pattern suggests filling a large area in the middle
        # and creating a specific shape
        
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        
        # Fill the large region (rows 14-49, cols 20-47) with color 13
        # This is inferred from the changes in rows 14-49
        for r in range(14, 50):
            for c in range(20, 48):
                if new_grid[r, c] != 13:
                    new_grid[r, c] = 13
        
        # Fill specific cells with color 0
        # Based on the changes, specific cells are filled with 0
        # This is inferred from the changes in rows 14-49
        for r in range(14, 50):
            for c in range(20, 48):
                if new_grid[r, c] != 0:
                    new_grid[r, c] = 0
        
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is a win state
    # Based on observed transitions, the win state is reached when
    # the grid is fully filled with color 13 (blue)
    # and there are no cells with color 0 (black)
    
    # Check if all cells are color 13
    if np.all(grid == 13):
        return True
    
    # Check if all cells are color 0
    if np.all(grid == 0):
        return True
    
    return False