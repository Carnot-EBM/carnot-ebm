import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 0:
        if data is None:
            # Action 0 with no data: do nothing
            return new_grid
        
        # Action 0 with data: toggle specific cells
        # The data is a dictionary with 'x' and 'y' keys
        if 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Toggle the cell at (py, px)
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 1 - new_grid[py, px]
            return new_grid
        
        # If data is not a dict with x/y, treat as no-op
        return new_grid
        
    elif action == 1:
        # Action 1: Move left
        # Shift all non-background cells left
        for r in range(H):
            row = new_grid[r, :]
            # Find non-zero cells
            non_zero = row[row != 0]
            # Shift left
            new_row = np.zeros(W, dtype=int)
            new_row[:len(non_zero)] = non_zero
            new_grid[r, :] = new_row
        return new_grid
        
    elif action == 2:
        # Action 2: Move right
        # Shift all non-background cells right
        for r in range(H):
            row = new_grid[r, :]
            non_zero = row[row != 0]
            new_row = np.zeros(W, dtype=int)
            new_row[-len(non_zero):] = non_zero
            new_grid[r, :] = new_row
        return new_grid
        
    elif action == 3:
        # Action 3: Move up
        # Shift all non-background cells up
        for c in range(W):
            col = new_grid[:, c]
            non_zero = col[col != 0]
            new_col = np.zeros(H, dtype=int)
            new_col[:len(non_zero)] = non_zero
            new_grid[:, c] = new_col
        return new_grid
        
    elif action == 4:
        # Action 4: Move down
        # Shift all non-background cells down
        for c in range(W):
            col = new_grid[:, c]
            non_zero = col[col != 0]
            new_col = np.zeros(H, dtype=int)
            new_col[-len(non_zero):] = non_zero
            new_grid[:, c] = new_col
        return new_grid
        
    elif action == 5:
        # Action 5: Rotate clockwise
        # Transpose and reverse rows
        new_grid = np.rot90(grid, k=-1)
        return new_grid
        
    elif action == 6:
        # Action 6: Rotate counter-clockwise
        # Transpose and reverse columns
        new_grid = np.rot90(grid, k=1)
        return new_grid
        
    elif action == 7:
        # Action 7: Flip horizontally
        new_grid = np.fliplr(grid)
        return new_grid
        
    return new_grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the grid
    # Based on the observed transitions, the win state has:
    # - A large connected component of color 3
    # - Specific patterns in other colors
    # For simplicity, we check if the grid matches the win state pattern
    # by checking if the grid has the same structure as the win state
    
    # Check if the grid has the same number of non-zero cells as the win state
    # This is a simplified check
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    return np.all(grid == 0)
