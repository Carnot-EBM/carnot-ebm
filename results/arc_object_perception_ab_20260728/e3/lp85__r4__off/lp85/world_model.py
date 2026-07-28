import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 0:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 1:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 2:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 3:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 4:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 5:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 6:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    elif action == 7:
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        new_grid[py, px] = 5
        
        # Apply gravity to the right
        for r in range(H):
            row = new_grid[r, :].copy()
            non_zero = row[row != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[r, :] = np.concatenate([non_zero, zeros])
            
        # Apply gravity to the bottom
        for c in range(W):
            col = new_grid[:, c].copy()
            non_zero = col[col != 0]
            zeros = np.zeros(len(non_zero), dtype=int)
            new_grid[:, c] = np.concatenate([non_zero, zeros])
            
        return new_grid
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the grid
    
    # Check for the presence of 14s in the first column
    if not np.all(grid[:, 0] == 14):
        return False
        
    # Check for the presence of 3s in the second column
    if not np.all(grid[:, 1] == 3):
        return False
        
    # Check for the presence of 4s in the rest of the grid
    if not np.all(grid[:, 2:] == 4):
        return False
        
    return True