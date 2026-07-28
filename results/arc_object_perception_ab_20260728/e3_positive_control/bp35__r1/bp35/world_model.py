import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        # Action 3: Move right
        # Find the player (color 15)
        player_pos = np.argwhere(new_grid == 15)
        if len(player_pos) == 0:
            return new_grid
            
        # Get the rightmost player
        player_row = player_pos[-1, 0]
        player_col = player_pos[-1, 1]
        
        # Check if there's space to move right
        if player_col < W - 1 and new_grid[player_row, player_col + 1] == 0:
            # Move player right
            new_grid[player_row, player_col] = 0
            new_grid[player_row, player_col + 1] = 15
            
            # Apply gravity to the right for all rows
            for row in range(H):
                row_data = new_grid[row, :].copy()
                # Find all non-zero elements
                non_zero = row_data[row_data != 0]
                if len(non_zero) > 0:
                    # Shift all non-zero elements to the right
                    # Find the rightmost non-zero element
                    last_non_zero_idx = np.where(row_data != 0)[0][-1]
                    # Shift everything to the right
                    new_row = np.zeros(W, dtype=int)
                    for i, val in enumerate(non_zero):
                        new_row[last_non_zero_idx - len(non_zero) + i + 1] = val
                    new_grid[row, :] = new_row
        return new_grid
    
    elif action == 4:
        # Action 4: Move left
        player_pos = np.argwhere(new_grid == 15)
        if len(player_pos) == 0:
            return new_grid
            
        player_row = player_pos[0, 0]
        player_col = player_pos[0, 1]
        
        if player_col > 0 and new_grid[player_row, player_col - 1] == 0:
            new_grid[player_row, player_col] = 0
            new_grid[player_row, player_col - 1] = 15
            
            # Apply gravity to the left for all rows
            for row in range(H):
                row_data = new_grid[row, :].copy()
                non_zero = row_data[row_data != 0]
                if len(non_zero) > 0:
                    first_non_zero_idx = np.where(row_data != 0)[0][0]
                    new_row = np.zeros(W, dtype=int)
                    for i, val in enumerate(non_zero):
                        new_row[first_non_zero_idx + i] = val
                    new_grid[row, :] = new_row
        return new_grid
    
    elif action == 6:
        # Action 6: Click at specific position
        if data is None:
            return new_grid
            
        px, py = data['x'], data['y']
        # Convert pixel to logical
        logical_row = py // 1
        logical_col = px // 1
        
        if 0 <= logical_row < H and 0 <= logical_col < W:
            # Toggle the cell
            if new_grid[logical_row, logical_col] == 0:
                new_grid[logical_row, logical_col] = 15
            else:
                new_grid[logical_row, logical_col] = 0
            
            # Apply gravity in the column
            for col in range(W):
                col_data = new_grid[:, col].copy()
                non_zero = col_data[col_data != 0]
                if len(non_zero) > 0:
                    last_non_zero_idx = np.where(col_data != 0)[0][-1]
                    new_col = np.zeros(H, dtype=int)
                    for i, val in enumerate(non_zero):
                        new_col[last_non_zero_idx - len(non_zero) + i + 1] = val
                    new_grid[:, col] = new_col
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    # Check if all cells are filled (no zeros)
    return np.all(grid != 0)