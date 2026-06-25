import numpy as np

def engine(grid, action, data):
    """
    Predict the next grid state given the current grid, action, and action data.
    This game is a "toggle" or "swap" puzzle where clicking a cell toggles its state
    and potentially swaps it with adjacent cells of the same color.
    """
    H, W = grid.shape
    new_grid = grid.copy()
    
    # Action 6 is a click action with pixel coordinates
    if action == 6:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical coordinates
        row, col = py // 1, px // 1  # Assuming 1:1 mapping for simplicity
        
        # Check bounds
        if row < 0 or row >= H or col < 0 or col >= W:
            return new_grid
        
        # Toggle the clicked cell
        current_val = new_grid[row, col]
        
        # If the cell is not 0 (empty), toggle it
        if current_val != 0:
            # Toggle logic: swap with adjacent cells of the same color
            # This is a simplified toggle mechanism
            new_grid[row, col] = 0  # Clear the cell
            
            # Check adjacent cells for same color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if new_grid[nr, nc] == current_val:
                        new_grid[nr, nc] = 0  # Clear adjacent cells of the same color
            
            # Fill the cleared cells with the original color
            new_grid[row, col] = current_val
            
            # Check adjacent cells again for same color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if new_grid[nr, nc] == current_val:
                        new_grid[nr, nc] = current_val  # Fill adjacent cells of the same color
        
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    This is a simplified check for a win state.
    """
    # Check if the grid is in a specific win state
    # For now, return False as a placeholder
    return False