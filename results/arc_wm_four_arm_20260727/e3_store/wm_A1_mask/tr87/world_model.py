import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state given the current grid, action, and action data.
    """
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        # Action 3: Toggle specific cells (0 <-> 10)
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical coordinates
        logical_x = px // 1
        logical_y = py // 1
        
        if 0 <= logical_y < H and 0 <= logical_x < W:
            current_val = new_grid[logical_y, logical_x]
            if current_val == 0:
                new_grid[logical_y, logical_x] = 10
            elif current_val == 10:
                new_grid[logical_y, logical_x] = 0
                
    elif action == 2:
        # Action 2: Move right (pushing blocks)
        # Find all non-background blocks (value != 0)
        blocks = np.where(new_grid != 0, 1, 0)
        
        # Iterate from right to left to push blocks
        for col in range(W - 1, -1, -1):
            for row in range(H):
                if new_grid[row, col] != 0:
                    # Try to move this block to the right
                    if col + 1 < W and new_grid[row, col + 1] == 0:
                        new_grid[row, col] = 0
                        new_grid[row, col + 1] = new_grid[row, col]
                    elif col + 1 < W and new_grid[row, col + 1] != 0:
                        # If blocked, try to push the block at col+1 further
                        # This is a recursive push logic
                        pass
        
        # Handle wrapping or edge cases if necessary
        # For this simple model, we just push right as far as possible
        
    return new_grid

def is_level_complete(grid):
    """
    Checks if the grid represents a win state.
    """
    # Check if all non-background cells are collected or arranged in a specific pattern
    # Based on the observed transitions, the win state seems to involve collecting all blocks
    # or reaching a specific configuration.
    # A simple heuristic: check if the grid is empty or has a specific pattern.
    # Given the complexity, we'll assume the win state is when all blocks are collected.
    
    # Count non-zero cells
    non_zero_count = np.count_nonzero(grid)
    
    # If the grid is empty or has a specific pattern, it's complete
    # For now, we'll return True if the grid is empty (all blocks collected)
    return non_zero_count == 0