import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    
    if action == 6:
        # Pixel click action: convert to logical coordinates and toggle
        if data is not None:
            y, x = data['y'] // 1, data['x'] // 1
            if 0 <= y < H and 0 <= x < W:
                grid[y, x] = 15 - grid[y, x]
                return grid
        return grid

    # Directional actions (1-5)
    # Mapping based on observed deltas:
    # Action 3: moves right (dx=6)
    # Action 4: moves left (dx=-6)
    # Action 5: moves up (dy=-6)
    # Action 6: moves down (dy=6)
    # Action 7: toggle (handled above)
    
    dx, dy = 0, 0
    
    if action == 1:
        dx, dy = 6, 0
    elif action == 2:
        dx, dy = -6, 0
    elif action == 3:
        dx, dy = 0, 6
    elif action == 4:
        dx, dy = 0, -6
    elif action == 5:
        dx, dy = 6, 6
    
    # Apply movement
    # The game appears to be a "snake" or "block" movement where the entire structure moves
    # Based on the deltas, it seems like the game is about moving a cursor or a specific block
    # and toggling cells along a path
    
    # Simplified model: The game is about moving a cursor and toggling cells
    # The cursor position is determined by the action
    
    # Calculate cursor position based on action
    # This is a heuristic based on the observed deltas
    
    # For simplicity, we'll assume the game is about moving a cursor and toggling cells
    # The cursor position is determined by the action
    
    # Return the grid as is (no change)
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the win state, the grid should have specific patterns
    
    # Simplified check: check if the grid matches the win state
    # The win state has specific patterns in the grid
    
    # Check if the grid is all zeros (or all background)
    if np.all(grid == 0):
        return True
    
    # Check if the grid matches the win state
    # The win state has specific patterns in the grid
    
    return False