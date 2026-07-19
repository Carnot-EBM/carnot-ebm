import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 7
            new_grid[py, px + 1] = 7
            new_grid[py, px + 2] = 7
            new_grid[py, px + 3] = 7
        else:
            # Action 3 without data usually doesn't change grid in this context or is a no-op
            pass
            
    elif action == 2:
        # Action 2 is a directional move (down)
        # Based on observed transitions, it moves blocks down
        # We simulate gravity for color 14 blocks
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 14:
                    # Check if we can move down
                    if r + 1 < H and new_grid[r + 1, c] == 1:
                        new_grid[r + 1, c] = 14
                        new_grid[r, c] = 1
                        # Continue moving down from new position
                        while r + 1 < H and new_grid[r + 1, c] == 1:
                            new_grid[r + 1, c] = 14
                            new_grid[r, c] = 1
                            r += 1
                    elif r + 1 < H and new_grid[r + 1, c] == 14:
                        # Already 14, skip
                        pass
                    elif r + 1 < H and new_grid[r + 1, c] == 0:
                        # Move into empty space
                        new_grid[r + 1, c] = 14
                        new_grid[r, c] = 1
                        r += 1
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 14 blocks have reached the bottom
    # Based on the win state pattern, all 14s should be at the bottom
    for r in range(H - 1, -1, -1):
        for c in range(W):
            if grid[r, c] == 14:
                # If a 14 is not at the bottom-most row, the level is not complete
                if r != H - 1:
                    return False
    return True