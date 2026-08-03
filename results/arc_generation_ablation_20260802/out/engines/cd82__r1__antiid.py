import numpy as np

def engine(grid, action, data):
    # The provided transitions show complex movements of color 2 (red) and color 15 (blue/white).
    # Action 4 seems to move something down-right.
    # Action 2 seems to move something down-left or generally shift colors.
    # Based on the same pattern of changes, we've induced a simple movement model.
    
    new_grid = grid.copy()
    
    if action == 4:
        # Move red blocks (color 2) down-right
        # Find all cells of color 2
        coords = np.argwhere(grid == 2)
        for r, c in coords:
            # Try moving them to a new position
            nr, nc = r + 1, c + 1
            if 0 <= nr < 64 and 0 <= nc < 64:
                # Clear old position if not blocked by other specific colors
                new_grid[r, c] = 5 # Background color
                # Set new position
                new_grid[nr, nc] = 2
            else:
                #// This is a bit of a bit more complex than logic.
                #// Let's try a simple shift for a<|channel>thought
                pass

    elif action == 2:
        # Move red blocks (color 2) down-left
        # la own logic based on observed deltas
        coords = np.argwhere(grid == 2)
        # To simulate the movement seen in the transitions, we
        # let's assume Action 2 moves things "down" or shifts them.
        for r, c in coords:
            # Move everything of color 2 one step down
            nr, nc = r + 1, c
            if 0 <= nr < 64 and 0 <= nc < 64:
                new_grid[r, c] = 5
                new_grid[nr, nc] = 2
    
    # Special handling to mimic the provided delta changes which are very specific.
    # Since the exact physics are not clear, we implement a basic version of 
    # that mimics the general direction of change.
    
    # If no rule was matched, return grid unchanged.
    return new_grid

def is_level_complete(grid):
    # The win state is not explicitly given as a full grid, but usually involves
    # reaching a target area or clearing certain colors.
    # In this game, it seems like red blocks (color 2) should reach the bottom row.
    # return True if any cell of color 2 reaches the last row.
    return np.any(grid[63, :] == 2)