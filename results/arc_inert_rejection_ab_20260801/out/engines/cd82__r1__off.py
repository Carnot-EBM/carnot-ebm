import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where objects (color 2) move and interact with walls/obstacles.
    # Based on the transitions, ACTION4 is likely 'down' or similar, and ACTION2 is 'up'.
    # However, the actual movements are complex and potentially involve gravity or sliding.
    #
    # Let's analyze the same-action repeated calls.
    # In the transition logs, we own a few sequences of ACTION2.
    # ACTION2 is called twice.
    #//
    # Given the limited data, it's a<|channel>thoughtt process.
    # We need to actually implement the logic based on the observed deltas.
    #
    # Looking at the delta for ACTION4:
    # r21c39:2x1, r22c38:2x3... this looks like a diagonal movement or shift of color 2 blocks.
    #
    # Looking at the delta for ACTION2:
    # r21c39:5x1, r22c38:5x3... these cells were color 2 and now they are color 5 (background).
    # This suggests that ACTION2 "clears" or "moves" the block of color 2.
    #
    # Since I cannot deduce the exact physics engine from three transitions, 
    # I will implement a simple rule: if action is 6 (click), do nothing. 
    # If action is 2, move existing color 2 blocks in some direction.
    # If action is 4, move them in another.
    #
    # Actually, looking closer at the INITIAL GRID:
    # Color 0: Empty/Void
    # Color 2: Player/Object
    # Color 3: Wall/Obstacle
    # Color 4: Boundary/Wall
    # Color 5: Background
    # Color 15: Goal/Target
    #
    # In ACTION4:
    # Cells change to 2. This means object 2 is moving into new positions.
    # In ACTION2:
    # Cells change to 5. This means object 2 is leaving those positions.
    #
    # Let's try to model this as a sliding puzzle where color 2 moves until it hits a wall (color 3 or 4).
    #
    # Action mapping (common ARC): 1=up, 2=down, 3=left, 4=right? Or 2=up, 4=down?
    # Looking at r21c39 -> r22c38... that's row+1, col-1.
    #
    # Given the constraints and the nature of these tasks, I will implement a basic movement logic for color 2 blocks.

    new_grid = grid.copy()
    
    # Find all cells with color 2
    objs = np.argwhere(grid == 2)
    if len(objs) == 0:
        return new_grid

    # Define directions based on action
    # Based on observed deltas:
    # ACTION4 seems to move things "down-ish" or shift them.
    # ACTION2 seems to move things "up-ish" or clear them.
    # Wait, looking at the delta for ACTION2 again:
    # r21c39:5x1 means cell (21, 39) becomes 5. It was 2.
    # This is a standard "move object" transition.
    
    directions = {
        2: (-1, 0), # Up
        4: (1, 0),  # Down
        1: (0, -1), # Left
        3: (0, 1),  # Right
    }
    
    if action not in directions:
        return new_grid
        
    dr, dc = directions[action]
    
    # To prevent objects from overlapping or moving through walls, we process them carefully.
    # For simplicity, we'll just shift all color 2s and check boundaries/walls.
    
    moved_objs = []
    for r, c in objs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr, nc] == 5: # Only move into background
                new_grid[r, c] = 5
                new_grid[nr, nc] = 2
                moved_objs.append((nr, nc))
            else:
                moved_objs.append((r, c))
        else:
            moved_objs.append((r, c))
            
    return new_grid

def is_level_complete(grid):
    # Usually complete if object 2 is on goal 15 or no more goals exist.
    # In the initial grid, there are many 15s.
    # Let's assume it's complete if any 2 is on a 15 (though my engine doesn't allow that).
    # Or simply return False as we don't have a win state example.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is based on the grid containing only 
    the target color (color 2) and the background color (color 0).
    """
    grid = np.array(grid)
    # The win condition is that all non-zero cells must be color 2.
    # This is a a simple general rule for this specific task.
    return np.all((grid == 0) | (grid == 2))
