import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where objects move 
    # and potentially change colors or interact with walls/boundaries.
    # The actions are directional movements of an agent/object.
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right.
    # ACTION6: Click.
    
    # In these specific observations, we only see ACTION2 (Down) and ACTION4 (Right).
    # ACTION4 moves something to the right, and it seems to actually "push" or "move" 
    # a set of pixels that were color 2 (the object) and color 15 (the target/path).
    # The delta changes suggest a movement of blocks of pixels.
    # Identify the 'player' or 'active' object. Color 2 is often the player in these games.
    #
    # Let's refine the logic based on the<|channel>thought process:
    # This appears to be a Sokoban-like or block-pushing game.
    # We need to find the position of the active entity (color 2).
    #
    # Since we can't induce a full general physics engine from just a few frames,
    # and the same action repeated leads to different results, the state depends on the grid.
    #
    # Given the constraints and the observed deltas, let's implement a simple
    # translation of all cells of color 2 and 15 if they are pushed by the agent.
    
    # Finding the coordinates of the 'agent' (color 2).
    # There might be multiple objects of color 2.
    # Let's assume the most bottom-right one is the agent.
    #
    # However, the observations show that ACTION4 (Right) moves things rightward,
    # # and ACTION2 (Down) moves things downward.
    #
    # The changes in r21c39:2x1 etc. suggest a shift.
    #
    # Let's try a simpler approach: identify all blocks of non-background (non-5) pixels.
    # If an action is taken, move those blocks in the direction of the action if possible.
    #
    # For this specific game 'cd82', it looks like the object being moved is a block of colors [0, 2, 3, 4, 15].
    # Background is 5.
    #
    # Let's refine the movement logic based on the observed deltas.
    # ACTION4 shifts everything to the right? No, only certain parts.
    #
    # Actually, looking at the INITIAL grid and the first delta for ACTION4:
    # Initial state has some structures. Action 4 moves something from column 37/38 area to 39+.
    # It seems color 2 is moving.
    #
    # Let's implement a simple "move all color 2 cells" logic as a baseline.
    # But wait, the deltas show that not just color 2 moves, but other colors too.
    #
    # Given the complexity and limited data, let's assume a simplified model where we shift 
    # the entire set of non-background (color != 5) pixels in the direction of the action, 
    # provided they are connected to the agent or part of a movable structure.
    
    # Direction mapping
    dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action not in dirs:
        return grid.copy()

    dr, dc = dirs[action]
    new_grid = grid.copy()
    
    # Identify all 'movable' objects (non-background).
    # In this game, background is 5.
    # We move everything that isn't 5.
    # To avoid infinite shifting, we only move things if there's space (a 5) at the destination.
    
    # This is a very rough approximation. Let's try to be more specific.
    # The deltas show that when ACTION4 is called, cells change from some value to another.
    # For example, r21c39 becomes 2x1. It was probably 5 before.
    # And something else became 5.
    
    # Let's implement a simple "shift non-5 pixels" logic.
    # We iterate backwards for Right/Down and forwards for Left/Up to avoid overwriting.
    
    rows, cols = grid.shape
    if action == 4: # Right
        for r in range(rows):
            for c in range(cols - 2, -1, -1):
                if grid[r, c] != 5 and grid[r, c+1] == 5:
                    new_grid[r, c+1] = grid[r, c]
                    new_grid[r, c] = 5
    elif action == 2: # Down
        for r in range(rows - 2, -1, -1):
            for c in range(cols):
                if grid[r, c] != 5 and grid[r+1, c] == 5:
                    new_grid[r+1, c] = grid[r, c]
                    new_grid[r, c] = 5
    elif action == 1: # Up
        for r in range(1, rows):
            for c in range(cols):
                if grid[r, c] != 5 and grid[r-1, c] == 5:
                    new_grid[r-1, c] = grid[r, c]
                    new_grid[r, c] = 5
    elif action == 3: # Left
        for r in range(rows):
            for c in range(1, cols):
                if grid[r, c] != 5 and grid[r, c-1] == 5:
                    new_grid[r, c-1] = grid[r, c]
                    new_grid[r, c] = 5
                    
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves moving an object to a target.
    # In many ARC games, the level is complete when certain colors are gone or aligned.
    # Based on common patterns, let's check if any color 2 (agent) has reached a specific area.
    # Or simply return False as we don't have a WIN STATE example.
    return False