import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where an object (represented by color 2 or 15) 
    moves through the grid based on directional actions.
    Action mapping inferred from transitions:
    ACTION4: Likely 'Right' or 'Down-Right' movement of a specific entity.
    ACTION2: Likely 'Up' or 'Left' movement of a specific entity.
    
    Looking at the deltas:
    Initial state has blocks of colors. ACTION4 moves something towards the bottom right.
    ACTION2 moves something back up/left.
    The changes are quite complex and involve shifting patterns of colors (2, 15, 5).
    Given the nature of ARC tasks, this looks like a "Sokoban" or "Pathfinding" 
    puzzle where moving a piece modifies the environment or pushes other pieces.
    
    However, without clear simple rules for the delta shifts, we implement a 
    generic transition handler that mimics the observed behavior if possible, 
    but since it's a world model for a specific seed, we focus on the most likely 
    mechanic: a cursor/entity moving and changing cell values along its path.
    """
    new_grid = grid.copy()
    
    # The provided transitions show very specific coordinate updates.
    # Since we need to return a deterministic engine, and the logic is not 
    # immediately obvious as a standard game (like Snake or Sokoban), 
    # we look for the pattern in the coordinates.
    
    if action == 6: # Click
        px, py = data['x'], data['y']
        # Clicks usually toggle or trigger an event at a location
        new_grid[py, px] = 15 if new_grid[py, px] != 15 else 5
        return new_grid

    # For directional actions, we simulate a movement of a 'player' entity.
    # We identify the player as the first instance of color 2 or 15.
    coords = np.argwhere(grid == 2)
    if coords.size == 0:
        coords = np.argwhere(grid == 15)
    
    if coords.size > 0:
        r, c = coords[0]
        dr, dc = 0, 0
        if action == 1: dr, dc = -1, 0  # Up
        elif action == 2: dr, dc = 0, -1 # Left
        elif action == 3: dr, dc = 1, 0  # Down
        elif action == 4: dr, dc = 0, 1  # Right
        elif action == 5: dr, dc = 1, 1  # Down-Right
        
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Move the entity
            new_grid[r, c] = grid[r, c] if grid[r, c] != 2 else 5
            new_grid[nr, nc] = 2
            
    return new_grid

def is_level_complete(grid):
    """
    A level in ARC is usually complete when a specific pattern is reached 
    or an object reaches a target location.
    Based on common patterns, we check if color 2 has reached the bottom right 
    or if certain blocks are cleared.
    """
    # Check for a win condition (e.g., color 2 at the bottom-right corner)
    if grid[-1, -1] == 2:
        return True
    
    # Alternatively, check if all cells of a certain 'obstacle' color are gone
    # or if a specific goal state is achieved.
    return False