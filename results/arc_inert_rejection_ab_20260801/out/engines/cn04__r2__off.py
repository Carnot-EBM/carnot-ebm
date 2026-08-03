import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    The game involves a player-controlled object (likely the 0s) moving through a maze.
    Based on the observed transitions, ACTION4 seems to be 'move right'.
    ACTION4 shifts a set of cells that are value 0 and their surrounding structure.
    Looking at the same delta patterns, it's a shift of the entire 'playable area' or 
    the 'entity' by 3 columns to the right.
    """
    next_grid = grid.copy()
    if action == 4: # Move Right
        # The pattern shows a shift of 3 units to the right for all non-wall (non-10) cells.
        # We need to identify which parts of the grid are shifted.
        # In this case, the transition is also changing some specific pixels in row 0.
        # Row 0 has a wall (10), then a gap/path (4), then a wall (10).
        # Row 0 column 16 is changed to 0, then 17, etc.
        #<|channel>thought
        # Let's refine: only certain regions are shifted.
        # Looking at the deltas, r0c16 becomes 0, r0c17 becomes 0...
        # This suggests a "cursor" or "player" moving across the top path.
        # The rest of the maze shifts as well.
        
        # To implement a general rule:
        # Shift everything that isn't color 10 (the background/wall) 
        # but keep it within the bounds of where paths exist.
        # However, the observed transitions show a very specific movement:
        # A block of size 3xN moves right by 3 columns.
        
        # Simple implementation for ACTION4 based on observations:
        # Find all cells with value != 10 and shift them right by 3.
        # But we must be careful not to overwrite walls.
        # Actually, looking closer at the delta:
        # Cells that were 0 become 10, and cells that were 10 become 0.
        # It looks like the 'empty space' (0) is shifting.
        
        mask = grid != 10
        shifted_mask = np.roll(mask, 3, axis=1)
        
        # We need to preserve the wall structure (color 10).
        # Only change values if they are part of the "playable" area.
        # In this game, the playable area seems to be defined by any cell that is NOT 10.
        # Let's try a simpler approach:
        # The deltas show that blocks of 3-width move right.
        # Specifically, it's as if the entire non-wall layout shifts.
        
        # Create a copy of the current state
        new_grid = grid.copy()
        
        # Identify the "entity" or "path" - everything not color 10
        # Shift these values.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 3):
                if grid[r, c] != 10:
                    # This is a bit complex for a simple engine.
                    # Let's use the observation that ACTION4 moves things right by 3.
                    pass

    # Based on the provided transitions, let's implement the shift logic more directly.
    # It looks like the 'empty space' (0) and other colors (4, 8, 14) are shifting.
    
    # Correct Logic for ACTION4:
    # 1. Find all cells that are NOT 10.
    # 2. Move them 3 units to the right.
    # 3. Fill the vacated spots with 10.
    # 4. The pixels at the very top (row 0) also move.
    
    next_grid = grid.copy()
    if action == 4:
        # We only shift if it doesn't collide with walls? No, the deltas show 
        # the structure itself moving.
        # Actually, looking at r0c16:0x1, then r0c17:0x1...
        # Row 0 has color 4 from col 16 to 47.
        # A single cell of color 0 is moving across this path.
        
        # Let's find the position of the 'player' (the 0 in row 0).
        # If no 0 exists in row 0, we can't move.
        player_pos = np.where(grid[0] == 0)[0]
        if len(player_pos) > 0:
            p_col = player_pos[0]
            # Move player right by 1
            if p_col + 1 < grid.shape[1] and grid[0, p_col+1] != 10:
                next_grid[0, p_col] = 4 # Restore old pos to path color
                next_grid[0, p_col+1] = 0
                
                # Now shift the rest of the maze blocks by 3?
                # The deltas show that for every ACTION4, a block of width 3 shifts.
                # This looks like a "screen scroll" or "maze shift".
                # We will simulate the shift of all non-wall cells by 3 columns.
                
                # To avoid complex logic, let's use a mask.
                non_walls = grid != 10
                shifted_non_walls = np.roll(non_walls, 3, axis=1)
                
                # Only apply shift where it doesn't wrap around (simple approximation)
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        if c >= 3:
                            if grid[r, c-3] != 10:
                                next_grid[r, c] = grid[r, c-3]
                        if c < 3:
                            next_grid[r, c] = 10 # Fill left edge
        else:
            # If no player in row 0, maybe they are elsewhere.
            # But based on transitions, this is the pattern.
            pass

    return next_grid

def is_level_complete(grid):
    # Level complete usually means reaching a goal or clearing something.
    # In these games, often color 14 or a specific position is the goal.
    # Without a win state example, we check for common patterns.
    # Let's assume the level is complete if the 'player' (0) reaches the end of the path.
    player_pos = np.where(grid[0] == 0)[0]
    if len(player_pos) > 0 and player_pos[0] >= grid.shape[1] - 17:
        return True
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for cn04 is that the grid contains only one color
    (excluding background color 0).
    """
    grid = np.array(grid)
    non_zero_colors = np.unique(grid[grid != 0])
    
    # Win if there is exactly one unique non-zero color present.
    return len(non_zero_colors) == 1
