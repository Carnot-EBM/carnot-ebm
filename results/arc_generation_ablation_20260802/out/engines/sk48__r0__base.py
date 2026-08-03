import numpy as np

def engine(grid, action, data):
    # The game appears to be a puzzle where the same pattern of blocks (color 6)
    # and other elements are shifted or manipulated by directional actions.
    # The grid size is 64x64.
    # Based on the observed transitions, ACTION1-ACTION4 likely correspond to
    # directions (Up, Down, Left, Right).
    # Let's assume standard mapping: ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right.
    # However, looking at the<|channel>thought process for the observations, it
    # let's check if therethought was no thought process provided.
    #
    # Looking at the delta changes in ACTION1 (which changed cells around row 18-29),
    # ACTION2 (changed cells around row 24-35),
    # ACTION3 (changed cells around row 19-22),
    # ACTION4 (changed cells around row 20-21).
    #
    # It seems like objects move based on these actions.
    # We will implement a simple movement logic for any block that isn't color 5 (background/wall).
    # Since we don't have enough data to actually deduce the exact physics,
    # and the goal is to return a predicted next grid, we will simply apply the deltas
    # if they were constant. But since this is a world model, we need general rules.
    #
    # The most likely scenario is a Sokoban-like or puzzle game where blocks are moved.
    #
    # Let's refine:
    # ACTION1 = Up, ACTION2 = Down, ACTION3 = Left, ACTION4 = Right.
    #
    # For a specific action, let'<|channel>thought// No thought process provided.
    # Just implementing basic directional shift for non-wall (color 5) cells.

    new_grid = grid.copy()
    
    # Direction vectors
    directions = {
        1: (-1, 0), # Up
        2: (1, 0),  # Down
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }
    
    if action not in directions:
        return new_grid
    
    dr, dc = directions[action]
    
    # Identify all "movable" objects (not color 5 and potentially not others)
    # In the initial grid, color 5 is very common (background).
    # Color 4 is also common (walls/boundaries).
    # We actually see colors like 6, 8, 9, 14, 0, 1, 2, 3.
    # Let's assume everything except 5 and 4 are movable blocks.
    
    # To avoid moving walls, we need to identify what stays put.
    # Looking at INITIAL GRID:
    # r12:5x17, 4x30, 5x17 -> Color 4 is a horizontal bar.
    # This suggests 4 and 5 are static boundaries.
    
    movable_mask = (grid != 5) & (grid != 4)
    
    # Simple movement logic: shift all movable cells by (dr, dc) if target is background (5)
    # We process them in an order that prevents overwriting before checking.
    rows, cols = np.where(movable_mask)
    
    # Sort rows/cols based on direction to move "front" first
    if dr == -1:
        sort_idx = np.argsort(rows)
    elif dr == 1:
        sort_idx = np.argsort(-rows)
    elif dc == -1:
        sort_idx = np.argsort(cols)
    elif dc == 1:
        sort_idx = np.argsort(-cols)
    else:
        sort_idx = np.arange(len(rows))

    for r, c in zip(rows[sort_idx], cols[sort_idx]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 64 and 0 <= nc < 64:
            if grid[nr, nc] == 5: # Only move into empty space
                new_grid[nr, nc] = grid[r, c]
                new_grid[r, c] = 5
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves blocks reaching a target or clearing a pattern.
    # Without a win state example, we check for a common completion condition
    # like all movable blocks being gone or specific colors aligned.
    # Given the data, it's hard to be sure. Let's assume any change that matches
    # a known "win" layout (which we don't have).
    # We will return False unless we see something obvious.
    return False