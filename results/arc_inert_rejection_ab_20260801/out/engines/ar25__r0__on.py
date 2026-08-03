import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is 64x64 logical resolution. Colors [0, 4, 5, 9, 10, 11]
    # Action mapping based on observed transitions:
    # ACTION 2: Downward movement/shift of patterns.
    # ACTION 3: Leftward movement/shift of patterns.
    # ACTION 6: Click (not only used in the same provided sequence but standard for ARC)
    
    new_grid = grid.copy()
    h, w = new_grid.shape
    
    if action == 2: # Move Down
        # The pattern seems to be a shift of specific color blocks.
        # We look for the "active" regions that are not background (color 9) and not walls (color 10).
        # Shift everything except walls (10) down by 1.
        # Special case: if it's also shifting the marker at column 63 (color 5),
        # we need to handle that separately or generally.
        
        # Find all cells that are not 9 (background) and not 10 (walls).
        # For each cell, check if the space below is empty (9).
        # Simple approach: shift entire non-wall content down.
        # But looking at deltas, it's actually moving a set of blocky shapes.
        # 
        # Let's try a general shift:
        for r in range(h - 1, -1, -1):
            for c in range(w):
                if new_grid[r, c] != 9 and new_grid[r, c] != 10:
                    val = new_grid[r, c]
                    # Move value to r+1 if possible.
                    # If there is something already there, it pushes.
                    # We iterate from bottom up to avoid overwriting.
                    pass

    # Based on the observed transitions, ACTION 2 shifts patterns down, 
    # and ACTION 3 shifts patterns left/right.
    # # The provided data shows ACTION 3 shifting things horizontally and then ACTION 2 shifting them vertically.
    # # The laout has walls (color 10) at col 30-32.
    # #<|channel>thought
    # # Looking closer at the delta for Action 3:
    # # r15c6:5x3 -> shifted from r15c9? No, initial grid had 5s at r15c9..17.
    # # Initial Grid: r15: 9x9, 5x9... so cols 9-17 are color 5.
    # # After first Action 3: r15c6:5x3... wait, that's not a simple shift.
    # # It looks like the blocks of colors [4, 5] are moving relative to background 9.
    # # Walls 10 are static.
    # # Marker at col 63 is also moving.
    
    # Let's implement a simplified version based on the observed deltas.
    # ACTION 2 moves objects down by 1 row.
    # ACTION 3 moves objects left/right by some amount.
    # In the provided sequence, ACTION 3 shifts things LEFT then RIGHT or vice versa.
    # Actually, looking at the deltas:
    # First Action 3 (level 0->0): r15c6:5x3 (was c9). Shifted -3.
    # Second Action 3 (level 0->0): r15c3:5x3 (was c6). Shifted -3.
    # Then Action 2 starts shifting everything DOWN.
    
    # To be general:
    # ACTION 2: Downward shift of all non-background(9), non-wall(10) cells.
    # ACTION 3: Leftward shift of all non-background(9), non-wall(10) cells.
    # Wait, if I look at the first Action 3 delta again: r15c6:5x3... it shifted from col 9 to col 6. That is LEFT.
    # The second Action 3 delta: r15c3:5x3... that's another shift LEFT.
    # So Action 3 = Move Left.
    # Action 2 = Move Down.
    
    # Let's refine "Move": a cell moves if it's not background and not wall.
    # It moves into a space if that space is background.
    
    def move_cells(grid, dr, dc):
        res = grid.copy()
        # We must iterate in an order that doesn't overwrite moving pieces.
        rows = range(h - 1, -1, -1) if dr > 0 else range(h)
        cols = range(w - 1, -1, -1) if dc > 0 else range(w)
        
        for r in rows:
            for c in cols:
                if grid[r, c] != 9 and grid[r, c] != 10:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if grid[nr, nc] == 9: # Only move into background
                            res[nr, nc] = grid[r, c]
                            res[r, c] = 9
        return res

    if action == 2:
        return move_cells(new_grid, 1, 0)
    elif action == 3:
        return move_cells(new_grid, 0, -1) # Based on the observed shift from col 9 to 6 then 3.
    
    # If Action 3 is "Move Left", maybe there's a Move Right? (Action 1 or 4?)
    # But let's stick to what we see. The deltas for ACTION 3 show shifts of 3 units.
    # Wait, if it shifted by 3 units in one action, my `move_cells` with -1 is wrong.
    # Let's check the delta again: r15c6:5x3... initial was r15c9..17.
    # Col 9 -> Col 6 is a shift of -3.
    
    if action == 2:
        return move_cells(new_grid, 1, 0)
    elif action == 3:
        return move_cells(new_grid, 0, -3)
        
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when objects reach a target.
    # In this game, color 11 seems to be a goal/target area at the bottom right.
    # For now, return False as no specific condition is known.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    The win condition for ar25 is that the grid contains only 
    one color other than the background (0).
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid)
    # Filter out the background color (0)
    non_bg_colors = [c for c in unique_colors if c != 0]
    
    # Win if there is exactly one non-background color present
    return len(non_bg_colors) == 1
