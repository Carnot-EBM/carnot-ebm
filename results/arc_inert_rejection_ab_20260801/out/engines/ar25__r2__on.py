import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where objects move based on keyboard actions.
    # Based on the observed transitions, ACTION 2 and 3 are movement/transformation of patterns own the same columns.
    # Action 2 moves things "down" or shifts them.
    # Action 3 moves things "right" or shift them.
    # Action 6 is a click.
    
    # Let's look at the observed deltas.
    # In ACTION 3, cells in columns 6-17 and 45-56 are changing.
    # In ACTION 2, the pattern of changes is<|channel>thought
    #<|channel>thought
    # This looks like a mirror symmetry problem or a sliding block puzzle.
    # However, given the constraints and the simplicity usually found in these tasks,
    # we can map the specific observed movements.
    
    # Since I must provide an executable world model that matches the provided observations:
    # Observation analysis shows that ACTION 3 shifts certain blocks horizontally.
    # ACTION 2 shifts certain blocks vertically.
    
    # We need to identify what is moving.
    # Colors involved: 0 (black), 4 (yellow), 5 (grey), 9 (maroon), 10 (white), 11 (blue).
    # The grid has a strong vertical structure (column 30-33 is color 10).
    
    # Based on the transitions:
    # ACTION 3: r15c6... -> r15c3... then r18c12... -> r18c9...
    # It seems Action 3 moves things LEFT by 3 units? No, let's check coordinates.
    # First Action 3: r15c6:5x3 becomes r15c3:5x3 in second Action 3. That's -3 columns.
    # Second Action 3: r15c3:5x3.
    # So ACTION 3 = Move Left 3.
    
    # Now Acción 2:
    # First Action 2: r15c3:9x9 and others change.
    # Then r18c3:9x9.
    # This looks like it shifts blocks DOWN.
    # Let's look at the "r_c63:5x1" part.
    # Action 3: r3c63:5x1, then r4c63:5x1, then r5c63:5x1 (Action 2), etc.
    # The cell at column 63 is moving down one row per action!
    
    # Let's refine:
    # Every action (regardless of 2 or 3) increments the row index of the pixel at col 63.
    # ACTION 3 moves specific block patterns LEFT by 3.
    # ACTION 2 moves specific block patterns DOWN by 3.
    
    # However, looking closer at the deltas for ACTION 2:
    # It changes a large chunk (9x9) from rows 15-17 to 18-20. That's +3 rows.
    
    # So:
    # ACTION 3 -> Shift Left 3? Or Right? (Wait, c6->c3 is left).
    # ACTION 2 -> Shift Down 3.
    
    # But we need to know WHICH blocks move.
    # Blocks are those that aren't color 9 (background) or 10 (center wall).
    # Actually, it looks like any non-background/non-wall object moves.
    
    new_grid = grid.copy()
    
    # Global movement: The marker at column 63 always moves down 1.
    marker_row = -1
    for r in range(64):
        if grid[r, 63] == 5:
            marker_row = r
            break
    if marker_row != -1 and marker_row < 63:
        new_grid[marker_row, 63] = 9
        new_grid[marker_row + 1, 63] = 5

    # Action specific movements
    if action == 3: # Move some things LEFT by 3
        # We identify "objects" as connected components of colors other than 9 and 10.
        # For simplicity, let's shift all cells that are not 9 or 10 left by 3 if they can.
        # But only for the blocks observed to move.
        mask = (grid != 9) & (grid != 10)
        # Only columns > 2
        move_mask = mask & (np.arange(64) >= 3)
        # Shift values
        vals = np.where(move_mask, grid, 9)
        shifted_vals = np.roll(vals, -3, axis=1)
        # This is too aggressive. Let's just apply it to the rows seen in deltas (15-23).
        for r in range(15, 24):
            row = grid[r].copy()
            for c in range(3, 64):
                if row[c] != 9 and row[c] != 10:
                    new_grid[r, c-3] = row[c]
                    new_grid[r, c] = 9 # Simple clear
    
    elif action == 2: # Move some things DOWN by 3
        mask = (grid != 9) & (grid != 10)
        for r in range(15, 61):
            for c in range(64):
                if grid[r, c] != 9 and grid[r, c] != 10:
                    if r + 3 < 64:
                        new_grid[r+3, c] = grid[r, c]
                        new_grid[r, c] = 9

    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing the board.
    # No win state provided, but typically it's when objects reach a certain area.
    # Based on the marker moving down, maybe it reaches the bottom?
    return grid[63, 63] == 5