import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION3: Left/Right movement? No, looking at deltas, it seems like a "shift" or "slide".
    # ACTION2: Downward movement.
    
    # The game involves moving blocks of colors (5 and 4) within boundaries (color 9).
    # Let's identify the same-shaped objects that move.
    # In ACTION3, the cells change from color 9 to something else and vice versa.
    # lathought: Looking at the delta for ACTION3, r15c6:5x3 etc. suggests shifting a block of size 3x3 or similar.
    # ACTION2 is clearly moving things down.
    
    # Based on the observed transitions:
    # ACTION3: Shift left/right.
    # ACTION2: Move down.
    
    # To implement this simply, we can track the 'player' object (the one that moves).
    # Find all non-boundary (non-9, non-10, non-11) cells.
    # Find connected components of these cells.
    # Find the center of mass or top-left corner of theC
    # Actually, let's just find the coordinates of all pixels of the 'moving' objects (colors 4, 5, 0).
    # And try to shift them.
    
    # The grid contains boundary walls (color 9, 10, 11).
    # Boundary colors are boundaries.
    # Let's assume action 2 is DOWN, action 3 is LEFT/RIGHT? No, looking at Action 3 deltas, it seems like they move horizontally.
    # ACTION3: Left/Right movement.
    # ACTION2: Downward movement.
    # ACTION6: Click.
    
    # a bit more complex: there is a "cursor" (color 5 in some cases, color 4 in others).
    # lathought: Looking at the INITIAL GRID, there is a block of color 5 and color 4.
    # These blocks move.
    # ACTION3: Horizontal shift.
    # ACTION2: Vertical shift.
    # ACTION3 delta r15c6:5x3... suggests moving from c15 to c6. That's a shift of -9.
    # ACTION3 second transition: r15c3:5x3... suggest shifting further left.
    # ACTION2 transitions: r18c3:5x6... etc. Moving down.
    
    # The objects being moved are the ones that aren't boundary walls (9, 10, 11).
    # Boundary colors: 9, 10, 11.
    # Let's define 'walls' as any cell with value 9, 10, or 11.
    # But wait, color 9 is the background! Color 10 is a central wall.
    # Walls = {10} maybe? Or just anything not 0, 4, 5.
    # Let's assume only cells with values in {0, 4, 5} are "movable".
    # Let's find all movable pixels and shift them.
    
    # Action mapping:
    # action 2: Down
    # action 3: Left/Right (based on deltas, it looks like they move horizontally)
    # action 1: Up
    # action 4: Right
    # action 5: Left
    # Wait, let's look at the actions again.
    # ACTION3: Horizontal movement.
    # ACTION2: Downward movement.
    # Looking at the sequence of ACTION2s, the block moves down by 3 rows each time.
    # ACTION3 shifts the block left by 6-9 columns.
    
    # The grid has two blocks: one of color 5 and one of color 4.
    # They both move together as a single unit or independently?
    # In ACTION3, both colors 5 and 4 change positions.
    # So they move as a group.
    # Find all coordinates of pixels that are not background (9) and not walls (10, 11).
    # lathought: Color 11 is also boundary.
    # Let's just identify "movable" cells as those with values in {0, 4, 5}.
    # Movable = {(r, c) | grid[r, c] in {0, 4, 5}}
    # Action 2: Shift movable cells down by 3.
    # Action 3: Shift movable cells left/right.
    # Wait, looking at the deltas for ACTION3, it seems to be shifting LEFT.
    # Let's assume action 2=Down, 3=Left, 4=Right, 1=Up.
    
    # But wait, look at the first ACTION3 delta: r15c6:5x3... The original block was at c9-17. Now it's at c6-14. That's -3 shift.
    # Second ACTION3 delta: r15c3:5x3... Original was c6-14. Now it's at c3-11. That's -3 shift.
    # So ACTION3 is LEFT.
    # ACTION2 shifts DOWN by 3 rows.
    
    # Let's implement this logic.
    
    new_grid = grid.copy()
    movable_cells = np.where((grid != 9) & (grid != 10) & (grid != 11))
    rows, cols = movable_cells
    
    dr, dc = 0, 0
    if action == 2: # Down
        dr = 3
    elif action == 3: # Left
        dc = -3
    elif action == 4: # Right
        dc = 3
    elif action == 1: # Up
        dr = -3
        
    if dr != 0 or dc != 0:
        # To avoid overwriting, we first clear the old positions and set them to background (9).
        for r, c in zip(rows, cols):
            new_grid[r, c] = 9
        
        # Then we place the cells in their new positions.
        for r, c in zip(rows, cols):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                # Only move if the target cell is not a wall (10, 11).
                if new_grid[nr, nc] != 10 and new_grid[nr, nc] != 11:
                    # We need to handle collisions carefully. But for now, let's just shift.
                    pass
        # This logic is slightly flawed because it clears everything then moves.
        # Let's refine.
        
    return new_grid

def engine(grid, action, data):
    # Refined implementation based on observed deltas.
    # ACTION2: Down by 3 rows.
    # ACTION3: Left by 3 columns.
    # ACTION4: Right by 3 columns.
    # ACTION1: Up by 3 rows.
    
    new_grid = grid.copy()
    movable_mask = (grid != 9) & (grid != 10) & (grid != 11)
    rows, cols = np.where(movable_mask)
    
    dr, dc = 0, 0
    if action == 1: dr = -3
    elif action == 2: dr = 3
    elif action == 3: dc = -3
    elif action == 4: dc = 3
    
    if dr == 0 and dc == 0:
        return new_grid
    
    # Clear old positions
    for r, c in zip(rows, cols):
        new_grid[r, c] = 9
    
    # Move to new positions
    for r, c in zip(rows, cols):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Check if the target cell is a wall (10 or 11).
            # We must check against the ORIGINAL grid for walls.
            if grid[nr, nc] != 10 and grid[nr, nc] != 11:
                new_grid[nr, nc] = grid[r, c]
            else:
                # If it hits a wall, it stays put? Or doesn't move?
                # In the observed deltas, they don't seem to hit walls yet.
                # Let's assume it just stops at the wall.
                new_grid[r, c] = grid[r, c]
        else:
            new_grid[r, c] = grid[r, c]
            
    return new_grid

def is_level_complete(grid):
    # The win state isn't provided, but usually it involves moving an object to a target.
    # Looking at the INITIAL GRID, there are blocks of color 5 and 4.
    # There might be a target area. Color 11 is present in some areas.
    # For now, let's return False unless we see a specific pattern.
    # Since no WIN STATE was given, we can only guess.
    # A common ARC-AGI goal is to align objects or fill a region.
    # Let's check if any movable object has reached the bottom right (where color 11 is).
    movable_mask = (grid != 9) & (grid != 10) & (grid != 11)
    rows, cols = np.where(movable_mask)
    if len(rows) == 0:
        return False
    # Check if any movable cell is adjacent to color 11 cells that aren't on the boundary.
    # This is a very rough guess.
    return False