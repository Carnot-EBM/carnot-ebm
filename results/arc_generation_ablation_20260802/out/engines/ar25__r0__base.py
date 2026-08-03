import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 2 is move down, Action 3 is move left/right? No, based on observed transitions:
    # ACTION 2 seems to shift a pattern downwards.
    # ACTION 3 shifts patterns horizontally.
    # Let's look at the same-color blocks.
    # The colors are [0, 4, 5, 9, 10, 11].
    # Color 10 is a vertical strip in the middle.
    # Color 11 is a right edge column.
    # Color 9 is background.
    # Color 5 and 4 are parts of the "player" or "object".
    # In ACTION 2, the object moves from rows 15-23 to 18-26, then 21-27... etc.
    # It moves by 3 rows each time.
    # ACTION 3 shifts it horizontally.
    #
    # Based on the the delta changes, let's identify the "object" as cells that aren't color 9, 10, 11.
    # Object cells = { (r, c) | grid[r, c] not in [9, 10, 11] }
    # Action 2: Move object down by 3 units.
    # Action 3: Move object left/right? Looking at r15c6 -> r15c3. That's -3 columns.
    # Let's check if there's any boundary. The center strip (color 10) is at col 30-32.
    #
    # Actually, looking at the transitions:
    # ACTION 3 (level 0->0): changed cells r15c6:5x3 ... r15c3:5x3. This is a shift of -3 cols.
    # ACTION 2 (level 0->0): shifted rows from 15 to 18. Shift of +3 rows.
    #
    # Let's refine:
    # Action 2: Down 3.
    # Action 3: Left 3.
    # Wait, let's look at the action sequence again.
    # ACTION 3 happened twice. First time it moved something. Second time it moved it further left.
    # Then ACTION 2 happened multiple times, moving it down.
    #
    # Also, notice that color 11 changes in the rightmost column (col 63).
    # r3c63:5x1 -> r4c63:5x1 -> r5c63:5x1...
    # This looks like a "cursor" or "marker" on the right edge.
    # The marker moves down by 1 for every action.
    #
    # Now let's implement this logic.

    new_grid = grid.copy()
    
    # Marker movement: find cell with value 5 in col 63 and move it down.
    # Col 63 is the last column.
    # Find where grid[r, 63] == 5.
    # If found, we are moving it to (r+1, 63) and replacing old one with 11.
    # Let's check if there's only one such cell.
    # In INITIAL GRID, r0-2 have 5 at c63, r3-62 have 11 at c63.
    # Wait, INITIAL GRID says r0:9x30,10x3,9x30,5x1. So r0,1,2 have color 5 at c63.
    # Then r3 has 11 at c63.
    # After ACTION 3, r3c63 becomes 5. This means the "block" of 5s moved from r0-2 to r3? No, a single 5 appeared at r3.
    # Let's look at the delta: "r3c63:5x1". It replaces whatever was at (3, 63).
    # The previous state had 5s at (0,63), (1,63), (2,63).
    # Now (3,63) is also 5.
    # Looking at subsequent actions: r4c63:5x1, r5c63:5x1...
    # It seems for every action, one more cell in col 63 becomes 5.
    # Or rather, the marker moves down.

    # Object movement:
    # Find all cells that are not background/walls.
    # Background = 9, Walls = 10, 11.
    # Object = { (r, c) | grid[r, c] not in [9, 10, 11] }
    # Action 2: Shift object by (dr, dc) = (3, 0)
    # Action 3: Shift object by (dr, dc) = (0, -3)
    
    # But we must handle collisions with walls (color 10, 11).
    # If a cell would move into a wall, it probably stays or stops.
    # However, looking at the delta, the object just shifts and replaces whatever was there.
    # Let's check if it "clears" its old position.
    # Yes, the deltas show both the new positions being set to colors and the old ones returning to color 9.
    
    # Marker logic again:
    # Every action -> find current bottom-most 5 in col 63, place 5 at r+1.
    # Actually, let's just say for every action, the marker moves down one step.
    # Find max row r where grid[r, 63] == 5. Set grid[r+1, 63] = 5.

    # Object movement:
    # Identify object cells.
    # mask = (grid != 9) & (grid != 10) & (grid != 11)
    # For Action 2: dr=3, dc=0
    # For Action 3: dr=0, dc=-3
    # But wait, ACTION 3 is used twice. First time it moved something. Second time it moved it further left.
    # In the first ACTION 3, the object shifted from some initial pos to another.
    # Initial object was at rows 15-23, cols 9-18 etc.
    # Let's see: "r15c6:5x3 r15c15:9x3 ...". This means old values are being replaced.
    # The most consistent rule:
    # Action 2 -> Shift all non-[9,10,11] cells by +3 rows.
    # Action 3 -> Shift all non-[9,10,11] cells by -3 columns.
    # After shifting, any cell that moves out of bounds or into a wall [10, 11] might be handled.
    # But in these examples, they just move.
    # Also, we must restore the original positions to background (color 9).

    if action == 2:
        dr, dc = 3, 0
    elif action == 3:
        dr, dc = 0, -3
    else:
        return new_grid

    # Find current object cells and their colors
    obj_cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] not in [9, 10, 11]:
                obj_cells.append((r, c, grid[r, c]))
    
    # Clear old positions
    for r, c, val in obj_cells:
        new_grid[r, c] = 9
        
    # Move and place
    for r, c, val in obj_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # If target is a wall, maybe it doesn't move? 
            # But let's try simple shift first.
            new_grid[nr, nc] = val
        else:
            # Out of bounds -> stay or disappear? Let's assume stay.
            new_grid[r, c] = val

    # Marker movement (col 63)
    # Find the last row that has color 5 in col 63.
    marker_rows = np.where(grid[:, 63] == 5)[0]
    if marker_rows.size > 0:
        last_row = marker_rows[-1]
        if last_row + 1 < grid.shape[0]:
            new_grid[last_row + 1, 63] = 5
    elif grid[0, 63] == 5: # fallback
        new_grid[1, 63] = 5

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when an object reaches a goal.
    # Looking at the INITIAL GRID, there are some blocks of color 11 at r45-53, cols 33-42.
    # Maybe the goal is to move the object into those cells?
    # Or maybe just return False since we don't have a win condition.
    return False

def is_level_complete(grid):
    import numpy as np
    # The same color (non-zero) is present in own same-position
    # This is a<|channel>thought-free induction of a win state for ar25
    # A win state is when the grid is entirely zeroed out or a specific pattern is
    # a//b = c. In ar25, the same color same-position is the same as
    # a win state.
    # A win state is when the grid is entirely zero.
    return np.all(grid == 0)
