import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where blocks of color 9 are moved around.
    # Based on observed transitions, Action 2 moves something down, Action 3 moves left, 
    # Action 4 moves right, and Action 1 might move up.
    # Let's identify the "player" or "active object".
    # In this case, it's likely the block of color 9.
    # 9x3 blocks are<|channel>thought
    # Looking at the delta changes:
    # ACTION2: r21-23 c33 becomes 0, r27-29 c33 becomes 9
    # This is a movement of a 3x3 block of color 9 from (21, 33) to (27, 33).
    # It jumps by 6 rows.
    # ACTION3: r33c27 -> 0, r33c21 -> 9. Jump of 6 columns.
    # ACTION4: r39c21 -> 0, r39c27 -> 9. Jump of 6 columns.
    # All movements are in increments of 6.
    
    # Find all cells of color 9
    rows, cols = np.where(grid == 9)
    if len(rows) == 0:
        return grid.copy()

    # The active object is the largest contiguous block of color 9.
    # Since we only see one moving block in the deltas, let's assume it's the "player".
    # We need to find which block of color 9 is moving.
    # In the initial grid, there is a block at r21-23, c33-35? No, wait.
    # Initial Grid: r21: ... 9x3 ... (at col 33?). Let's check.
    # r21: 5x21 (21), 0x3 (24), 5x3 (27), 0x3 (30), 2x3 (33), 9x3 (36). Wait.
    # Let's re-calculate indices for INITIAL GRID r21:
    # 0-20: 5
    # 21-23: 0
    # 24-26: 5
    # 27-29: 0
    # 30-32: 2
    # 33-35: 9  <-- Here it is! Block of 9s starts at col 33.
    # So ACTION2 moves it from row 21 to row 27. Jump = 6.
    
    # The movement rules are based on a grid of "slots" spaced by 6 units.
    # Action 1: Up (-6 rows)
    # Action 2: Down (+6 rows)
    # Action 3: Left (-6 cols)
    # Action 4: Right (+6 cols)
    
    # We need to identify the moving block. In these transitions, only one block of color 9 moves.
    # Since we don't have multiple blocks of 9 in the deltas, let' same assume any 3x3 block of 9s that can move does.
    # But wait, there's also r63c55 changing to 0. This looks like a progress bar or counter.
    # Let's focus on the main object first.
    
    # Find all 3x3 blocks of color 9.
    blocks = []
    for r in range(0, 64 - 2):
        for c in range(0, 64 - 2):
            if np.all(grid[r:r+3, c:c+3] == 9):
                blocks.append((r, c))
                # Skip cells to avoid overlapping detections of the same block
                # (though for 3x3 it doesn't matter much if we just take the first)
    
    # If no 3x3 block found, try finding any cell of color 9.
    if not blocks:
        rows, cols = np.where(grid == 9)
        if len(rows) == 0: return grid.copy()
        # Just use the top-leftmost 9 as the reference point.
        ref_r, ref_c = rows[0], cols[0]
    else:
        # In this specific level, only one block moves. We'll move the first one found.
        ref_r, ref_c = blocks[0]

    new_grid = grid.copy()
    dr, dc = 0, 0
    if action == 1: dr = -6
    elif action == 2: dr = 6
    elif action == 3: dc = -6
    elif action == 4: dc = 6
    
    # Move the block
    # First, clear old position
    # To be safe, let's find all cells that are part of the moving "entity" (all connected 9s).
    # But based on deltas, it's a 3x3 block.
    new_grid[ref_r:ref_r+3, ref_c:ref_c+3] = 5 # Use background color 5? No, look at deltas.
    # Delta for ACTION2 r21c33:0x3... wait, it becomes 0.
    # Let's check INITIAL GRID again. Background is 5.
    # The delta says r21c33:0x3. This means it changes from 9 to 0.
    # Wait, if it was 9 and became 0, then 0 is the "empty" space in the maze.
    # Let's re-examine: Initial grid r21 has 9s at c33-35. Action 2 makes them 0s.
    # Then it makes r27c33-35 into 9s.
    # So the "player" moves between slots of color 0.
    
    # Correct logic:
    # 1. Find current position of the 3x3 block of 9s.
    # 2. Calculate new position based on action (jump by 6).
    # 3. Set old position cells to whatever they were before the block arrived (color 0).
    # 4. Set new position cells to 9.
    
    # To determine what to set the old position to, we can look at the initial grid or deltas.
    # In ACTION2, r21c33 becomes 0. In INITIAL GRID, r21c33 was 9.
    # So the player replaces a slot of 0s with 9s, and leaves behind 0s.
    
    new_grid[ref_r:ref_r+3, ref_c:ref_c+3] = 0
    nr, nc = ref_r + dr, ref_c + dc
    if 0 <= nr < 64 - 2 and 0 <= nc < 64 - 2:
        new_grid[nr:nr+3, nc:nc+3] = 9

    # Handle the "counter" at r63.
    # Action 2: r63c55 -> 0, then r63c54 -> 0...
    # It seems every move decrements/changes a cell in row 63.
    # Let's find the rightmost non-zero cell in row 63 and change it to 0.
    row63 = new_grid[63, :]
    nonzero_cols = np.where(row63 != 0)[0]
    if len(nonzero_cols) > 0:
        last_col = nonzero_cols[-1]
        # Based on deltas: ACTION2 (level 0->0): r63c55:0x1, then r63c54:0x1...
        # This is exactly what we are doing.
        new_grid[63, last_col] = 0

    return new_grid

def is_level_complete(grid):
    # Level complete when all cells in row 63 are 0? Or some specific condition.
    # In the observed transitions, the counter is decreasing.
    # Usually, this means the level ends when the counter reaches 0 or player reaches a goal.
    # Since no win state was provided, let's assume row 63 being all 0s is the win.
    return np.all(grid[63, :] == 0)