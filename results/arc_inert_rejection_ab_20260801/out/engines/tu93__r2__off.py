import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where some object (represented by color 9/4)
    # moves across a grid of obstacles (color 2/0/5).
    # Action 2: Down
    # Action 3: Left
    # Action 3: Left (again)
    #<|channel>thought
    # Let's analyze the movement patterns from the deltas.
    # The "player" or "active object" seems to be represented by color 9 (and sometimes 4).
    # It moves in blocks of 3x3.
    # ACTION 2: Down
    # ACTION 3: Left
    # ACTION 4: Right
    # ACTION 1: Up (implied)
    # In each transition, the old position of the 3x3 block of color 9 is replaced by something else (often 0),
    # and a new position is filled with color 9.
    # The cells at r63c55... are also changing, which might be a progress bar or counter.

    # Find current position of the active object (color 9).
    # We look for the top-left corner of the 3x3 block of color 9.
    pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                pos = (r, c)
                break
        if pos: break

    if pos is None:
        return grid.copy()

    new_grid = grid.copy()
    r, c = pos
    
    # Movement offsets
    dr, dc = 0, 0
    if action == 2: # Down
        dr, dc = 3, 0
    elif action == 3: # Left
        dr, dc = 0, -3
    elif action == 4: # Right
        dr, dc = 0, 3
    elif action == 1: # Up
        dr, dc = -3, 0

    nr, nc = r + dr, c + dc

    # Check boundaries and simple collision (simplified as we don't have full map rules)
    if nr < 0 or nr + 3 > grid.shape[0] or nc < 0 or nc + 3 > grid.shape[1]:
        return grid.copy()

    # Update the old position to be "empty" (color 0) if it was color 9/4
    # This is a simplification; in reality, it might restore the original background.
    # The observed deltas show that when moving from (r,c), those cells often become 0.
    for i in range(3):
        for j in range(3):
            new_grid[r+i, c+j] = 0 if grid[r+i, c+j] == 9 else grid[r+i, c+j]

    # Move the block of color 9 to the new position
    for i in range(3):
        for j in range(3):
            new_grid[nr+i, nc+j] = 9
            # Handle the special case where some cells are 4 instead of 9
            # In the data: r29c33:9x1,4x1,9x1 -> means one cell is 4.
            # We'll just use 9 for simplicity as most are 9.

    # Special handling for the progress bar at row 63
    # Every move seems to decrement values or change colors at the bottom right.
    # Looking at the deltas: r63c55:0x1, r63c54:0x1, etc.
    # It looks like a sequence of cells starting from (63, 55) and moving left are being set to 0.
    # Let's find the first non-zero cell from the right in row 63.
    last_col = 63 # Max col is 63
    while last_col >= 0 and grid[63, last_col] == 0:
        last_col -= 1
    if last_col >= 0:
        new_grid[63, last_col] = 0

    return new_grid

def is_level_complete(grid):
    # The level is complete when the object reaches a target or the progress bar is empty.
    # In this game, it might be reaching color 14.
    # We don't have a win state example, but usually, it involves interacting with a goal.
    # Check if any part of the active block (color 9) overlaps with the goal (color 14).
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                # Check surrounding for goal color 14
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if 0 <= r+dr < grid.shape[0] and 0 <= c+dc < grid.shape[1]:
                            if grid[r+dr, c+dc] == 14:
                                return True
    return False