import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The observed transitions suggest a movement of some object/entity represented by colors.
    # Action 2: Up? No, based on delta, it seems to be moving something down.
    # Action 3: Left.
    # Action 4: Right.
    # Action 2: Down.
    # Action 1: ?
    # Action 3: Left.
    # Action 4: Right.
    # Action 5: ?
    # Action 6: Click.
    # Action 7: ?
    # Action 7: ?
    # {Action 2: Down, Action 3: Left, Action 4: Right, Action 5: Up}
    
    # Based on the provided deltas, ACTION 4 moves an entity right, ACTION 3 moves it left, and ACTION 2 moves it down.
    # This implies ACTION 5 should move it up.
    
    # Find the "player" or "active entity". In this game, color 0 is often used as empty space or player.
    # Let's identify the active entity which is a block of cells.
    # Looking at the same grid, color 0 appears in r31c21 (initial) and then changes.
    # The object being moved is likely the region containing color 0.
    
    # We need to find the current position of the 'hole' (color 0).
    #<|channel>thought
    # The hole is small. Let's look for all zeros.
    zeros = np.argwhere(grid == 0)
    if len(zeros) == 0:
        return grid
    
    # Assume there is only one primary hole/entity moving.
    # Use the center of mass of zeros.
    cy, cx = int(np.mean(zeros[:, 0])), int(np.mean(zeros[:, 1]))
    
    # Movement vectors
    move_map = {
        2: (1, 0),  # Down
        3: (-1, 0), # This doesn't match ACTION 3 Left? Wait.
        4: (0, 1),  # Right
        5: (-1, 0), # Up
    }
    
    # Re-evaluating based on deltas:
    # Action 4: changed cells r30c18... r31c18... r32c18... -> then r30c21... etc.
    # It moves a block from col 18 to col 21. That's RIGHT.
    # Action 3: changed cells r30c21... -> r33c18... wait.
    # Let's look at Action 3 again: "r30c21:14x3,1x3" ... "r63c61:0x1".
    # The hole is moving!
    # Initial hole: r31c20 (approx).
    # After Action 4: Hole moves right.
    # After Action 3: Hole moves left.
    # After Action 2: Hole moves down.
    
    # Correct Map:
    # Action 2: Down (y+1)
    # Action 3: Left (x-1)
    # Action 4: Right (x+1)
    # Action 5: Up (y-1)
    
    dy, dx = 0, 0
    if action == 2: dy, dx = 1, 0
    elif action == 3: dy, dx = 0, -1
    elif action == 4: dy, dx = 0, 1
    elif action == 5: dy, dx = -1, 0
    else: return grid

    # Find the 'hole' (color 0 cells)
    holes = np.argwhere(grid == 0)
    if len(holes) == 0: return grid
    
    # We move the entire cluster of zeros as a unit.
    new_grid = grid.copy()
    for r, c in holes:
        # The hole swaps with the cell it moves into.
        nr, nc = r + dy, c + dx
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            new_grid[nr, nc] = 0
            new_grid[r, c] = grid[nr, nc]
            
    return new_grid

def is_level_complete(grid):
    # Based on typical ARC games, completion is often reaching a target or clearing something.
    # In this specific case, we don't have a win state provided.
    # Let's assume completing means moving the hole to a certain position or color.
    # For now, return False unless a clear condition is found.
    # Looking at the deltas "r63c63:0x1", "r63c62:0x1"... these look like counters!
    # It seems every action decrements a counter in the bottom right corner.
    # This suggests the level ends when the counter reaches zero or a certain value.
    return False