import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Inducing rules from observed transitions:
    # The game seems to be a movement/puzzle where certain colors are acting as walls or blocks.
    # Action 2: Down, Action 3: Right, Action 4: Left, Action 1: Up (implied).
    # Based on the deltas, it's<|channel>thought
    # Action 2: Move object (color 14) of size 3x3 centered at some position? No, 
    # looking at the deltas, color 14 areas are shifting.
    # Let's assume simple directional shifts for specific objects.
    # Object identified by color 14.
    # ACTION 2 -> Down, ACTION 3 -> Right, ACTION 4 -> Left, ACTION 1 -> Up.
    
    # Find all cells of color 14.
    # We need to identify "objects" which are connected components of color 14.
    # But in this case, let's try moving all color 14 pixels that aren't blocked by something other than 0 or 15.
    # Actually, looking at the data, only one group moves.
    # Let's find the bounding box of color 14 and move it.
    
    rows, cols = np.where(grid == 14)
    if len(rows) == 0:
        return grid.copy()

    # Grouping color 14 into a single entity if they are close.
    # For simplicity, we treat all color 14 as one object since there is only one set of movements.
    
    new_grid = grid.copy()
    dr, dc = 0, 0
    if action == 1: dr = -1
    elif action == 2: dr = 1
    elif action == 3: dc = 1
    elif action == 4: dc = -1
    
    # Check for collisions with colors other than 0, 15, 14 (walls/obstacles).
    # The "empty" space seems to be 0 or 15.
    # Based on INITIAL GRID, 2 is background. 15 is empty? 0 is also used in deltas.
    # Let's assume walls are everything except 0, 15, 14, and the moving object itself.
    
    # To move the object, we shift all cells of color 14.
    # We check if any cell of color 14 would land on a wall.
    # If it does, the movement is blocked.
    
    # Find current positions of color 14
    mask = (grid == 14)
    
    # Shift mask
    shifted_mask = np.zeros_like(mask)
    if dr != 0:
        if dr == 1: shifted_mask[1:, :] = mask[:-1, :]
        else: shifted_mask[:-1, :] = mask[1:, :]
    elif dc != 0:
        if dc == 1: shifted_mask[:, 1:] = mask[:, :-1]
        else: shifted_mask[:, :-1] = mask[:, 1:]
    else:
        return grid.copy()

    # Collision detection:
    # A collision occurs if a shifted color-14 pixel lands on a "wall".
    # Walls are colors other than 0, 15, 2, 14.
    # Wait, looking at INITIAL GRID, background is 2.
    # Let's define walls as anything not in {0, 15, 2, 14}.
    # But wait, ACTION 4 shifts 14 into areas that were 1x3 or something.
    # Let' same just move the pixels and overwrite.
    
    # Based on observed transitions, it looks like the object (color 14) moves and replaces what was there.
    # The cells it leaves behind become... something else?
    # Looking at r30c18:1x3,14x3 -> This means col 18 becomes 1 for 3 cells, then 14 for 3 cells.
    # It seems the moving object (color 14) is replacing values.
    
    # Let's implement simple shift of all 14s.
    # And we need to know what they leave behind.
    # In Action 4 (Left), r30c18:1x3,14x3 suggests color 1 replaced some cells.
    # Actually, let's look closer: "r30c18:1x3,14x3" means from col 18, 3 cells are now 1, then 3 cells are now 14.
    # This implies a block of size 3 moved left.
    
    # Simple implementation: find all connected components of 14, move them if possible.
    # But given the constraints, maybe just shifting all 14s and filling the gap with 1 is enough.
    
    res = grid.copy()
    mask = (grid == 14)
    
    # Shift positions
    rows, cols = np.where(mask)
    new_rows, new_cols = rows + dr, cols + dc
    
    # Bounds check
    if np.any(new_rows < 0) or np.any(new_rows >= grid.shape[0]) or \
       np.any(new_cols < 0) or np.any(new_cols >= grid.shape[1]):
        return grid.copy()

    # Fill old positions with color 1 (as seen in deltas)
    for r, c in zip(rows, cols):
        res[r, c] = 1
        
    # Place new positions as color 14
    for r, c in zip(new_rows, new_cols):
        res[r, c] = 14
        
    return res

def is_level_complete(grid):
    # No win state provided, but usually it's when an object reaches a target.
    # Since we don't have one, return False.
    return False