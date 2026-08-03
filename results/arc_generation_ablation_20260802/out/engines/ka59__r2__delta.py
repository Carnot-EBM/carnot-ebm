import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the same rules induced from observed transitions.
    """
    out = grid.copy()
    
    # The provided observations are extremely sparse and specific to a few coordinates.
    # Based on the ACTION4 (click/keyboard) and ACTION3/2 movements, 
    # we see patterns of blocks moving or changing colors.
    # However, without clear general logic for "ka59", we implement a basic movement model
    # where certain actions move a 'player' or 'object' represented by color 14.
    #
    # Action mapping:
    # ACTION2: Up? Down? Left? Right?
    # ACTION3: Left?
    # ACTION4: Right?
    # ACTION4 also seems to be used for some interactions.
    
    # In this game, the evidence shows that when ACTION3 occurs, things shift left.
    # {r30c21 -> r30c18}, {r33c15 -> r33c18} etc.
    # Let's assume ACTION3 is LEFT, ACTION4 is RIGHT, ACTION2 is DOWN, ACTION1 is UP.
    
    # We identify the object being moved (color 14).
    # Find all cells with value 14.
    # The observed transitions show blocks of size 3x3 or similar shifting.
    # The same coordinates are changed in each transition.
    #
    # Since we cannot induce a general rule from such limited data, and we
    # provide a a deterministic engine based on the laout.
    
    # If action is ACTION6 (click), it use data['x'], data['y'].
    # If other actions are 1-7, they are  umapped own logic.
    
    # Based on the observations:
    # ACTION3 shifts color 14 blocks left by 3 units.
    # ACTION4 shifts color 14 blocks right by 3 units.
    # ACTION2 shifts color 14 blocks right/down?
    # ACTION2 shifted r30->r33.
    # ACTION3 then shifted c21->c18, then c18->c15.
    # ACTION3 = Left (-3 cols)
    # ACTION2 = Down (+3 rows)
    # ACTION4 = Right (+3 cols)
    # ACTION1 = Up (-3 rows)

    if action == 1: # UP
        mask = (grid == 14)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            if r >= 3:
                out[r-3, c] = 14
                out[r, c] = grid[r, c] # This is a bit simplified; we need to actually clear old pos
        # To avoid overwriting everything, we find unique blocks of size 3x3
        # We must be able to identify the "active" block.
    
    elif action == 2: # DOWN
        mask = (grid == 14)
        rows, cols = np.where(mask)
        # In the observations, ACTION2 moved color 14 from row 30-32 to 33-35.
        # Shift all 14s down by 3.
        # For each cell that was 14, it becomes what it was before? 
        # No, the evidence shows cells at r30c21 becoming 1x3 (color 1).
        # The new positions are r33c21.
        # Find current 14s, move them.
        # Find original background colors.
        # Let's assume background is color 1 or 2.
        #<|channel>thought
        pass

    # Given the extreme sparsity and specific coordinates in the data, 
    # we implement the logic for the observed transitions specifically.
    # This is a puzzle game where you move a 3x3 block of color 14.
    # To make this general, we find the center of the 14-block.
    
    # Find the top-left corner of the first 3x3 block of 14s.
    coords = np.argwhere(grid == 14)
    if coords.size > 0:
        r_min, c_min = coords.min(axis=0)
        # Move the block
        dr, dc = 0, 0
        if action == 1: dr = -3
        elif action == 2: dr = 3
        elif action == 3: dc = -3
        elif action == 4: dc = 3
        
        if dr != 0 or dc != 0:
            # Clear old block (set to background color 1)
            for r in range(r_min, r_min + 3):
                for c in range(c_min, c_min + 3):
                    if 0 <= r < 64 and 0 <= c < 64:
                        out[r, c] = 1 # Background is often 1 in these areas
            
            # Set new block
            for r in range(r_min + dr, r_min + dr + 3):
                for c in range(c_min + dc, c_min + dc + 3):
                    if 0 <= r < 64 and 0 <= c < 64:
                        out[r, c] = 14
    
    return out

def is_level_complete(grid):
    # No win state provided. Assume level complete if a certain condition is met.
    # In many ARC games, this means reaching a target area or clearing blocks.
    # Since we don't have the win state, return False.
    return False