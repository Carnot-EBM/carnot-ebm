import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the action.
    Based on observed transitions, ACTION 4 seems to move a set of blocks (color 0)
    and ACTION 2/1 seem to modify colors within specific regions.
    ACTION 6 is usually click, but not used here.
    """
    out = grid.copy()
    
    if action == 4:
        # Observed ACTION 4 moves color 0 blocks horizontally.
        # It looks like there's a pattern of moving 'holes' or 'blocks'.
        # Find all current positions of color 0.
        zeros = np.argwhere(grid == 0)
        if len(zeros) > 0:
            # This is a complex movement. Based on the evidence, 
            # we see r48c15:3x5, r49c15:3x1, etc. which means cells at (48, 15-19), (49, 15), (49, 19), (59, 15), (59, 19), (60, 15-19).
            # The delta shows that existing zeros are replaced by 3 and new zeros are created.
            # Shift everything in those coordinates relative to the same rows.
            # For each transition, it seems to shift right by 7 columns.
            shift = 7
            for r, c in zeros:
                out[r, c] = 3 # Fill old zero with background/wall color 3
                # New position for zero
                nr, nc = r, c + shift
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                    out[nr, nc] = 0
    elif action == 2:
        # ACTION 2 modifies colors in a specific region.
        # Observed: r52c22:5x1, r53c22:5x1...
        # This is likely modifying blocks of color 7 or 5.
        # Find all cells of color 5 or 7 in a specific range.
        #<|channel>thought
        # Based on observed transitions, ACTION 2 changes some 5s and 7s.
        # It's like a "toggle" or "swap" within a local area.
        # We can't induce a general rule from such few examples.
        # However, we looking at the grid layout, it's a puzzle game.
        # Let's try to implement a simple logic that matches the deltas.
        pass
    elif action == 1:
        # ACTION 1 also modifies colors in a specific region.
        # la own logic for a uma similar to a toggle.
        pass
    
    # Special case for the bottom right corner cell (63, 62) which seems to be a counter/timer.
    # The value at (63, 62) decreases by 1 each time an action is taken?
    # No, it starts at 4 and goes 4 -> 4 -> 4 -> 4 -> 4... wait.
    # r63c62:4x1 then r63c61:4x1 then r63c60:4x1 then r63c59:4x1.
    # This means color 4 moves left one column per ACTION 4.
    if action == 4:
        # Find current position of color 4 on row 63.
        pos = np.where(grid[63] == 4)[0]
        if len(pos) > 0:
            out[63, pos[0]] = 2 # Background color for that area
            out[63, pos[0]-1] = 4 if pos[0] > 0 else 4
    
    return out

def is_level_complete(grid):
    """
    The level is complete when the target state is reached.
    Since no win state was provided, we assume a common ARC-AGI goal.
    """
    # Usually, this involves clearing blocks or reaching a specific configuration.
    # Return False as we don't have a clear win condition from the data.
    return False