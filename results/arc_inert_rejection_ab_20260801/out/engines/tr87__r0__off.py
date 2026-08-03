import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the same pattern of changes occurs in multiple regions.
    # Based on the observed transitions, it's<|channel>thought
    # that certain actions move or change colors of blocks.
    # Action 4 moves something horizontally? No, looking at the deltas,
    # Action 4 affects rows 48, 49, 59, 60. These are specific "slot" areas.
    # Action 1 and 2 affect different columns/rows (e.g., 52-56).
    # Action 6 is not seen here but usually means click.
    # Let's analyze the ACTION4 delta:
    # r48c15:3x5, r48c22:0x5... This looks like shifting a 5-wide block from col 15 to 22?
    # Wait, no. It says r48c15 becomes 3x5 AND r48c22 becomes 0x5.
    # This implies a swap or movement.
    # Looking closer at the grid layout, there are several repeating structures.
    # The action effects are very localized.
    # Since we don't have enough data to fully induce the logic (like what triggers the win),
    # and the provided transitions are just snippets, I will implement a basic state machine
    # that reflects the observed changes if possible, but since it's an ARC game,
    # it likely involves moving blocks of color to target positions.
    
    # However, the prompt asks for a general rule.
    # In this specific case, let's look at the "cursor" in r63c64.
    # r63c63 is 1x63, then 4x1. So cell (63, 63) is color 4.
    # After ACTION2, it moves to r63c62:4x1. Then r63c61... r63c60... r63c59.
    # It seems the cursor (color 4) moves left with some actions.
    # Action 2: cursor moves left.
    # Action 4: cursor moves left.
    # Action 1: cursor moves left sometimes? No, not always.
    
    # Let's refine:
    # Transition 1: ACTION4 -> r63c62 doesn't change? Wait, no delta for r63 in first ACTION4.
    # Transition 2: ACTION2 -> r63c62:4x1. (Was c63). Cursor moved left.
    # Transition 3: ACTION2 -> no mention of r63.
    # Transition 4: ACTION4 -> r63c61:4x1. Cursor moved left.
    # Transition 5: ACTION1 -> no mention of r63.
    # Transition 6: ACTION1 -> r63c60:4x1. Cursor moved left.
    # Transition 7: ACTION1 -> no mention of r63.
    # Transition 8: ACTION4 -> r63c59:4x1. Cursor moved left.
    
    # It seems the "cursor" at (63, col) is a state indicator.
    # The actions move it and modify blocks above.
    
    # Since I cannot deduce the full complex logic from these few transitions,
    # and the prompt requires an executable world model, I will implement the cursor movement
    # and the specific block changes observed.
    
    new_grid = grid.copy()
    
    # Find cursor position
    cursor_pos = np.where(grid == 4)
    if len(cursor_pos[0]) > 0:
        curr_r, curr_c = cursor_pos[0][0], cursor_pos[1][0]
        
        if action == 2: # Move left
            new_grid[curr_r, curr_c] = grid[curr_r, curr_c-1] if curr_c > 0 else 4
            if curr_c > 0:
                new_grid[curr_r, curr_c-1] = 4
        elif action == 4: # Move left
            new_grid[curr_r, curr_c] = grid[curr_r, curr_c-1] if curr_c > 0 else 4
            if curr_c > 0:
                new_grid[curr_r, curr_c-1] = 4
        elif action == 1: # Move left (sometimes)
            # In the data, ACTION1 moved it in transition 6 but not 5 or 7.
            # This suggests a conditional move.
            # For simplicity, let's assume it moves every other time or based on some state.
            # But we must be deterministic. Let's try moving it every time for now.
            new_grid[curr_r, curr_c] = grid[curr_r, curr_c-1] if curr_c > 0 else 4
            if curr_c > 0:
                new_grid[curr_r, curr_c-1] = 4

    # The block changes are too specific to generalize without more data.
    # I will implement the cursor movement as the primary logic.
    return new_grid

def is_level_complete(grid):
    # Win condition usually involves blocks reaching a target or cursor reaching a point.
    # Without win state data, this is a guess.
    cursor_pos = np.where(grid == 4)
    if len(cursor_pos[0]) > 0:
        return cursor_pos[1][0] == 0 # Cursor reached left edge
    return False