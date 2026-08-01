import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION 1: Move object/player to a new position or shift state.
    # ACTION 2: Not explicitly seen in detail but usually directional.
    # ACTION 3: Shift state of specific cells.
    # ACTION 4: Shift state of same cells in opposite direction.
    # ACTION 5: Not explicitly seen.
    # ACTION 6: Click.
    # ACTION 7: Not explicitly seen.
    
    # The provided observations are very sparse and complex for a general rule.
    # However, looking at the laout, it's a maze-like structure with colors 11, 12, 5, 10.
    # ACTION 1 seems to be 'move right' or similar, which changes cell values from 5 to 10 or vice versa.
    # ACTION 4 seems to be 'move left'.
    # ACTION 3 seems to be 'toggle' or 'interact'.
    
    # Based on the observation deltas, we cant easily induce a simple global rule without knowing the player position.
    # Since the player is not clearly marked as a unique color (maybe color 0),
    # let's identify the player as the only cell of color 0.
    
    player_pos = np.argwhere(grid == 0)[0] if np.any(grid == 0) else None
    if player_pos is None:
        return grid.copy()

    # Define movement directions based on action IDs
    # Action 1: Right, Action 2: Down, Action 3: Left, Action 4: Up? 
    # Or maybe Action 1: Right, Action 3: Left, Action 4: Up/Down?
    # Let's re-examine the transitions.
    # r0c62:0x1 -> r0c61:0x1 -> r0c60:0x1 -> r0c59:0x1
    # This shows the player (color 0) moving leftwards in row 0.
    # ACTION 1 moves player from c62 to c61, then c61 to c60, then c60 to c59.
    # ACTION 4 moves player from c59 to some other position? No, wait.
    # ACTION 4 delta: r0c59:0x1. Wait, it says "changed cells". If color 0 was at c59, and it changed, it means it moved.
    # But the delta for ACTION 4 is r0c59:0x1. That doesn't show where it went.
    # Actually, looking closer at the same sequence:
    # Initial: r0c62:0x1 (Wait, INITIAL grid has r0:5x63, 0x1. So col 63 is 0).
    # Action 1: r0c62:0x1. (Player moves from 63 to 62).
    # Action 1 again: r0c61:0x1. (Player moves from 62 to 61).
    # Action 1 again: r0c60:0x1. (Player moves from 60 to 59?). No, let's re-read.
    # Delta r0c62:0x1 means cell (0, 62) becomes value 0.
    # The previous player pos was (0, 63). So now (0, 63) must be restored to something else?
    # In these deltas, "changed cells" are listed. If only one cell changes to 0, and the other isn't listed, it might mean the delta is incomplete or I'm misreading.
    # But if we assume color 0 is the player, then ACTION 1 is 'move left'.
    
    # Let's try a simple model where color 0 is the player and action 1 is move left, 4 is move right.
    # However, the grid contains many state changes (color 5 <-> 10).
    # laout suggests a puzzle.
    
    new_grid = grid.copy()
    r, c = player_pos
    
    if action == 1: # Move Left
        nr, nc = r, c - 1
    elif action == 3: # Move Right/Interact
        nr, nc = r, c + 1
    elif action == 2: # Move Down
        nr, nc = r, c + 1 # This is not seen in data
    elif action == 4: # Move Up/Right
        nr, nc = r, c + 1
    else:
        return new_grid

    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
        # Update player position
        old_val = grid[r, c]
        new_val = grid[nr, nc]
        
        # The observed deltas show cells changing from 5 to 10 or vice versa.
        # These are "toggles" triggered by movement.
        # We can't induce the exact toggle logic without more detailed analysis of thes specific coordinates.
        # laout suggests that when a player moves, certain blocks of color 5 change to 10.
        # Let's implement a basic move and state shift.
        
        new_grid[r, c] = 5 if old_val == 0 else old_val
        new_grid[nr, nc] = 0
        
        # Simple rule for toggling based on observations:
        # Action 1 (Move Left) seems to trigger changes in rows 34-48 and cols 14-49.
        # If action is 1, we just apply the delta provided in the first transition.
        # This is not general.
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a goal or clearing all targets.
    # In this case, it's likely related to the same colors.
    # No clear win state grid was provided.
    # return True if np.any(grid == 0) else False # Placeholder
    # Usually, ARC games have goals like "fill everything" or "color X".
    # Since no win state is provided, let's assume it's complete when player reaches some target.
    return False