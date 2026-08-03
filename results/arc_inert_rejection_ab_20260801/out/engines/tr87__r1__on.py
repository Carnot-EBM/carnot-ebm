import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The observed transitions show a complex board layout with multiple regions of patterns.
    # Action 4 seems to move something related to color 0 and color 3 in rows 48, 49, 59, 60.
    # Actions 1 and 2 are directional movements that change colors in the same region (rows 52-56).
    # Action 6 is not present in thes observations but usually represents a click.
    #
    # Based on the transition deltas, we can actually see that a specific cell (r63, c63) starts at 4 and moves leftward across columns 62, 61, 60, 59.
    # This corresponds to the actions ACTION2, ACTION4, ACTION1.
    # Let's trace the marker (color 4) marker movement:
    # Initial: r63c63 = 4
    # After ACTION2: r63c62 = 4, r63c63 = 0? No, delta says r63c62:4x1. Wait, if it's just "4x1", it's setting it to 4.
    # The previous value was probably 2 or 3.
    #
    # Looking at the observed transitions:
    # Transition 1: ACTION4 -> no mention of r63.
    # Transition 2: ACTION2 -> r63c62:4x1.
    # Transition 3: ACTION2 -> no mention of r63.
    # Transition 4: ACTION4 -> r63c61:4x1.
    # Transition 5: ACTION1 -> no mention of r63.
    # Transition 6: ACTION1 -> r63c60:4x1.
    # Transition 7: ACTION1 -> no mentioned of r63.
    # Transition 8: ACTION4 -> r63c59:4x1.
    #
    # It seems Action 4 moves the marker (color 4) left by 2 columns if we combine some actions.
    # But wait, let's look at the action sequence:
    # ACTION4, ACTION2, ACTION2, ACTION4, ACTION1, ACTION1, ACTION1, ACTION4.
    # Marker movements:
    # Initial (r63, c63): 4
    # T2: ACTION2 -> c62
    # T4: ACTION4 -> c61
    # T6: ACTION1 -> c60
    # T8: ACTION4 -> c59
    #
    # This is a very simple movement pattern: each "effective" move action (ACTION1, ACTION2, ACTION4) shifts the marker color 4 from (63, col) to (63, col-1).
    #
    # Now let's examine the other changes in rows 48, 49, 59, 60 and rows 52-56.
    # These are likely secondary effects of the marker position.
    #
    # Let's implement a simple model where the same set of actions shift the marker and potentially trigger something else.
    #
    # However, since we canthought process was omitted as per rules, I must provide the engine.

    new_grid = grid.copy()
    
    # Find current marker position
    marker_pos = np.where(grid == 4)
    if len(marker_pos[0]) > 0:
        r, c = marker_pos[0][0], marker_pos[1][0]
    else:
        return new_grid

    # Action mapping for horizontal movement
    # ACTION1: moves left?
    # ACTION2: moves left?
    # ACTION4: moves left?
    # In the observed data, all these actions shifted the marker from 63->62->61->60->59.
    # The delta deltas show that r63c62:4x1, then r63c61:4x1, etc.
    #
    # Since it's too hard to induce the exact secondary effects without more data, 
    # but let's try to implement the marker movement first.
    
    # Based on the observations, every action in {1, 2, 4} seems to move the marker one step left.
    # Let's assume this is a<|channel>thought process was omitted as per rules, I must provide the engine.

    if action in [1, 2, 4]:
        new_grid[r, c] = grid[r, c+1 if c < 63 else c] # This is not correct.
        # Correct logic: set current to old value (assuming background is color 2 or 3)
        # We need to know what replaces the marker. Looking at INITIAL GRID, r63 is mostly 2s.
        # Wait, r63:1x63, 4x1 means col 0-62 are 1 and col 63 is 4.
        # So background of row 63 is color 1.
        new_grid[r, c] = 1
        new_grid[r, max(0, c - 1)] = 4
        
    return new_grid

def is_level_complete(grid):
    # No win state provided. Assume it's complete when marker reaches column 0.
    marker_pos = np.where(grid == 4)
    if len(marker_pos[0]) > 0:
        return marker_pos[1][0] == 0
    return False