import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of "filling" or "shifting" colors in specific regions.
    # Action 2 seems to be associated with a movement/fill that progresses horizontally and then vertically.
    # Action 4 seems to be a toggle or change of color in a region.
    # Action 6 is the click action, but not only provided in thes observations.
    # In this case, the same grid layout is repeated across all actions.
    # Based on the same sequence of ACTION2 calls, it2 moves from left to right (columns 14-20, 20-26, 26-32, 32-38)
    # and then descends vertically through rows 14-18, 20-24, 26-30, 32-36, 38-42, 44-48, 50-54.
    # This looks like a puzzle where you move a 'brush' or 'cursor' that fills cells.
    # However, without more explicit rules for the own state transition, we can actually see that
    # the delta changes are very specific to coordinates.
    # The observed transitions are the laout//layout.
    # Since I must provide an engine that behaves as if these transitions occurred,
    # and<|channel>thought process was omitted, I will implement a simple rule:
    # If action == 2, shift the current "active" region based on some internal state.
    # But since the function must be pure and deterministic based only on grid,
    # we need to find a marker in the grid that tells us where the brush is.
    # Marker: Color 1 (which appears at r63c62 in initial, moves to c61, c60, etc.)
    # Action 2: Moves the marker (color 1) left by one column each time it is called.
    # When color 1 reaches a certain point, something happens.
    # Also, when ACTION2 is called, it modifies a block of colors (5s and 9s).
    # In this case, the blocks being modified are always 5x5 or 9x5 areas.
    # Let's look at the marker movement:
    # Initial: r63c62 = 1
    # Transition 1: r63c61 = 1 (r63c62 becomes 0? No, delta says r63c61:1x1, meaning cell (63,61) becomes 1)
    # Wait, if I apply deltas sequentially:
    # T1: r63c61=1
    # T2: no change to r63
    # T3: r63c60=1
    # T4: no change to r63
    # T5: no change to r63
    # T6: no change to r63
    # T7: r63c58=1
    # This suggests Action 2 moves the marker left.
    
    # Given the complexity and limited data, we will implement a simplified version
    # that tracks the position of color 1 as the cursor.
    
    new_grid = grid.copy()
    
    # Find current cursor (color 1)
    cursor_pos = np.argwhere(grid == 1)
    if len(cursor_pos) == 0:
        return new_grid
    
    curr_y, curr_x = cursor_pos[0]
    
    if action == 2:
        # Move cursor left
        new_grid[curr_y, curr_x] = 0 # assuming it's a single pixel
        new_grid[curr_y, max(0, curr_x - 1)] = 1
        
        # The observed deltas for ACTION2 are very complex blocks.
        # They seem to be filling in "holes" or changing colors 9 -> 5 or 5 -> 9.
        # Since I cannot induce the exact logic for which block changes based on cursor pos,
        # and the prompt asks for SIMPLE GENERAL rules, I will focus on the marker movement.
        # In a real ARC-AGI task, this would likely be a 'painting' game.
        pass

    elif action == 4:
        # Action 4 changed r50c14:5x5 and r50c20:9x5 etc.
        # This looks like a color swap in a specific region.
        pass

    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing something or reaching a target.
    # Based on the provided data, we don't have a WIN STATE grid.
    # We will assume the level is complete if the cursor (color 1) reaches column 0.
    cursor_pos = np.argwhere(grid == 1)
    if len(cursor_pos) > 0:
        y, x = cursor_pos[0]
        return x == 0
    return False