import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        # Determine which "zone" was clicked based on pixel coordinates
        # The board has a top zone (rows 0-9) and a main area (rows 10+)
        if py < 10:
            # Top zone click - no visible change in observed transitions
            pass
        else:
            # Main area click - fill 2 cells at bottom row (row 63)
            # Pattern: columns decrease by 2 each time starting from 62
            # We need to figure out the column offset
            # From observations: clicks seem to always produce r63cXX:5x2
            # Let's determine XX based on some counter or pattern
            
            # Looking at the sequence more carefully:
            # The bottom row starts all 0s. Each click fills 2 consecutive cells with color 5.
            # The positions seem to follow a specific order.
            
            # Let me re-examine: the deltas show r63c62, r63c60, r63c58, r63c56, r63c54, 
            # r63c52, r63c62, r63c60, r63c58, r63c56, r63c54, r63c52...
            # This looks like it cycles through even columns from right to left.
            
            # But wait - we can't use state beyond the grid itself.
            # Let me look at what determines the position.
            
            # Actually, looking at the initial grid, row 63 is all 0s (64 zeros).
            # After each click in the main area, 2 cells get filled with 5.
            # The column position seems to depend on how many cells are already filled.
            
            # Count existing non-zero cells in row 63 to determine next position
            row63 = grid[63]
            # Find the rightmost gap of exactly 2 consecutive zeros
            # Or find where the next pair should go
            
            # From the pattern: starts at col 62-63, then 60-61, 58-59, etc.
            # So it fills pairs from right to left: (62,63), (60,61), (58,59)...
            
            # Find the first pair of zeros starting from the right that hasn't been filled yet
            # Pairs are at positions: 62, 60, 58, 56, ... (even columns)
            # Each pair covers cols [c, c+1]
            
            # Determine which pair to fill by finding the rightmost unfilled pair
            # A pair at col c is "unfilled" if both grid[63][c] and grid[63][c+1] are 0
            
            # But we need to figure out the ORDER. Looking at transitions:
            # 1st click -> c62 (cols 62-63)
            # 2nd click -> c60 (cols 60-61)  
            # 3rd click -> c58 (cols 58-59)
            # 4th click -> c56 (cols 56-57)
            # 5th click -> c54 (cols 54-55)
            # 6th click -> c52 (cols 52-53)
            # 7th click -> c62 again?? That doesn't make sense unless it wraps or resets
            
            # Wait - transition 7 shows r63c62:5x2 again. Let me re-read...
            # Transition 7: ACTION6 data={'x': 48, 'y': 15} -> r63c62:5x2
            # But transition 1 already filled c62-c63 with 5s. So this would be setting them to 5 again?
            # Unless the delta only shows CHANGED cells, and if they're already 5, they wouldn't appear.
            
            # Hmm, but the problem says "delta = FULL set of changed cells". If a cell is already 5
            # and we set it to 5, it's not changed. So maybe I'm misreading.
            
            # Actually wait - let me reconsider. Maybe the clicks DON'T always fill row 63.
            # Some clicks also modify other rows (like transitions 3 and 9).
            
            # Let me look at what distinguishes those special clicks:
            # Transition 3: x=10, y=53 -> modifies rows 52-54 AND 58-60 AND 63
            # Transition 9: x=8, y=55 -> modifies rows 52-56 AND 58-60 AND 63
            
            # These seem to click on specific objects (the 0-colored holes or 15-colored blocks)
            
            # For now, let me handle the simple case: clicking in main area fills next pair in row 63
            # from right to left.
            
            # Find which pairs are already filled (non-zero) in row 63
            # Pairs start at even columns: 0, 2, 4, ..., 62
            # Fill order is right-to-left: 62, 60, 58, ...
            
            # But transition 7 re-fills c62... unless it's a different mechanism.
            
            # Let me just implement: find the rightmost pair of zeros in row 63 and fill it with 5.
            # Actually no - the pattern is specifically even-column pairs from right to left.
            
            # Count how many pairs have been filled from the right
            filled_count = 0
            for c in range(62, -1, -2):
                if grid[63][c] != 0 and grid[63][c+1] != 0:
                    filled_count += 1
                else:
                    break
            
            # Next pair to fill is at column 62 - 2*filled_count
            col = 62 - 2 * filled_count
            if col >= 0:
                grid[63][col] = 5
                grid[63][col + 1] = 5
    
    return grid


def is_level_complete(grid):
    # Check if all cells in row 63 are non-zero (all filled)
    # Or some other win condition
    # From observations, we never see a win state, so let's check if row 63 is fully filled
    return np.all(grid[63] != 0)