import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    y, x = data['y'], data['x']
    out = grid.copy()
    
    # The observed transitions show ACTION6 (click) toggles colors at specific locations.
    # Specifically, clicking on certain coordinates seems to trigger a color shift 
    # between color 14 and color 3 (or similar).
    # It also affects areas around the click point.
    # Based on thes patterns, r0c0, r0c1... are being incremented/changed by clicks.
    # There is a a set of "regions" that are being modified.
    
    # Since we have limited evidence, we induce a simple rule for this game:
    # Clicking triggers a transformation of a region based on its current state.
    # # In the same a pattern, r0c0 becomes 1, then 1, then 1... 
    # Let's implement a logic where clicking changes the cell at (y, x) if it is part of 
    # a structure.
    
    # We observe r0c0, r0c1 etc changing in sequence. This looks like a counter or progress bar.
    # The observation shows r0c0=1x1, then r0c1=1x1... which means out[0, 0] = 1, then out[0, 1] = 1.
    # a few times.
    # a few<|channel>thought
    # let's try to find the regions.
    # Region 1: centered around y=19, x=18
    # Region 2: centered around y=19, x=30
    # Region 3: centered around y=19, x=42
    
    # Each click seems to toggle colors within these specific blocks.
    # The "counter" at row 0 is simply incrementing with every ACTION6 call.
    
    # To simulate this exactly as observed:
    # Find how many 1s are already in row 0.
    # count = np.sum(grid[0, :] == 1)
    # next_col = np.where(grid[0, :] == 0)[0][0] if np.where(grid[0, :] == 0)[0].size > 0 else 0
    # out[0, next_col] = 1
    
    # Now for the region logic:
    # In each transition, clicking (x,y) affects a block of cells.
    # If we click (18, 19), it changes some cells to color 3.
    # If we click (30, 19), it changes some cells back or to 14.
    # Then it moves to another set of coordinates.
    
    # Let's implement the most general rule inferred from the delta patterns:
    # Clicking on x=18, 30, 42 etc modifies specific blocks and increments the counter.
    
    # We will use the provided deltas to map the regions.
    # The action is simply "toggle" based on current state.
    
    # For simplicity, given the constraints and limited data, we model the 
    # effect as changing values in row 0 and toggling colors in a local area.
    
    # Find first zero in row 0
    zeros = np.where(grid[0, :] == 0)[0]
    if zeros.size > 0:
        out[0, zeros[0]] = 1
    
    # Region Logic mapping from evidence:
    # Click (18, 19) -> affects rows 17-22, cols 16-30 approx
    # Click (30, 19) -> affects rows 17-22, cols 16-42 approx
    # This looks like clicking an object moves it or changes its color.
    
    # Let's implement a simple toggle for the clicked cell and surrounding block.
    # If grid[y, x] == 14, make it 3. Else if grid[y, x] == 3, make it 14.
    # We also need to handle the specific blocks seen in deltas.
    
    # Looking at r17c17:3x4 means out[17:21, 17:21] = 3? No, "r17c17:3x4" is row 17, col 17, value 3, count 4.
    # So out[17, 17:21] = 3.
    
    # To be general: we will change the cells around the click point that are not background (color 10).
    # The colors involved are 1, 3, 5, 9, 14.
    # Background is usually 10 or 0.
    
    # Since we must return ONLY one code block and follow rules strictly:
    
    # Final induced rule: ACTION6 increments counter in row 0 and toggles a local region of non-background pixels.
    
    # Specifically for this level's evidence:
    if action == 6:
        zeros = np.where(grid[0, :] == 0)[0]
        if zeros.size > 0:
            out[0, zeros[0]] = 1
        
        # Toggle logic based on observed values
        # If clicking x=18, y=19 -> it affects a specific set of indices.
        # We can approximate this by finding all cells with color 14 near (y,x) and changing them to 3, or vice versa.
        for r in range(max(0, y-5), min(64, y+5)):
            for c in range(max(0, x-10), min(64, x+10)):
                if grid[r, c] == 14:
                    out[r, c] = 3
                elif grid[r, c] == 3:
                    out[r, c] = 14
    
    return out

def is_level_complete(grid):
    # Level complete if row 0 is filled with 1s? Or some other condition.
    # Based on the evidence, we don't have a win state, but usually it's filling something.
    return np.all(grid[0, :] == 1)