import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The game seems to be a "collection" or "clearing" puzzle where clicking on specific 
    # target areas changes the same size blocks of colors.
    # In this observed sequence, clicking at (x, y) triggers a change in a 3x3 block centered at (y, x).
    # Note: Pixel coords are logical*1, so x=pixel_x, y=pixel_y.
    # Grid coordinates are row=y, col=x.
    
    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # Target area is a 3x3 block centered at (py, px)
    # We assume based on the delta that clicking a cell modifies a 3x3 region.
    # The same color block was replaced by another color.
    # # Example: ACTION6 data={'x': 10, 'y': 53} -> r52c9:15x3 r53c9:15x3 r54c9:15x3
    # # This means rows 52, 53, 54 and cols 9, 10, 11 were changed to 15.
    # # Changed cells are actually shifted from center (53, 10).
    # # Center of 3x3 is (53, 10), top-left is (52, 9).
    
    # Define the 3x3 window
    r_start, c_start = py - 1, px - 1
    r_end, c_end = py + 2, px + 2
    
    # Apply change to the 3x3 block
    # In the observed transitions, it's not just the 3x3 block at click location, but also some other blocks.
    # Therethought: clicking a target area triggers a "block" move or swap.
    # Let's look at the delta again.
    # ACTION6 data={'x': 10, 'y': 53} -> r52c9:15x3 r53c9:15x3 r5x4c9:15x3 AND r58c3:5x3 r59c3:5x3 r60c3:5x3
    # It seems like there's a<|channel>thoughtthought: Clicking on a specific cell changes its neighborhood and potentially another region.
    # However, for most ARC games of this type, we can simulate the local effect.
    # The core logic is likely that clicking (px, py) turns a 3x3 block into color 15.
    # The secondary effects (like clearing others) are part of the game mechanics.
    
    # Based on the provided deltas, the clicked 3x3 block becomes color 15.
    # # Example 1: x=10, y=53 -> rows 52-54, cols 9-11 become 15.
    # # Example 2: x=16, y=47 -> rows 46-48, cols 15-17 become 15.
    # # Example 3: x=22, y=41 -> rows 40-42, cols 21-23 become 15.
    # # Example 4: x=28, y=35 -> rows 34-36, cols 27-29 become 15.
    # # Example 5: x=34, y=29 -> rows 28-30, cols 33-35 become 15.
    
    # Let's implement this local effect and a simple "clearing" rule for previous blocks.
    # In each transition, the new 3x3 block is set to 15, and the *previous* 3x3 block (from the same sequence) is reset to its background color (which seems to be 5).
    # The observed deltas show that when clicking at (16, 47), the block at (53, 10) which was 15 becomes 5 again (r52c9:5x3 r53c9:5x1,3x1,5x1 r54c9:5x3).
    # Wait, r53c9:5x1,3x1,5x1 means col 9 is 5, col 10 is 3, col 11 is 5. This looks like it restores the original grid state.
    
    # To simulate this perfectly, we need to know the initial grid or restore values.
    # Since engine() must be pure and deterministic based on current grid, let's assume the action simply toggles/sets the clicked area to 15 and resets others.
    
    # But a simpler interpretation: Clicking (px, py) sets the 3x3 centered there to 15, and if any other 3x3 block of 15s exists, it might be cleared.
    
    # Let's refine:
    # 1. Find all existing 3x3 blocks of color 15.
    # 2. Reset them to their "original" colors (which are often 5 in these areas).
    # 3. Set the new 3x3 block at (py, px) to 15.
    
    # For simplicity and given the constraints, setting the target 3x3 to 15 and resetting previous ones to background 5 is the most likely general rule.
    
    for r in range(py - 1, py + 2):
        for c in range(px - 1, px + 2):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                new_grid[r, c] = 15
    
    # To handle the "reset" observed in deltas:
    # Look for other 3x3 blocks of 15 and reset them.
    # Since we don't have the original grid, we use a default background color 5.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15:
                # If this cell is not part of the newly placed block, reset it.
                if not (py-1 <= r < py+2 and px-1 <= c < px+2):
                    new_grid[r, c] = 5

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly provided as a full grid, but usually involves 
    # clearing all targets or reaching a specific configuration.
    # In many ARC games, if no more target colors are left or a certain pattern is formed, it's complete.
    # Given the data, there's no clear win condition, so we return False unless a common one is met.
    return False