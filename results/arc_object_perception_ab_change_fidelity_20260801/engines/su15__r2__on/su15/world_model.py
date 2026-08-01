import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a "collect" or "toggle" mechanism where clicking an area 
    # creates a 3x3 square of color 15 and replaces some other elements.
    # Based on the observed transitions, ACTION6 at (x, y) creates a 3x3 square of color 15 centered at (py, px).
    # It also seems to actually move or spawn a 3x3 square of color 15 that was previously existing elsewhere.
    # Also, it seems to<|channel>thought
    # Let's look at the delta: r46c15:15x3 r47c15:15x3 r48c15:15x3 for x=16, y=47.
    # This is exactly a 3x3 block of color 15 centered at (47, 16).
    # When this happens, the previous 3x3 block of color 15 (which was at 58, 3) is replaced by color 5.
    # And a small change occurs at the bottom row (r63), which looks like a progress bar or counter.
    
    # Find current position of the 3x3 block of color 15 own object type
    # We are looking for the "player" block (the 3x3 block of color 15).
    # Scan for any 3x3 block of color 15.
    found_block = None
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                found_block = (r, c)
                break
        if found_block: break

    # If no 3x3 block exists, we can't move it.
    if not found_block:
        return new_grid

    # Remove old block
    old_r, old_c = found_block
    new_grid[old_r:old_r+3, old_c:old_c+3] = 5
    
    # Place new block at clicked coordinates
    # The data x, y is likely center or top-left. Looking at ACTION6 x=10, y=53 -> r52c9:15x3...
    # This means py=53, px=10 corresponds to the center of the 3x3 block.
    # Top left would be (py-1, px-1).
    nr, nc = py - 1, px - 1
    
    # Clip to grid boundaries
    nr = max(0, min(nr, grid.shape[0] - 3))
    nc = max(0, min(nc, grid.shape[1] - 3))
    
    new_grid[nr:nr+3, nc:nc+3] = 15
    
    # Handle "collection" of objects color 3
    # If the new block overlaps any cells of color 3, those cells are replaced by 15.
    # But looking at the deltas, it seems if we click on a cell of color 3, it's consumed.
    # In ACTION6 x=16, y=47, the delta says r47c15:15x3 but also mentions r52c9:5x3 etc.
    # Wait, the delta for x=16, y=47 is: r46c15:15x3 r47c15:15x3 r48c15:15x3 (the new block)
    # and r52c9:5x3 r53c9:5x1,3x1,5x1 r54c9:5x3.
    # This means the previous block was at (52, 9). The center cell of that old block (53, 10) was color 3?
    # No, look at INITIAL GRID: r53 has 0x3 starting at col 9. So there were zeros there.
    # Let's re-examine: in the first transition, clicking (10, 53) moved the block from (58, 3) to (52, 9).
    # And it changed r63c62:5x2.
    # In the second transition, clicking (16, 47) moved the block from (52, 9) to (46, 15).
    # And it changed r63c60:5x2.
    # It seems every time we move the block, a pair of cells at the end of row 63 changes from 0 to 5.
    
    # Update progress bar at r63
    # Find the rightmost 0s and change them to 5s.
    # We only do this if the block actually moves.
    if (nr, nc) != (old_r, old_c):
        # The deltas show r63c62:5x2 then r63c60:5x2 etc.
        # This is moving backwards from the right edge.
        # Current state of r63 can be inferred. Let's just find the last two zeros.
        for c in range(grid.shape[1] - 1, 1, -1):
            if new_grid[63, c] == 0 and new_grid[63, c-1] == 0:
                new_grid[63, c] = 5
                new_grid[63, c-1] = 5
                break

    return new_grid

def is_level_complete(grid):
    # Level complete when all target objects are collected or a certain condition is met.
    # Looking at the transitions, we are filling row 63.
    # It might be complete when row 63 is mostly filled with 5s.
    # Or maybe when no more color 3 objects exist.
    # Let's check for any remaining color 3 pixels.
    return not np.any(grid == 3)