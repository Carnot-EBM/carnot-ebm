import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The game seems to be a "collection" or "toggling" puzzle where clicking on specific coordinates
    # triggers changes in the same area and potentially affects other elements.
    # Based on the<|channel>thought process, it's a sequence of clicks that clears items.
    # We observe that clicking at (x, y) creates a 3x3 block of color 15 (magenta own)
    # and replaces some existing blocks of color 3 (green) or others.
    # It looks like the target cells are the single pixels of color 3.
    # Let's implement the logic based on the observed transitions.
    
    y, x = data['y'], data['x']
    new_grid = grid.copy()
    
    # Create a 3x3 block of color 15 starting from (y-1, x-1)
    # This matches the delta patterns: r4c30:15x3, etc.
    # In the initial grid, we have several 3x3 blocks of color 15 already present.
    # The action is likely creating/removing these blocks.
    # Clicking on a pixel of color 3 seems to be the core mechanic.
    # { 'x': 10, 'y': 53 } -> r52c9:15x3, r53c9:15x3, r54c9:15x3
    # These coordinates correspond exactly to (y-1, x-1).
    
    # Apply the change: set a 3x3 area around the click point to color 15.
    # Note: The provided deltas show that if a 3x3 block was there, it might be removed.
    # Toggle logic: If center is 15, maybe remove? But here they are being added.
    # Let's check the target cells (color 3 pixels).
    # Initial Grid: r52c10:0x1, r53c9:0x3, r54c10:0x1. This looks like a "hole" or specific shape.
    # Action6 data={'x': 10, 'y': 53} -> r52c9:15x3... which fills this hole with magenta.
    #
    # Based on the observed transitions, clicking at (x, y) creates a 3x3 block of color 15
    # centered at (y, x), but shifted by -1 in both directions.
    # So top-left is (y-1, x-1).
    #
    # Let's implement the simple rule: Clicking at (x, y) sets a 3x3 area around (y, x) to 15.
    # Also, we observe some other changes: r58c3:5x3, etc. and r63c62:5x2.
    # These look like side effects or progress indicators.
    # We will focus on the primary change.
    
    # Primary effect: 3x3 block of color 15 starting at (y-1, x-1)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                new_grid[ny, nx] = 15
    
    # The deltas also show that previous blocks are changed back to color 5?
    # e.g., ACTION6 data={'x': 16, 'y': 47} -> r46c15:15x3... AND r52c9:5x3...
    # This means only ONE 3x3 magenta block can exist at a time, or they move.
    # It's more likely that clicking "moves" the active block.
    
    # Find existing 3x3 magenta blocks and reset them to their background color.
    # Background colors seem to be 4 or 5.
    # We'll use a simple heuristic: if it was 15, change it back to the surrounding color.
    # For simplicity, we'll assume background is 5 for most of the board.
    
    # Let's refine: find all cells that were 15 and aren't in the new 3x3 area.
    # Reset them to 5 (or 4).
    for ry in range(grid.shape[0]):
        for rx in range(grid.shape[1]):
            if grid[ry, rx] == 15:
                # Check if this cell is part of the new 3x3 block
                if not (y-1 <= ry < y+2 and x-1 <= rx < x+2):
                    new_grid[ry, rx] = 5 if ry > 9 else 4
    
    return new_grid

def is_level_complete(grid):
    # A level is typically complete when certain targets are cleared or a pattern is reached.
    # In these games, often the "win" state is when no more target pixels (color 3) remain.
    # Or when a specific number of blocks have been collected.
    # Since we don't have a win state example, we check if any color 3 pixels are left.
    return np.sum(grid == 3) == 0