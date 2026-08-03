import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific areas changes colors of blocks.
    # Based on the observed transitions, ACTION6 clicks change color values in rectangular regions.
    # Let's identify the patterns of changed cells.
    # Each click at (px, py) corresponds to a same-shaped "brush" or "stamp own" that modifies a region.
    # Color 5 is often replaced by 3 and vice versa, or 14 is replaced by 1.
    # Color 0 is usually unchanged unless it's part of the<|channel>thought process.
    # In the provided delta, we see things like r17c17:3x4, which means row 17, col 17-20 are now color 3.
    # For example, if we click at (18, 19), the modified area is around y=19, x=18.
    # Let's implement a simple rule based on the observations.
    # We actually need to find the logic for what happens when you click.
    # Since this is an ARC task, we umaze the pattern from the deltas.
    #
    # Looking at the first transition:
    # Click (18, 19) -> changes colors in rows 17-22, cols 16-32 approx.
    # It seems clicking a cell (px, py) affects a local neighborhood.
    # The specific values changed depend on the current value.
    #
    # However, looking closer at the observed transitions, there's a very strange thing:
    # "r0c0:1x1", "r0c1:1x1" etc. appear in every single ACTION6 call.
    # This looks like a counter or a state tracker in the top left corner of the grid.
    #
    # Let's refine the engine to handle these specific clicks and their effects.
    # Given the limited data, let's try to deduce if it's a toggle.
    # In the first few transitions, color 5 becomes 3, then 3 becomes 0? No.
    # Actually, look at r17c17:3x4. Color 3 replaces something.
    # Then later r17c17:0x4. Color 0 replaces that.
    #
    # Wait, the deltas are actually quite complex. Let' same as follow the pattern:
    # Clicking changes colors based on some internal logic.
    # Since we must provide an executable world model, and the patterns are highly repetitive,
    # let's assume clicking (px, py) toggles values in a region around (py, px).
    
    # Based on the observed deltas:
    # Transition 1: Click(18, 19) -> rows [17, 22], cols [16, 32] modified.
    # Transition 2: Click(30, 19) -> rows [17, 22], cols [16, 32] modified again.
    # It looks like the click coordinates (px, py) might be the center of a modification area.
    
    # Let's try to implement the "counter" part first.
    # The top-left cell grid[0,0] increments by 1 each time ACTION6 is called.
    new_grid[0, 0] = (grid[0, 0] + 1) % 256
    
    # Now for the main effect. Looking at the transitions, it seems that
    # when you click (px, py), you change colors in a specific pattern.
    # For example, if color was 14, it becomes 1. If it was 5, it becomes 3.
    # This looks like a "painting" or "toggling" game.
    
    # Since we don't have enough data to perfectly induce the brush shape,
    # let's use a simple approximation based on the deltas provided.
    # In transition 1, clicking (18, 19) affects cells around y=19, x=18.
    # Specifically: r17c17, r18c16, etc.
    # It seems the affected region is roughly [py-2, py+3] and [px-2, px+14].
    
    # Let's try this logic:
    # For every cell (y, x) in the grid:
    # if (y, x) is within some distance of (py, px):
    #   if grid[y, x] == 5: new_grid[y, x] = 3
    #   elif grid[y, x] == 3: new_grid[y, x] = 0 # or something else
    #   elif grid[y, x] == 14: new_grid[y, x] = 1
    #   elif grid[y, x] == 1: new_grid[y, x] = 14
    
    # Looking at Transition 1 vs 2:
    # Click(18, 19) -> r17c17 becomes 3x4.
    # Click(30, 19) -> r17c17 becomes 0x4.
    # This suggests a toggle: 5 -> 3 -> 0? Or maybe it depends on the click position.
    
    # Actually, looking at the deltas again:
    # Transition 1: Click(18, 19) -> r17c17:3x4
    # Transition 2: Click(30, 19) -> r17c17:0x4 (Wait, this is for the same area!)
    # This means clicking at (30, 19) affected the region that was modified by clicking at (18, 19).
    
    # Let's try to implement a simple "region-based" modification.
    # The regions seem to be blocks of size roughly 6x16 or similar.
    # If we click inside a block, all cells in that block are updated.
    
    # Given the complexity and limited data, let's focus on the most obvious pattern:
    # Clicking increments grid[0,0] and changes some colors.
    # Since I cannot deduce the exact brush/block logic perfectly,
    # I will implement a generic version that handles the observed color swaps.
    
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if abs(y - py) <= 3 and abs(x - px) <= 8:
                val = grid[y, x]
                if val == 5: new_grid[y, x] = 3
                elif val == 3: new_grid[y, x] = 0
                elif val == 14: new_grid[y, x] = 1
                elif val == 1: new_grid[y, x] = 14
                elif val == 0: new_grid[y, x] = 5
                
    return new_grid

def is_level_complete(grid):
    # Win state is usually when certain blocks are filled or a specific color is gone.
    # In this game, it's not explicitly shown, but often it's about clearing the board
    # or reaching a target pattern. Without a win-state example, we return False.
    return False