import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Right movement of a specific object/entity.
    # ACTION5: Left movement or state change? Let's look at the same column indices.
    # ACTION1: Upward movement of some elements.
    # The game seems to be a puzzle where objects move in directions.
    # Based on the observations, it's a<|channel>thought
    # that certain blocks are moving across the grid.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Find all cells that are not color 5 (background) and shift them right if possible.
        # We need to identify which entities are "movable"
        # In the delta, we see columns shifting by 3 units in many rows.
        # This suggests a block-based movement.
        # Simple implementation: find non-5 values and shift them.
        # However, the deltas show very specific shifts.
        # Let's try a simple approach: shift everything not background right by 3.
        mask = (grid != 5)
        shifted_mask = np.roll(mask, 3, axis=1)
        # To avoid wrapping, we handle boundaries.
        # But let's look at the data again. It's more like a sliding puzzle.
        # For ACTION 4, column indices increase by 3.
        # The logic is likely: for every cell (r, c), if grid[r, c] != 5, move it to (r, c+3).
        # If (r, c+3) is occupied or out of bounds, it might be blocked.
        # But the observed transitions show multiple blocks moving together.
        
        # Refined Action 4: Shift all non-background cells right by 3.
        # We must clear old positions first.
        temp_grid = new_grid.copy()
        for r in range(new_grid.shape[0]):
            for c in range(new_grid.shape[1]-1, -1, -1):
                if temp_grid[r, c] != 5:
                    val = temp_grid[r, c]
                    if c + 3 < new_grid.shape[1]:
                        new_grid[r, c+3] = val
                    new_grid[r, c] = 5
        return new_grid

    if action == 5: # Move Left
        temp_grid = new_grid.copy()
        for r in range(new_grid.shape[0]):
            for c in range(new_grid.shape[1]):
                if temp_grid[r, c] != 5:
                    val = temp_grid[r, c]
                    if c - 3 < 0:
                        pass # blocked
                    else:
                        new_grid[r, c-3] = val
                    new_grid[r, c] = 5
        return new_grid

    if action == 1: # Move Up
        # In ACTION 1 deltas, we see rows shifting up by 3 or something similar.
        # Let's try shifting non-background cells up by 3.
        temp_grid = new_grid.copy()
        for r in range(new_grid.shape[0]):
            for c in range(new_grid.shape[1]):
                if temp_grid[r, c] != 5:
                    val = temp_grid[r, c]
                    if r - 3 >= 0:
                        new_grid[r-3, c] = val
                    new_grid[r, c] = 5
        return new_grid

    return new_grid

def is_level_complete(grid):
    # The win state usually involves a specific configuration or clearing the board.
    # Looking at the INITIAL GRID and transitions, there are some values like '1' (color 1)
    # that might be targets.
    # For now, let's check if any cell has color 1 in a specific area or if most of the grid is background.
    # Based on common ARC patterns, it might be when certain blocks reach a target.
    # In the observed data, we don't have a WIN STATE grid, but ACTION4/ACTION1 continue.
    # Let's assume completion happens when all non-background cells are gone or aligned.
    # Since no win state was provided, this is a guess.
    # A common pattern is checking for a specific value (like 1) at a specific location.
    # Or simply return False unless a clear condition is met.
    return False