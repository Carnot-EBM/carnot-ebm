import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 4: # ACTION4 seems to be a right-shift of some specific pattern/objects
        new_grid = grid.copy()
        # Identify all cells that are not color 10 (background)
        # The observed transitions show a shift of objects moving right by 3 columns.
        # We need to find what's actually shifting.
        # In thisthought process, we umautomatize the same logic for the laout//
        # Based on the observations, it looks like everything except background (color 10) 
        # is shifted right by 3 units if they belong to certain "active" regions.
        # However, simpler rule: just shift everything non-10 to the right by 3?
        # No, the delta shows very specific changes.
        # Let's look at the deltas again. r14c11:10x3, r14c26:0x3...
        # It means cells from c11..13 became 10 and c26..28 became 0.
        # Wait, no. r14c11:10x3 means cols 11, 12, 13 become 10.
        # r14c26:0x3 means cols 26, 27, 28 become 0.
        # This is a classic "move object" pattern.
        # The objects are the non-10 colors.
        # Action 4 is 'Right'.
        
        # To implement this correctly, we find all non-10 blocks and move them.
        # We need to be careful about overlapping or boundaries.
        # In these examples, it seems like only some parts of the grid are moving.
        # Specifically, the area where the "maze" or "objects" are.
        # Let's try shifting all non-10 pixels right by 3 columns if they don't hit a boundary (color 4).
        # Color 4 is at the top border (r0) and potentially other places.
        # Actually, looking at the INITIAL GRID, color 4 is ONLY at r0, c16-47.
        # And there are walls/obstacles.
        # Let's just shift everything that isn't background (10) right by 3 units.
        # If it hits something, it might stop. But in the observed deltas, 
        # it looks like a simple translation of the whole active set.
        
        # Find all coordinates of cells that are not 10.
        coords = np.argwhere(grid != 10)
        new_grid[coords[:, 0], coords[:, 1]] = 10
        for r, c in coords:
            nc = c + 3
            if nc < grid.shape[1]:
                new_grid[r, nc] = grid[r, c]
            else:
                # Boundary check - if it goes off board, we can't move it.
                # In these examples, it doesn't seem to happen.
                pass
        return new_grid

    return grid

def is_level_complete(grid):
    # return True if `grid` is a level-complete / win state, else False.
    # Based on the provided data, no WIN STATE was given.
    # Usually, completion is when an object reaches a target or a certain color disappears.
    # Let's assume for now it's based on some condition.
    # Since we only have ACTION4 (Right), and maybe others?
    # The game might be about moving something into a slot.
    # Look at INITIAL GRID: there are slots of color 0.
    # For example, r14c11..15 are 0.
    # If all non-background pixels reach their targets, it's complete.
    # But without a win state, this is a guess.
    # A common pattern in ARC is that the same color as the background but different value (like 0)
    # represents a goal.
    # Let's check if any non-10 pixel has moved into a position where it was 0.
    # However, the simplest "win" is often just reaching a specific coordinate.
    # In these examples, the objects move right. Maybe they need to reach the far right?
    # Or perhaps the level is complete when the grid contains no more '0' cells?
    # Let's try: return True if no cell is 0.
    return not np.any(grid == 0)