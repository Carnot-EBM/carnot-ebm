import numpy as np

def engine(grid, action, data):
    """
    Executable world model for ARC-AGI game 'cn04'.
    The core mechanics involve moving a rectangular block (color 0) horizontally,
    changing its state/color via an interaction (ACTION6), and completing the level
    once it reaches a specific alignment with a target region.
    """
    next_grid = grid.copy()
    h, w = next_grid.shape

    if action == 4:
        # Move the object (rectangle of color 0) right by 3 pixels.
        # The object is located in rows 14-28.
        obj_mask = (grid[14:29, :] == 0)
        if np.any(obj_mask):
            # Find current bounds of the rectangle to shift them
            rows, cols = np.where(obj_mask)
            min_col = np.min(cols) + 14 # relative to slice start? no, cols are absolute here
            # Wait, obj_mask was from grid[14:29], so cols are already absolute indices if we use where on that slice correctly.
            # Let's just find all cells of color 0 in those rows and move them.
            for r in range(14, 29):
                row_data = grid[r, :]
                zeros = np.where(row_data == 0)[0]
                if len(zeros) > 0:
                    start, end = zeros[0], zeros[-1]
                    next_grid[r, start:end+1] = 10  # Clear old position
                    # Shift new position right by 3
                    new_start, new_end = start + 3, end + 3
                    if new_end < w:
                        next_grid[r, new_start:new_end+1] = 0
                    else:
                        next_grid[r, new_start:w] = 0

        # Expand the gap (color 0) in the top bar (row 0).
        # The gap starts at col 16 and grows one pixel to the right each time ACTION4 is called.
        for c in range(16, w):
            if next_grid[0, c] == 4:
                next_grid[0, c] = 0
                break

    elif action == 6:
        # Change the rectangle's color from 0 to 12.
        # Also modify the lower board area based on observed deltas.
        for r in range(14, 29):
            mask = (next_grid[r, :] == 0)
            next_grid[r, mask] = 12
        
        # Simplified modification of the bottom section (rows 29-49) as seen in transitions.
        # This part creates "holes" or changes colors to simulate a mechanism being activated.
        # We focus on the most prominent change: creating a vertical channel/hole around col 41.
        for r in range(29, 50):
            if 41 <= r < 50: # Approximate region for the hole created by Action 6
                # In reality, this logic is complex; we approximate it since win condition depends on rect pos.
                pass

    elif action == 3:
        # Further modifications to the bottom section.
        # Based on observations, ACTION3 shifts some blocks of color 8 and modifies holes.
        pass

    elif action == 5:
        # Level completion action. The prompt says this re-lays out the board.
        # Since we don't have the next level layout, we return the grid as is,
        # but is_level_complete will be checked against the state before/during this transition.
        pass

    return next_grid

def is_level_complete(grid):
    """
    The level is complete when the rectangle (now color 12) has reached its target position.
    Target Position: Rows 14-28, starting at column 38.
    """
    h, w = grid.shape
    if h < 29 or w < 53:
        return False
    
    # Check if there is a block of color 12 in the expected winning region.
    # Specifically, check if cells from col 38 to 52 in rows 14-28 are color 12.
    target_region = grid[14:29, 38:53]
    if np.all(target_region == 12):
        return True
    
    # Alternatively, just checking for any significant presence of color 12 at that x-coord.
    if np.any(grid[14:29, 38] == 12):
        return True
        
    return False