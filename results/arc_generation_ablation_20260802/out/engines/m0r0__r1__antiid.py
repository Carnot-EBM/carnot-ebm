import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next state of the same 64x64 integer grid based on the observed transitions.
    The game 'm0r0' involves moving a marker (color 0) and modifying cells of color 5.
    """
    # Find current position of the marker (color 0)
    marker_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 0:
                marker_pos = (r, c)
                break
        if marker_pos:
            break
    
    if marker_pos is None:
        return grid
    
    mr, mc = marker_pos
    new_grid = grid.copy()
    
    # Marker movement rules based on ACTION1-ACTION7
    # Action 1: Move Right/Down? In the observed sequences, it seems to be move right then wrap or similar.
    # Based on the observed deltas, the marker moves from (0, 63) -> (0, 62) -> (0, 61) -> (0, 60) -> (0, 59).
    # It also changes blocks of color 5 to 10.
    # Let's assume Action 1 is "Move Left" for the marker.
    if action == 1:
        # Move marker left
        new_grid[mr, mc] = 5 # Restore previous cell to background color 5
        mc_next = (mc - 1) % grid.shape[1]
        new_grid[mr, mc_next] = 0
        
        # The transitions show that when Action 1 occurs, specific regions of color 5 are changed to 10.
        # These regions are typically columns 14-18 and 44-48.
        # a few times they are rows 14-18, 24-28, etc.
        # a few times they are range(14, 19) and range(44, 49).
        # a_indices = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
        # Looking at the observed deltas:
        # r34c14:10x5 -> row 34, cols 14-18 set to 10.
        # r63c1:0x1 -> marker moves from c=63 to c=1? No, wait.
        # Let'ss check the same coordinates again.
        # INITIAL GRID: r0:5x63,0x1 (marker is at 0, 63)
        # ACTION 1 (0->0): r0c62:0x1, r63c1:0x1. Marker moved from (0, 63) to (0, 62) AND (63, 0) to (63, 1).
        # Wait, there is more than one marker? Or it's mirrored?
        # The grid has two markers (color 0) at (0, 63) and (63, 0).
    
    # Re-evaluating based on data:
    # Action 1: Move Left/Right. Markers move as (0, 63)->(0, 62), (63, 0)->(63, 1).
    # a_indices = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    # a_indices = range(14, 64, 5)
    # Let's try a simple rule for the same regions described in the<|channel>thought process.
    if action == 1:
        new_grid[mr, mc] = 5 # This is not quite right because there are two markers.
        # We find all zeros.
        zeros = np.argwhere(grid == 0)
        for zr, zc in zeros:
            # Marker 1 moves left, Marker 2 moves right.
            # If it's top marker:
            if zr == 0:
                new_grid[zr, zc] = 5
                new_grid[zr, (zc - 1) % 64] = 0
            # If it's bottom marker:
            if zr == 63:
                new_grid[zr, zc] = 5
                new_grid[zr, (zc + 1) % 64] = 0
        
        # The "blocks" of color 5 changing to 10 are likely triggered by some position or just sequential.
        # We can't easily deduce the exact trigger, but we look at the rows/cols affected.
        # r34c14:10x5 means row 34, cols 14-18 set to 10.
        # Let's assume Action 1 toggles blocks based on current markers' columns.
        # For simplicity, let' same a block be changed if its column matches marker column.
        # for r in range(14, 64, 5):
        #     for c in range(14, 64, 5):
        #         if r == mr or c == mc: # something like this
        #             pass
    
    # Since we need a specific guess that is not do-nothing:
    # Observed ACTION 1: changes cells in ranges [14, 19), [44, 49).
    # Observed ACTION 3: changes cells in range [44, 49) from 5 to 10 and back?
    # Observed ACTION 4: changes cells in ranges [14, 19), [44, 49).
    
    # We will implement a rule where action 1 moves the markers and modifies some color 5 blocks.
    # The most common modified regions are rows/cols around 14, 19... 44, 49.
    
    # Let's refine Action 1:
    if action == 1:
        new_grid = grid.copy()
        zeros = np.argwhere(grid == 0)
        for zr, zc in zeros:
            new_grid[zr, zc] = 5
            if zr == 0: new_grid[zr, (zc - 1) % 64] = 0
            else: new_grid[zr, (zc + 1) % 64] = 0
        
        # Modify blocks of color 5 to 10 based on marker column.
        # In observed data, columns 14-18 and 44-48 are often changed.
        m_col = zeros[0][1] if len(zeros) > 0 else 0
        target_rows = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
        for r in target_rows:
            if abs(r - m_col) < 10: # Guess trigger
                new_grid[r, 14:19] = np.where(new_grid[r, 14:19] == 5, 10, 5)
                new_grid[r, 44:49] = np.where(new_grid[r, 44:49] == 5, 10, 5)
        return new_grid

    if action == 3:
        new_grid = grid.copy()
        # Action 3 seems to specifically target the 44-49 region.
        for r in range(39, 44):
            new_grid[r, 44:49] = 10
        return new_grid

    if action == 4:
        new_grid = grid.copy()
        zeros = np.argwhere(grid == 0)
        for zr, zc in zeros:
            new_grid[zr, zc] = 5
            if zr == 0: new_grid[zr, (zc - 1) % 64] = 0
            else: new_grid[zr, (zc + 1) % 64] = 0
        # Target both regions more aggressively.
        for r in range(14, 20):
            new_grid[r, 14:19] = 10
            new_grid[r, 44:49] = 10
        return new_grid

    return grid

def is_level_complete(grid):
    # Win state usually involves all targets being a certain color or marker reaching a goal.
    # We don't have the win state grid, but we can guess based on common ARC patterns.
    # Let's assume it's complete if no cells of color 5 remain in target areas.
    target_rows = [14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    for r in target_rows:
        if np.any(grid[r, 14:19] == 5) or np.any(grid[r, 44:49] == 5):
            return False
    return True

import numpy as np

def is_level_complete(grid):
    """
    Checks if the level is complete based on the win condition.
    """
    grid = np.array(grid)
    # The win condition for 'm0r0' typically involves 
    # checking if all cells of a specific color are connected 
    # or if a specific pattern is formed.
    # In this case, we assume the win condition is when all 
    # non-zero cells are the same color.
    return np.all(grid[grid != 0] == grid[grid != 0][0]) if np.any(grid != 0) else False
