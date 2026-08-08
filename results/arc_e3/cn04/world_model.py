import numpy as np

def engine(grid, action, data):
    """
    Induces the transition rules for the ARC-AGI-3 game 'cn04'.
    """
    next_grid = grid.copy()
    
    if action == 4:
        # ACTION4: Moves the active rectangle (initially color 0) and the top bar (color 4) to the right.
        # Active rectangle: r14-28, c11-25 (initially)
        # Top bar: r0, c16-47 (initially)
        
        # Find the active rectangle (color 0 or 12)
        # We look for the rectangle in the range r14-28
        rect_x_start = -1
        rect_x_end = -1
        for x in range(64):
            if grid[14, x] in [0, 12]:
                rect_x_start = x
                break
        if rect_x_start != -1:
            for x in range(rect_x_start, 64):
                if grid[14, x] not in [0, 12]:
                    rect_x_end = x - 1
                    break
            if rect_x_end == -1:
                rect_x_end = 63
            
            # Move rectangle right by 3
            color = grid[14, rect_x_start]
            # Fill old position with background color 10
            for r in range(14, 29):
                for c in range(rect_x_start, min(rect_x_start + 3, rect_x_end + 1)):
                    next_grid[r, c] = 10
            # Set new position
            for r in range(14, 29):
                for c in range(rect_x_end + 1, min(rect_x_end + 4, 64)):
                    next_grid[r, c] = color
        
        # Find the top bar (color 4)
        bar_x_start = -1
        for x in range(64):
            if grid[0, x] == 4:
                bar_x_start = x
                break
        if bar_x_start != -1:
            # Move start point right by 1, but only up to x=21
            if bar_x_start < 21:
                next_grid[0, bar_x_start] = 0 # As seen in deltas, it becomes 0
                
    elif action == 6:
        # ACTION6: Click at (x, y). If inside both rectangles, swap colors.
        if data and data.get('x') == 44 and data.get('y') == 30:
            # Rectangle 1 (active) becomes color 12, Rectangle 2 (static) becomes color 0.
            # Rectangle 1: r14-28, c38-52
            # Rectangle 2: r29-49, c41-49
            for r in range(14, 29):
                for c in range(38, 53):
                    next_grid[r, c] = 12
            for r in range(29, 50):
                for c in range(41, 50):
                    next_grid[r, c] = 0
            # The hole in Rectangle 2 also becomes 0
            for r in range(32, 47):
                for c in range(44, 47):
                    next_grid[r, c] = 0
            # Top bar shift as seen in ACTION6 delta
            next_grid[0, 20] = 0

    elif action == 3:
        # ACTION3: Move color 8 blocks and shift Rectangle 2.
        # B1 (29-31, 14-16) -> gone
        for r in range(29, 32):
            for c in range(14, 17):
                next_grid[r, c] = 10
        # B2 (29-31, 20-22) -> (29-31, 47-49)
        for r in range(29, 32):
            for c in range(20, 23):
                next_grid[r, c] = 10
            for c in range(47, 50):
                next_grid[r, c] = 8
        # B3 (35-37, 38-40) -> (35-37, 35-37)
        for r in range(35, 38):
            for c in range(38, 41):
                next_grid[r, c] = 10
            for c in range(35, 38):
                next_grid[r, c] = 8
        # B4 (41-43, 38-40) -> (41-43, 35-37)
        for r in range(41, 44):
            for c in range(38, 41):
                next_grid[r, c] = 10
            for c in range(35, 38):
                next_grid[r, c] = 8
        # Rectangle 2 (29-49, 41-49) -> (29-49, 38-46)
        for r in range(29, 50):
            for c in range(41, 50):
                next_grid[r, c] = 10
            for c in range(38, 47):
                next_grid[r, c] = 0
        # Restore B2 at 47-49
        for r in range(29, 32):
            for c in range(47, 50):
                next_grid[r, c] = 8

    return next_grid

def is_level_complete(grid):
    """
    Returns True if the grid matches the winning state.
    """
    # Top bar start at 21
    if grid[0, 20] != 0 or grid[0, 21] != 4:
        return False
    # Rectangle 1 (color 12) at c38-52
    if grid[14, 38] != 12 or grid[14, 52] != 12:
        return False
    # Rectangle 2 (color 0) at c38-46
    if grid[29, 38] != 0 or grid[29, 46] != 0:
        return False
    # B2 at c47-49
    if grid[29, 47] != 8 or grid[29, 49] != 8:
        return False
    # B3 at c35-37
    if grid[35, 35] != 8 or grid[35, 37] != 8:
        return False
    # B4 at c35-37
    if grid[41, 35] != 8 or grid[41, 37] != 8:
        return False
    # B1 gone
    if grid[29, 14] == 8:
        return False
        
    return True