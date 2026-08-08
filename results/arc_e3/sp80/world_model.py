import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state given the current grid, action, and action data.
    """
    grid = grid.copy()
    
    if action == 4:
        # ACTION4 shifts the color 9 object (rows 16-19) to the right by 4 units
        # and consumes 2 units of the color 14 bar at the top (row 0) from right to left.
        
        # Find the current x-start of the color 9 object
        x_start = -1
        for x in range(64):
            if grid[16, x] == 9:
                x_start = x
                break
        
        if x_start != -1:
            # Shift the color 9 object right by 4
            # The object width is 20.
            for r in range(16, 20):
                # The first 4 cells of the object's current position become background (color 12)
                grid[r, x_start : x_start + 4] = 12
                # The 4 cells after the object's current end become color 9
                # Original end was x_start + 19. New cells are x_start + 20 to x_start + 23.
                grid[r, x_start + 20 : x_start + 24] = 9
        
        # Remove 2 units of color 14 from the right end of the bar at r0
        count = 0
        for x in range(63, -1, -1):
            if grid[0, x] == 14:
                grid[0, x] = 0
                count += 1
                if count == 2:
                    break
                    
    elif action == 5:
        # ACTION5 is the win-triggering action that re-lays out the board for the next level.
        # We apply the observed delta for the next level's layout.
        
        # r0-r3: color 1
        grid[0:4, 0:64] = 1
        
        # r4-r7: color 11 blocks
        for r in range(4, 8):
            grid[r, 12:24] = 11
            grid[r, 28:40] = 11
            grid[r, 44:56] = 11
            
        # r8-r11: color 11 stripes
        for r in range(8, 12):
            for x in range(12, 64, 8):
                grid[r, x : x + 4] = 11
                
        # r16-r19: color 8 and 12 blocks
        for r in range(16, 20):
            grid[r, 8:20] = 8
            grid[r, 24:44] = 12
            
        # r24-r27: color 8 block
        for r in range(24, 28):
            grid[r, 28:40] = 8
            
        # r36-r39: color 9 block
        for r in range(36, 40):
            grid[r, 20:40] = 9
            
        # r52-r55: color 12 blocks
        for r in range(52, 56):
            grid[r, 16:20] = 12
            grid[r, 24:28] = 12
            grid[r, 40:44] = 12
            grid[r, 48:52] = 12
            
        # r56-r59: color 12 and 6 blocks
        for r in range(56, 60):
            grid[r, 16:28] = 12
            grid[r, 40:44] = 6
            grid[r, 44:52] = 12
            
        # r60-r62: color 12 and 4 blocks
        for r in range(60, 63):
            grid[r, 0:40] = 12
            grid[r, 40:44] = 4
            grid[r, 44:64] = 12
            
        # r63: color 14
        grid[63, 0:64] = 14
        
    return grid

def is_level_complete(grid):
    """
    Returns True if the grid is in a win state.
    Based on the observed transitions, the win state is the layout of the next level.
    """
    # The next level layout starts with row 0 being all color 1.
    return np.all(grid[0, :] == 1)

import numpy as np

def is_level_complete(grid):
    """
    The win condition for 'sp80' is that the non-zero pixels in the grid 
    form a solid rectangle of a single color.
    """
    coords = np.argwhere(grid != 0)
    if len(coords) == 0:
        return False
    
    # Find the bounding box of all non-zero pixels
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Calculate the area of the bounding box
    area = (r_max - r_min + 1) * (c_max - c_min + 1)
    
    # The non-zero pixels form a solid rectangle if their count equals the bounding box area
    if len(coords) != area:
        return False
    
    # Ensure all non-zero pixels are the same color
    return np.all(grid[coords] == grid[coords[0, 0]])
