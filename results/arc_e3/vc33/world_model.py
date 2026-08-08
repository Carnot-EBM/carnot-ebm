import numpy as np

def engine(grid, action, data):
    """
    Induces the transition rules for the ARC-AGI-3 game 'vc33'.
    The game involves pushing a wall and objects by clicking at (61, 33).
    """
    grid = grid.copy()
    
    if action == 6 and data == {'x': 61, 'y': 33}:
        # 1. Top half (rows 1-27): Wall (color 3) expands right by 4 columns.
        for r in range(1, 28):
            # Find the rightmost column of the color 3 region.
            # The wall starts at col 0 and extends to some column c.
            c = -1
            for col in range(63, -1, -1):
                if grid[r, col] == 3:
                    c = col
                    break
            if c != -1:
                for i in range(c + 1, min(c + 5, 64)):
                    grid[r, i] = 3
        
        # 2. Bottom half (rows 32-63): Wall (color 3) shrinks from the right by 4 columns.
        for r in range(32, 64):
            c = -1
            for col in range(63, -1, -1):
                if grid[r, col] == 3:
                    c = col
                    break
            if c != -1:
                # Shrink the wall by setting the rightmost 4 cells of the wall to 0.
                for i in range(max(0, c - 3), c + 1):
                    grid[r, i] = 0
            
            # 3. Bottom half: Objects (color 4, 11) shift left by 4 columns.
            # These objects are specifically in rows 44-49.
            if 44 <= r < 50:
                # Identify cells that are color 4 or 11.
                obj_mask = (grid[r] == 4) | (grid[r] == 11)
                # Create a new row for the shifted objects.
                new_row = np.zeros_like(grid[r])
                # The wall (color 3) also needs to be preserved in the shifted row.
                # But the wall is already handled by the shrink logic.
                # Let's just shift the objects.
                for col in range(64):
                    if obj_mask[col]:
                        if col - 4 >= 0:
                            new_row[col - 4] = grid[r, col]
                
                # Now we need to merge the shifted objects with the wall.
                # The wall in these rows is everything to the left of the objects.
                # Let's find the new wall boundary.
                wall_end = -1
                for col in range(63, -1, -1):
                    if grid[r, col] == 3:
                        wall_end = col
                        break
                
                # Re-apply the wall to the new row.
                for col in range(0, wall_end + 1):
                    new_row[col] = 3
                
                # Update the grid row.
                grid[r] = new_row

        # 4. Top row (r=0): Progress indicator.
        # Initial: all 7. 1st click: col 63=4. 2nd click: col 61-63=4. 3rd click: col 61-63=7.
        if grid[0, 63] == 7:
            grid[0, 63] = 4
        elif grid[0, 63] == 4 and grid[0, 61] == 7:
            grid[0, 61:64] = 4
        elif grid[0, 61] == 4:
            grid[0, 61:64] = 7

    return grid

def is_level_complete(grid):
    """
    The level is completed after 3 clicks. 
    After 3 clicks, the top wall (rows 1-27) has expanded from col 31 to col 43.
    """
    # Check if the top wall has reached the target width (col 43).
    if grid.shape[0] > 1 and grid.shape[1] > 43:
        return grid[1, 43] == 3
    return False