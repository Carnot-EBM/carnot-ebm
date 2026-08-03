import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    # The game seems to involve clicking on areas that trigger changes in other parts of the grid.
    # Based on thes transitions, clicks at y=59 (bottom area) seem to clear/set cells to 0.
    # If we click near the bottom blocks, we're essentially 'activating' or 'clearing' them.
    # Let's map the same x coordinates used in the<|channel>thought process.
    
    new_grid = grid.copy()
    
    # Clicks at y=30 (middle area) move colors from the bottom to the middle.
    if py == 30:
        # Map x to color and position
        # x=23 -> color 9, cols 21-24
        # x=29 -> color 14, cols 27-28
        # x=35 -> color 11, cols 33-36
        if px == 23:
            color = 9
            col_start = 21
            col_end = 25
            target_row_start = 28
            target_row_end = 32
            source_row_start = 56
            source_row_end = 62
            source_col_start = 33
            source_col_end = 39
        elif px == 29:
            color = 14
            col_start = 27
            col_end = 31
            col_start_src = 17
            col_end_src = 23
            target_row_start = 28
            target_row_end = 32
            source_row_start = 56
            source_row_end = 62
            source_col_start = 17
            source_col_end = 23
        elif px == 35:
            color = 11
            col_start = 33
            col_end = 37
            target_row_start = 28
            target_row_end = 32
            source_row_start = 56
            source_row_end = 62
            source_col_start = 41
            source_col_end = 47
        else:
            return new_grid

        # Apply changes to the middle area
        new_grid[target_row_start:target_row_end, col_start:col_end] = color
        
        # Clear source area at bottom
        new_grid[source_row_start:source_row_end, source_col_start:source_col_end] = 4
        
        # Increment a counter in the top right (r53c60s)
        if px == 23:
            new_grid[53, 63] = 3
        elif px == 29:
            new_grid[53, 62] = 3
        elif px == 35:
            new_grid[53, 61] = 3
        
    elif py == 59:
        # Clicks at y=59 clear specific blocks.
        # x=36 -> cols 33-38
        # x=20 -> cols 17-22
        # x=44 -> cols 41-46
        if px == 36:
            col_start, col_end = 33, 39
        elif px == 20:
            col_start, col_end = 17, 23
        elif px == 44:
            col_start, col_end = 41, 47
        else:
            return new_grid
        
        # Set cells to 0
        new_grid[56:62, col_start:col_end] = 0
        
    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually involves filling a target area or clearing others.
    # Based on the observed transitions, maybe it's when certain colors are moved to the middle.
    return False