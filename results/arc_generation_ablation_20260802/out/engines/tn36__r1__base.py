import numpy as np

import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on certain areas (ACTION6)
    # triggers changes in two distant locations: a cell at the top (r1) and 
    # specific blocks of color 0 or 1 in the bottom area (r42-r46).
    # Based on the same pattern:
    # Click (24, 41) -> r1c61=3, r42c25:c27=5
    # Click (34, 41) -> r1c59=3, r42c35:c37=5
    # Click (39, 41) -> r1c57=3, r42c40:c42=5
    # Click (24, 44) -> r1c60=3, r44:r46 c26=5
    # Click (34, 44) -> r1c58=3, r44:r46 c36=5
    # Click (39, 44) -> r44:r46 c36=5? No, wait.
    
    # Let's map coordinates to effects.
    # x is col, y is row.
    # The target cells are usually color 0 or 1.
    # We look for where the click happened and if it was a cell of a certain color.
    # If we clicked on a cell that is part of a "button" or "switch", it changes state.
    # In this case, the same coordinate clicks result in specific delta updates.
    
    # grid[y, x] = value
    # The observed data shows clicking at (x, y) modifies the grid.
    #<|channel>thought
    # The logic seems to be: when you click a cell (x, y), if it's not background (color 5),
    # it might trigger a change.
    # The targets are always changing from some value to 5.
    # The top bar (r1) is being filled with color 3.
    # The bottom area (r42-r46) is also being filled with color 5.
    
    # Since we need a general rule, let's check if the click point itself is modified.
    # No, the click points (24, 41), (34, 41), etc., are not mentioned as changed cells.
    # Instead, they act as triggers.
    
    # Let's look at the coordinates again:
    # Click (24, 41): r42c25=5, r42c26=5, r42c27=5 AND r1c61=3
    # Click (34, 41): r42c35=5, r42c36=5, r42c37=5 AND r1c59=3
    # Click (39, 41): r42c40=5, r42c41=5, r42c42=5 AND r1c57=3
    # Click (24, 44): r44c26=5, r45c26=5, r46c26=5 AND r1c60=3
    # Click (34, 44): r44c36=5, r45c36=5, r46c36=5 AND r1c58=3
    
    # It looks like clicking a cell of color 0 or 1 in the bottom region triggers 
    # filling that specific "gap" with background color 5 and marking progress at the top.
    
    new_grid = grid.copy()
    if action == 6:
        x, y = data['x'], data['y']
        # If we click on a non-background cell in the trigger area
        if grid[y, x] != 5:
            # Find the connected component of the clicked cell to fill it
            # For simplicity, based on observed deltas, we just fill the same value as the target
            # The targets are always changing to 5.
            # We need to find which 'progress' cell to mark in row 1.
            # Row 1 has cells from c0 to c63. Progress seems to move right-to-left starting from c61.
            
            # Let's implement a simple flood fill for the clicked point to turn it into 5.
            stack = [(y, x)]
            target_color = grid[y, x]
            while stack:
                curr_y, curr_x = stack.pop()
                if 0 <= curr_y < new_grid.shape[0] and 0 <= curr_x < new_grid.shape[1]:
                    if new_grid[curr_y, curr_x] == target_color:
                        new_grid[curr_y, curr_x] = 5
                        stack.append((curr_y + 1, curr_x))
                        stack.append((curr_y - 1, curr_x))
                        stack.append((curr_y, curr_x + 1))
                        stack.append((curr_y, curr_x - 1))
            
            # Now handle the progress bar at r1.
            # The observed sequence of clicks: (24, 41), (24, 44), (34, 41), (34, 44), (39, 41)
            # Resulting top cells: c61, c60, c59, c58, c57
            # This is a simple counter. We can find the rightmost cell in row 1 that is not 3 or 5.
            # Actually, it's filling from right to left starting at index 61.
            for col in range(61, -1, -1):
                if new_grid[1, col] != 3:
                    new_grid[1, col] = 3
                    break
    return new_grid

def is_level_complete(grid):
    # Level complete when all target areas are filled with color 5?
    # Or when the progress bar reaches a certain point.
    # In most ARC games, completion is when a specific pattern is achieved.
    # Here, we don't have a win state grid, but usually it's when no more 'targets' exist.
    # Let's check if any cells of color 0 or 1 remain in the trigger area.
    # Trigger area seems to be r8-r46 and r42-r46 specifically.
    # For now, return False unless we see a clear win condition.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for tn36 is that the grid contains only 
    one color (excluding background 0) and that color is 
    distributed in a way that matches the target pattern.
    """
    grid = np.array(grid)
    # Find all unique colors present in the grid, excluding background 0.
    colors = np.unique(grid)
    colors = colors[colors != 0]
    
    # The win condition is that there is exactly one non-zero color present.
    # If the grid is all zeros, it' same not a win state.
    if len(colors) == 0:
        return False
    
    # Check if the grid contains only one non-zero color.
    if len(colors) > 1:
        return False
    
    # Check if the grid contains a specific number of non-zero pixels.
    # This is a a general rule for this task.
    non_zero_count = np.count_nonzero(grid)
    if non_zero_count == 0:
        return False
    
    # For tn36, the win state is typically a single-color 
    # block or a specific pattern of a single color.
    # return True if the grid contains only one non-zero color.
    return True
