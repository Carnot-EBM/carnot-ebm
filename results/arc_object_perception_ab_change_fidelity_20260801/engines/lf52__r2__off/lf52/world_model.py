import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    y, x = data['y'], data['x']
    new_grid = grid.copy()
    
    # The game seems to be a "painting" or "toggling" mechanism where clicking on specific areas 
    # changes colors of blocks. Based on the observed transitions, ACTION6 clicks change color 
    # values in clusters.
    # Let's identify the cluster that was clicked and find its connected components of same-color cells.
    # 
    # This kind of pattern suggests a "flood fill" or "cluster toggle" toggle.
    # target_color = grid[y, x]
    # However, looking at the observations, the deltas are very specific.
    # Clicking (18, 19) changed cells around it.
    #<|channel>thought
    # Looking closer at the delta for Action 6: r0c0:1x1 is always present.
    # This means there's a global state counter or something similar being tracked in row 0.
    # Row 0 is mostly 0s. Every click increments the value at (0, col).
    # Col index corresponds to number of clicks.
    #
    # Now let's look at the block changes.
    # Click (18, 19): Changed cells in rows 17-22.
    # Click (30, 19): Changed cells in rows 17-22.
    # Click (30, 19) again? No, wait. The data says ACTION6 data={'x': 30, 'y': 19}.
    # Wait, the observed transitions list the same action twice but different results.
    # It looks like clicking on a cell toggles its color and potentially others.
    # Let's implement a simple toggle logic based on the patterns.
    
    # Find current click count from grid[0, :]
    click_count = np.sum(grid[0, :] != 0)
    new_grid[0, click_count] = 1
    
    # Identify the clicked area. In this game, it seems blocks are defined by boundaries of color 5.
    # We need to find the "room" containing (y, x).
    # Use flood fill to find all connected cells of any color except boundary color 5.
    visited = np.zeros_like(grid, dtype=bool)
    stack = [(y, x)]
    room = []
    while stack:
        curr_y, curr_x = stack.pop()
        if 0 <= curr_y < grid.shape[0] and 0 <= curr_x < grid.shape[1]:
            if not visited[curr_y, curr_x] and grid[curr_y, curr_x] != 5:
                visited[curr_y, curr_x] = True
                room.append((curr_y, curr_x))
                stack.extend([(curr_y+1, curr_x), (curr_y-1, curr_x), (curr_y, curr_x+1), (curr_y, curr_x-1)])
    
    # Based on deltas, clicking a room changes colors within that room.
    # Specifically, it seems to toggle between some values like 3 and 14 or 2 and 1.
    # Let's try a simple rule: if the cell is 14, make it 1; if it's 1, make it 14.
    # If it's something else, maybe change it to 3?
    for ry, rx in room:
        val = grid[ry, rx]
        if val == 14:
            new_grid[ry, rx] = 1
        elif val == 1:
            new_grid[ry, rx] = 14
        elif val == 0:
            # Only change 0s if they are part of a specific pattern? 
            # In the observations, cells with value 0 were changed to 3 or 14.
            pass

    # The observed transitions show very complex patterns. Let's refine based on own logic.
    # Actually, looking at the delta r17c17:3x4, this means rows 17-20, cols 17-19 become color 3.
    # This looks like "filling" a block.
    
    # Let's use a simpler heuristic for the engine since we don't have enough data to be perfect.
    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing blocks or reaching a certain configuration.
    # Given no win state provided, return False unless row 0 is full (as a guess).
    return np.sum(grid[0, :] != 0) >= 64