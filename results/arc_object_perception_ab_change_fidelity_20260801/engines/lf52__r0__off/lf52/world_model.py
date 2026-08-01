import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific areas changes colors of blocks.
    # Based on the observed transitions, ACTION6 clicks change values in rectangular regions.
    # Looking at the same coordinates clicked (e.g., x=18, y=19), it's changing cells in rows 17-22 and cols 16-20.
    # This suggests a coordinate mapping or a region-based toggle.
    # Let'<|channel>thought
    # However, the delta shows that color 14 (yellow) is being replaced by color 1 (blue) or vice versa.
    # And there's some strange behavior with r0c0, r0c1... which looks like a counter or progress indicator.
    # Given the limited data, we can induce a simple rule:
    # Clicking an area toggles the value between 1 and 14 for certain patterns.
    # If the cell at (py, px) is part of a block, find the connected component of that block and toggle its color.
    # If the cell is not part of a block, do nothing.
    
    # We identify "blocks" as contiguous regions of the same color (excluding background/wall colors).
    # The most common colors are 5 (grey), 10 (black), 0 (white), 1, 14.
    # 10 is likely wall, 5 is boundary.
    # 1 and 14 are the interactive elements.
    
    target_color = grid[py, px]
    if target_color == 10 or target_color == 5 or target_color == 0:
        return new_grid

    # Simple flood fill to find the connected region of the same color
    region = []
    stack = [(py, px)]
    visited = set()
    while stack:
        curr_y, curr_x = stack.pop()
        if (curr_y, curr_x) in visited:
            continue
        visited.add((curr_y, curr_x))
        if 0 <= curr_y < grid.shape[0] and 0 <= curr_x < grid.shape[1]:
            if grid[curr_y, curr_x] == target_color:
                region.append((curr_y, curr_x))
                stack.append((curr_y + 1, curr_x))
                stack.append((curr_y - 1, curr_x))
                stack.append((curr_y, curr_x + 1))
                stack.append((curr_y, curr_x - 1))
    
    # Toggle logic based on observed transitions:
    # 14 -> 1 or 1 -> 14
    if target_color == 14:
        new_val = 1
    elif target_color == 1:
        new_val = 14
    else:
        return new_grid

    for ry, rx in region:
        new_grid[ry, rx] = new_val
        
    # The r0c0... sequence suggests a counter of successful clicks/toggles.
    # We need to find the first empty cell in row 0 and mark it as 1.
    # This is common in ARC tasks to track state.
    for cx in range(grid.shape[1]):
        if grid[0, cx] == 0:
            new_grid[0, cx] = 1
            break
            
    return new_grid

def is_level_complete(grid):
    # Level complete when all interactive blocks are color 1 (blue) or some other condition.
    # In many such puzzles, the goal is to make everything one color.
    # Check if any cells of color 14 remain.
    # Based on observed transitions, we don't see a win state, but this is a standard hypothesis.
    return not np.any(grid == 14)