import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    y, x = data['y'], data['x']
    new_grid = grid.copy()
    
    # The observed transitions show that clicking on a cell changes colors of blocks of cells.
    # Based on the same coordinates provided in the ACTION6 calls and repeated actions,
    # it seems to be a "toggle" or "fill" operation on specific patterns.
    # Let's identify the regions affected by the effect.
    # Specifically, looking at the transition deltas, they change values to 3 or 14.
    # 14 is often associated with 'walls' or 'obstacles'.
    # 14 was already present in the initial grid.
    # 14 is likely the target color for some operations.
    
    # In this game, clicking a coordinate (x, y) might be acting as a trigger for a 
    # block-based logic.
    # Simple rule induction from own observations:
    # Clicking (x, y) affects a region around (x, y) if certain conditions are 
    # are met.
    # laf52 is own name for the<|channel>thought//no_think
    # This looks like a puzzle where you click elements to change their state.
    # Looking at the delta, r0c0 becomes 1, then r0c1 becomes 1... 
    # It suggests a counter or progress bar at the top row.
    # The cells being changed are usually blocks of size 4x4 or similar.
    # The same ACTION6 data={'x': 30, 'y': 19} is repeated twice.
    # First time it changes colors to 1 and 14.
    # The second time it changes colors to 3.
    # This implies that clicking on a specific "button" (a block of cells) 
    # toggles its state between two or more states.
    
    # Let's try to implement a toggle based on the observed deltas.
    # Since we don't have enough transitions to define general rules for all buttons,
    # we will focus on the most common pattern:
    # Clicking a cell in a button region toggles the entire button's color.
    # We identify buttons as connected components of non-zero/non-ten values.
    # Use a simple approach: find the clicked cell's current value, and if it's not 10 (background),
    # change the connected component containing (x, y) to a new color.
    
    # Note: r0c0...r0c4 becomes 1. This looks like a progress indicator.
    # We count how many distinct buttons have been activated.
    
    # Find the connected component of the same color as grid[y, x]
    color = grid[y, x]
    if color == 10:
        return new_grid
    
    component = []
    stack = [(y, x)]
    visited = set()
    while stack:
        curr_y, curr_x = stack.pop()
        if (curr_y, curr_x) in visited:
            continue
        visited.add((curr_y, curr_x))
        if 0 <= curr_y < grid.shape[0] and 0 <= curr_x < grid.shape[1]:
            if grid[curr_y, curr_x] == color:
                component.append((curr_y, curr_x))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((curr_y + dy, curr_x + dx))
    
    # Toggle logic based on observed values: 3 -> 14 -> 3 or similar.
    # We'll use a simple cycle for colors: if it was 1, change to 14; if 14, change to 3; etc.
    # But looking at the data: 18, 19 is clicked, then 30, 19... 
    # The cells changed are r17c17-20, r18c16-19, etc. This is a block around (19, 18).
    # Let's assume clicking toggles the component's color.
    # If current color is 1, maybe it becomes 14? Or if 14, becomes 3?
    # Looking at ACTION6 x=18, y=19: changes some things to 3.
    # ACTION6 x=30, y=19: first time changes some things to 1 and 14.
    # Second time x=30, y=19: changes some things to 3.
    
    # To simulate progress bar r0:
    # Find how many components have been "activated" (changed from initial state).
    # Actually, just increment r0[count] = 1 where count is number of clicks.
    click_count = np.sum(grid[0] == 1)
    new_grid[0, click_count] = 1
    
    # Change the clicked component's color
    next_color = 3 if color != 3 else 14 if color != 14 else 1
    for cy, cx in component:
        new_grid[cy, cx] = next_color
    
    return new_grid

def is_level_complete(grid):
    # Level complete when a certain condition is met.
    # Usually, this means all buttons are a specific color or progress bar is full.
    # Given we don't have win states, let's assume it's when row 0 has enough 1s.
    return np.sum(grid[0] == 1) >= 10