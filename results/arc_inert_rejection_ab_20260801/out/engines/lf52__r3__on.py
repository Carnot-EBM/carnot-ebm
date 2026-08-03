import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    # ACTION6 is a click at (px, py). Logical coords are px, py.
    # Based on the observed transitions, clicking on specific regions of color 14 or 1 changes 
    # the colors in those regions to something else and potentially affects other areas.
    # The pattern suggests that if you click on a cell (x, y), it might be "activating" 
    # a region.
    # Looking at thes deltas:
    # Clicking x=18, y=19 -> changed cells include r17c17:3x4, r18c16:3x2...
    # r17c17:3x4 means row 17, col 17-20.
    # r18c16:3x2 means row 18, col 16-17.
    # This looks like a toggle or replacement.
    # Let's analyze the pixels clicked: (18, 19) then (30, 19) then (30, 19) again? No, (30, 19) twice.
    # Then (42, 19) then (42, 19).
    # The clicks are moving across the grid horizontally.
    # The same coordinates are often repeated.
    # The laest delta shows changes in rows 17-22 and also rows 30-33.
    
    # However, the provided observed transitions are very limited and only show "level 0->0", which implies they might not be
    # just changing state but perhaps interacting with an interaction map.
    # Given the constraints and thes specific deltas, we can actually see that clicking on color 14 (which is red/yellowish) 
    # replaces it with color 3 (greenish) or vice versa.
    # Specifically, if you click on a cell of color 14, the surrounding block of color 14 cells becomes color 3.
    # If you click on color 3, it becomes 14.
    # Let's try to implement a toggle for connected components of a certain color.
    
    x, y = data['x'], data['y']
    target_color = grid[y, x]
    if target_color == 0:
        return grid

    # Find connected component of the same color
    component = []
    stack = [(y, x)]
    visited = np.zeros_like(grid, dtype=bool)
    while stack:
        curr_y, curr_x = stack.pop()
        if 0 <= curr_y < grid.shape[0] and 0 <= curr_x < grid.shape[1]:
            if not visited[curr_y, curr_x]:
                visited[curr_y, curr_x] = True
                if grid[curr_y, curr_x] == target_color:
                    component.append((curr_y, curr_x))
                    stack.append((curr_y + 1, curr_x))
                    stack.append((curr_y - 1, curr_x))
                    stack.append((curr_x + 1, curr_y)) # wait, this is (curr_x+1, curr_y) should be (curr_y, curr_x+1)
                    stack.append((curr_y, curr_x - 1))
    
    # This is a bit wrong. Let's fix the connected component logic.
    
    return grid

def engine_fixed(grid, action, data):
    if action != 6:
        return grid
    
    x, y = data['x'], data['y']
    target_color = grid[y, x]
    if target_color == 0:
        return grid

    # Connected component search
    component = []
    stack = [(y, x)]
    visited = np.zeros_like(grid, dtype=bool),
    # Wait, I can't use commas at the end of line.
    
    return grid

# Since the observed transitions are very sparse and only "level 0->0", 
# and it's actually quite hard to induce the exact rules from these deltas without more info,
# let's implement a simple toggle for the connected components of color 14 or 1.
# Based on the deltas, clicking on a cell changes its own region and potentially others.
# The laest delta shows r30c42:2x2 which means row 30, col 42-43.
# In the initial grid, r30 is 10x10, 9x23, 5x1, 0x1, 1x4...
# This corresponds to (30, 33) being color 5, then (30, 34) being 0, then (30, 35-38) being 1.
# So clicking at (42, 19) affected cells in row 30.
# This looks like a trigger system where one block activates another.
# Let's try to implement a basic version of this.

def engine(grid, action, data):
    if action != 6:
        return grid
    
    x, y = data['y'], data['x'] # Wait, the data says 'x':px, 'y':py. Logical coords are px, py.
    # Looking at the observed transitions: ACTION6 data={'x': 18, 'y': 19} -> changed cells include r17c17:3x4...
    # x=18, y=19. Row 17, Col 17. This matches.
    # x, y = data['x'], data['y']
    # But wait, if I click on (18, 19), it changes things around it.
    # The deltas show that color 14 is replaced by color 3 or vice versa.
    # And some other colors might be change.
    # In the initial grid, r18 has 14x2 and 14x4.
    # Clicking at (18, 19) affects those regions.
    # laest delta shows r0c0:1x1, r0c1:1x1 etc. which means a counter is being incremented in row 0.
    # Let's implement a simple toggle for connected components of color 14 and 1.
    
    px, py = data['x'], data['y']
    target_color = grid[py, px]
    if target_color == 0: return grid
    
    new_grid = grid.copy()
    stack = [(py, px)]
    visited = np.zeros_like(grid, dtype=bool)
    component = []
    while stack:
        curr_y, curr_x = stack.pop()
        # Correcting coordinates
        if 0 <= curr_y < grid.shape[0] and 0 <= curr_x < grid.shape[1]:
            if not visited[curr_y, curr_x]:
                visited[curr_y, curr_x] = True
                if grid[curr_y, curr_x] == target_color:
                    component.append((curr_y, curr_x))
                    stack.append((curr_y + 1, curr_x))
                    stack.append((curr_y - 1, curr_x))
                    stack.append((curr_y, curr_x + 1))
                    stack.append((curr_y, curr_x - 1))
    
    # Toggle color based on observed deltas (14 -> 3, 3 -> 14, 1 -> 2)
    for cy, cx in component:
        if target_color == 14:
            new_grid[cy, cx] = 3
        elif target_color == 3:
            new_grid[cy, cx] = 14
        elif target_color == 1:
            new_grid[cy, cx] = 2
        else:
            new_grid[cy, cx] = 0 # fallback
            
    return new_grid

def is_level_complete(grid):
    # Win state usually means all targets are reached or a certain pattern is formed.
    # Since no win state was provided, we'll assume it's complete when row 0 has some values.
    return np.any(grid[0, :] != 0)