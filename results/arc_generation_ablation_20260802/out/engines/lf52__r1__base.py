import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The observed transitions show a pattern of changing colors based on clicks.
    # It seems like clicking on specific regions changes the color of blocks in that region.
    # Specifically, it looks like some cells are change to color 3 or 14 depending on the same click.
    # Let's analyze the coordinates and clicked pixels.
    # r17-r22 area corresponds to y=19 approx.
    # r30-r33 area corresponds to y=32 approx.
    # Looking at the<|channel>thoughtt process, we could actually be simulating a "toggle" or "fill" mechanism.
    # However, since I must induce simple general rules from very few examples,
    # let's look at the delta.
    # Action 6 (click) at (18, 19) changed cells in rows 17-22 and row 0.
    # Row 0 is acting as a counter/score.
    # In each ACTION6, one cell in row 0 changes from 0 to 1.
    #
    # Based on the deltas:
    # Click (18, 19): r17c17:3x4, r18c16:3x2... etc.
    # This suggests clicking a point triggers a change in a local neighborhood or a specific object.
    # The objects are likely defined by color boundaries.
    #
    # Let's implement a simple rule: find the connected component of the clicked cell's color,
    # and change its color to something else. But wait, the colors change to 3 and then to 14.
    # Color 5 seems to be a boundary.
    #
    # Looking closer at the INITIAL GRID:
    # Rows 12-15 have blocks of color 1.
    # Rows 18-21 have blocks of color 1 and 14.
    #
    # Actually, looking at the delta for Action 6 data={'x': 18, 'y': 19}:
    # It changed cells to color 3. Then subsequent clicks changed them to 14.
    #
    # Given the constraints and the observed transitions, it looks like this is a puzzle where you click "objects" (connected components)
    # and they cycle through colors or toggle state.
    # Row 0 tracks how many times you've successfully performed an action.
    
    # Since I cannot perfectly deduce the complex geometry from these few deltas,
    # but I see that row 0 increments, let's simulate the most basic part:
    # The click changes some pixels and increments row 0.
    
    # To actually pass the ARC test, we need the exact logic.
    # Let's try: find connected component of clicked cell, change its color.
    # But what color? In transition 1, it became 3. In transition 2, it became 14.
    # Wait, in transition 2, the same area was clicked again and it became 14.
    # This looks like: Color X -> Color Y -> Color Z.
    
    # Let's refine:
    # 1. Find connected component of grid[py, px].
    # 2. Change all cells in that component to a new color.
    # 3. Increment row 0 counter.
    
    # What is the "new color"?
    # Transition 1: Click (18, 19) -> Component becomes 3.
    # Transition 2: Click (30, 19) -> Component becomes 14.
    # Looking at INITIAL GRID, r18-r21 already have 14s.
    # Maybe it's not components but specific regions.
    
    # Actually, looking at the deltas again:
    # ACTION6 data={'x': 18, 'y': 19} changed pixels to 3.
    # ACTION6 data={'x': 30, 'y': 19} changed those pixels from 3 back to something else or to 14?
    # No, the delta says r17c17:0x4... which means no change for some, and others became 14.
    
    # This is too ambiguous. Let's implement a simple "connected component toggle" and see.
    # But wait, the most important thing is the state transition.
    # The cells in row 0 are changing from 0 to 1.
    
    # Let's try this logic:
    # If action == 6:
    #   Find connected component of grid[py, px] using color = grid[py, px].
    #   If current_color == 1: new_color = 3
    #   elif current_color == 3: new_color = 14
    #   else: new_color = 1 # fallback
    #   Change component to new_color.
    #   Find first 0 in row 0 and set it to 1.
    
    current_color = grid[py, px]
    if current_color == 1:
        new_color = 3
    elif current_color == 3:
        new_color = 14
    elif current_color == 14:
        new_color = 1
    else:
        new_color = current_color # No change for boundaries (5, 9, 10)

    if new_color != current_color:
        # BFS for connected component
        component = []
        queue = [(px, py)]
        visited = set([(px, py)])
        while queue:
            cx, cy = queue.pop(0)
            component.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < new_grid.shape[1] and 0 <= ny < new_grid.shape[0]:
                    if (nx, ny) not in visited and new_grid[ny, nx] == current_color:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        for cx, cy in component:
            new_grid[cy, cx] = new_color
            
        # Increment row 0 counter
        for c in range(new_grid.shape[1]):
            if new_grid[0, c] == 0:
                new_grid[0, c] = 1
                break
                
    return new_grid

def is_level_complete(grid):
    # Level complete if some condition is met. 
    # Usually it's when all target objects are a certain color or row 0 is filled.
    # Given the data, we don't have a win state grid.
    # Let's assume it's complete if row 0 has a certain number of 1s.
    # Or more simply, just return False unless we can identify a clear win state.
    return np.sum(grid[0]) >= 5 # Based on the 5 transitions shown