import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game involves clicking on a "block" or "object".
    # Based on observed transitions, ACTION6 changes colors in specific patterns.
    # It seems like it's clicking on an object and that object transforms.
    # Specifically, color 1 blocks transform into color 14 blocks if they are clicked?
    # No, looking at the delta, when x=18, y=19 is clicked, cells around (19, 18) change.
    # Let's identify the target object.
    # Find all connected components of color 1 or 14.
    #
    # From observations:
    # Click (18, 19) -> Target object at (19, 18) transformed to color 3.
    # Click (30, 19) -> Target object at (19, 30) transformed to color 14.
    # Click (30, 19) again -> Target object at (19, 29) transformed to color 3.
    # Wait, the coordinates are (x, y). So data['x'] is col, data['y'] is row.
    # Click (18, 19) -> Col 18, Row 19. Object there is color 14. It becomes color 3.
    # Click (30, 19) -> Col 30, Row 19. Object there is color 1. It becomes color 14.
    # Click (30, 19) -> Col 30, Row 19. Now it's color 14. It becomes color 3.
    # Let's check if this is a cycle: Color 1 -> Color 14 -> Color 3 -> Color 1?
    # No, look at the deltas carefully.
    # r17c17:3x4 means rows 17-20, cols 17-20 change to 3.
    # The target object was color 14 (bbox=(18, 17, 21, 26)).
    # Actually, let's just find the connected component containing (py, px).
    #
    # Target object identification:
    # Find all cells of same color as grid[py, px] that are connected.
    #
    # Transition rules for colors:
    # If clicked cell is color 1, it becomes 14.
    # If clicked cell is color 14, it becomes 3.
    # If clicked cell is color 3, it becomes 1.
    # But wait, looking at the delta for x=18, y=19:
    # Initial grid has color 14 at (19, 18) - no, (19, 17..20).
    # Let's re-verify coordinates.
    # ACTION6 data={'x': 18, 'y': 19} -> py=19, px=18. grid[19, 18] = 14.
    # Delta: r17c17:3x4... this means a block around there became color 3.
    # So Color 14 -> Color 3.
    # ACTION6 data={'x': 30, 'y': 19} -> py=19, px=30. grid[19, 30] = 1.
    # Delta: r18c30:14x2... This means the object became color 14.
    # So Color 1 -> Color 14.
    # ACTION6 data={'x': 30, 'y': 19} again -> py=19, px=30. Now it is color 14.
    # Delta: r17c29:3x4... It becomes color 3.
    # So Color 14 -> Color 3.
    # ACTION6 data={'x': 42, 'y': 19} -> py=19, px=42. grid[19, 42] = 1.
    # Delta: r18c42:14x2... It becomes color 14.
    # So Color 1 -> Color 14.
    # ACTION6 data={'x': 42, 'y': 19} again -> py=19, px=42. Now it is color 14.
    # Delta: r17c41:3x4... It becomes color 3.
    #
    # Summary of cycle: Color 1 -> Color 14 -> Color 3.
    # Let's check if there's a "win" condition. The deltas also show r0c0:1x1 etc.
    # This looks like a counter in the top left corner (r0, c0), (r0, c1)...
    # Every click increments the counter at (0, 0).
    #
    # Object transformation logic:
    # 1. Find connected component containing (py, px) of same color.
    # 2. If color is 1, change to 14.
    # 3. If color is 14, change to 3.
    # 4. If color is 3, maybe back to 1? Or stays 3?
    #
    # Connected Component Logic:
    # Use BFS to find all cells of the same color as grid[py, px].
    
    if py >= new_grid.shape[0] or px >= new_grid.shape[1]:
        return new_grid
    
    target_color = grid[py, px]
    if target_color not in [1, 14, 3]:
        # Based on observations, only these colors are transformed.
        # But let's be general.
        pass

    # BFS for connected component
    component = []
    queue = [(py, px)]
    visited = set([(py, px)])
    while queue:
        curr_y, curr_x = queue.pop(0)
        component.append((curr_y, curr_x))
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = curr_x + dx, curr_y + dy # wait, x and y swapped
            # Correcting coordinates
            pass
    
    # Let's rewrite the BFS properly.
    component = []
    queue = [(py, px)]
    visited = set([(py, px)])
    while queue:
        cy, cx = queue.pop(0)
        component.append((cy, cx))
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < new_grid.shape[0] and 0 <= nx < new_grid.shape[1]:
                if (ny, nx) not in visited and grid[ny, nx] == target_color:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
    
    # Transform color
    new_color = target_color
    if target_color == 1:
        new_color = 14
    elif target_color == 14:
        new_color = 3
    elif target_color == 3:
        new_color = 1 # Cycle back to 1? Or maybe it's just 1->14->3.
    
    for cy, cx in component:
        new_grid[cy, cx] = new_color
    
    # Increment counter at r0, cX
    # Find the first empty cell in row 0 from left to right.
    # Row 0 is initially all 0s except for clicks.
    # The deltas show r0c0:1x1, then r0c1:1x1...
    # This means every click puts a '1' at the next available column in row 0.
    # For simplicity, let's find the first index where grid[0, col] == 0 and set it to 1.
    # But wait, initial grid has r0:0x64. So (0,0) is color 0.
    # After first click, r0c0 becomes 1.
    # Let's check if this happens even if we click on something that doesn't change.
    # Yes, ACTION6 always increments the counter.
    
    col = 0
    while col < new_grid.shape[1] and new_grid[0, col] != 0:
        col += 1
    if col < new_grid.shape[1]:
        new_grid[0, col] = 1
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves some pattern or all objects of certain colors.
    # In these games, often "all blocks are color X".
    # Based on observations, there's no win state provided.
    # Let's assume a common win condition for this type of puzzle.
    # Or maybe when row 0 fills up? No.
    # Let's look at the object structure again.
    # Maybe when all target-type blocks are color 3?
    # return False # Default to False as no WIN STATE grid was given.
    
    # Since no win state was shown, let's try to find a logical one.
    # If any block is still color 1 or 14, it's not complete.
    # But we don't know which blocks are targets.
    # Let's just check if all cells that were originally 1 or 14 are now 3.
    # This is hard without knowing original grid.
    # For now, return False.
    return False