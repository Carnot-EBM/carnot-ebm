import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific blocks changes their color.
    # Based on the observed transitions, ACTION6 clicks change colors of 4x4 or 3x4 areas.
    # Let's identify the "objects" that might be affected by the click.
    # Looking at the same-shape objects in the same coordinates.
    # # obj14, obj17, obj21, obj37, obj48 are all color 14 and have size 3x3 (bbox=(19.5, 18.5) etc.)
    # Actually they are bbox (18, 17, 21, 20) which is 4x4 but pixels=12? Wait.
    # Bbox (18, 17, 21, 20) means rows 18, 19, 20, 21 and cols 17, 18, 19, 20. That's 4x4.
    # If px, py are logical coords, then we can actually see what was changed.
    # In transition 1: data={'x': 18, 'y': 19}, delta = r17c17:3x4, r18c16:3x2...
    # This looks like it's changing a block of cells around the click point.
    # The observed deltas show that clicking on a specific "object" changes its state.
    # Let's look at the<|channel>thought
    # Transition 1: x=18, y=19. Changes color 14 to something else or vice versa.
    # Color 14 blocks are located at (18, 17), (18, 23), (18, 35), (24, 41), (36, 41).
    # Click (18, 19) is inside the first block (18-21, 17-20).
    # Result: r17c17:3x4 etc. It seems to be toggling colors in a pattern.
    # Looking closely at the deltas:
    # Trans 1: x=18, y=19 -> r17c17:3x4, r18c16:3x2, r18c20:3x2 ...
    # Trans 2: x=30, y=19 -> r18c16:0x1, 1x4, 0x1...
    # This looks like it's changing color 14 to color 1 and color 1 to color 14?
    # Or maybe just toggling between color 1 and color 14 for specific "button" objects.
    # Let's try a simpler rule: if you click on an object of color 1 or 14, toggle it.
    # Also, there's a counter at the top (r0c0, r0c1...).
    # Every successful click increments the value at r0c(count).
    # The goal is likely to turn all these buttons to a certain color.
    
    # Find which button was clicked. Buttons are 4x4 blocks.
    # We can identify them by their center or bbox.
    # Button centers from INITIAL OBJECTS:
    # obj14: (19.5, 18.5), obj17: (19.5, 24.5), obj21: (19.5, 36.5), obj37: (25.5, 42.5), obj48: (37.5, 42.5)
    # These are the ones with color 14 initially. Others have color 1.
    # Let's define the button bboxes (y0, x0, y1, x1):
    buttons = [
        (18, 17, 21, 20), # obj14
        (18, 23, 21, 26), # obj17
        (18, 35, 21, 38), # obj21
        (24, 41, 27, 44), # obj37
        (36, 41, 39, 44), # obj48
    ]
    # Wait, there are more buttons of color 1.
    # Looking at INITIAL OBJECTS: obj5, 6, 7, 8, 9, 10, 11... these are all 4x4 blocks of color 1.
    # Let's just find any 4x4 block that is either color 1 or 14 and contains (py, px).
    
    clicked_button = None
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1 or grid[y, x] == 14:
                # Check if this is the top-left of a 4x4 block
                if y+3 < grid.shape[0] and x+3 < grid.shape[1]:
                    if np.all(grid[y:y+4, x:x+4] == grid[y, x]):
                        if y <= py < y+4 and x <= px < x+4:
                            clicked_button = (y, x)
                            break
        if clicked_button: break

    if clicked_button:
        y, x = clicked_button
        current_color = grid[y, x]
        new_color = 14 if current_color == 1 else 1
        new_grid[y:y+4, x:x+4] = new_color
        
        # Increment counter at r0
        count = 0
        while count < new_grid.shape[1] and new_grid[0, count] != 0:
            count += 1
        if count < new_grid.shape[1]:
            new_grid[0, count] = 1
            
    return new_grid

def is_level_complete(grid):
    # The level is complete when all buttons are color 14? Or some other condition.
    # In the observed transitions, we don't see a win state.
    # But usually it's about turning all targets to a specific color.
    # Let's assume it's when all "buttons" (blocks of size 4x4 that were initially 1 or 14) are now 14.
    # Since we don't have the final state, this is a guess.
    # Another possibility: the counter reaches a certain value.
    # Let's check for any block of color 1 that could be a button.
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if grid[y, x] == 1:
                if y+3 < grid.shape[0] and x+3 < grid.shape[1]:
                    if np.all(grid[y:y+4, x:x+4] == 1):
                        return False
    return True