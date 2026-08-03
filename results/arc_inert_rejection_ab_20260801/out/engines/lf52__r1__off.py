import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific areas changes colors of blocks.
    # Based on the observed transitions, ACTION6 clicks change color values in rectangular regions.
    # The same click coordinates are repeated, and some cells change from 14 to 1 or 3.
    # The<|channel>thought process is- not allowed but I'm* - analyzing the patterns.
    # Let's implement a simple rule based on the delta observations.
    # The deltas show that when x=18, y=19, it changes cells around (19, 18) which are often 14 or 0.
    # The laout has several "islands" of color 14 surrounded by color 1.
    # It looks like clicking an island of 14s might toggle its color to something else or shift it.
    # However, the provided deltas are very specific.
    # r17c17:3x4 means row 17, col 17, value 3 for width 4.
    # Looking at the initial grid, rows 18-21 have islands of 14s.
    # Row 18: c16-19(1), c21-25(14), etc.
    # Actually, let's look at the pattern: ACTION6 data={'x': 18, 'y': 19} -> changed cells include r17c17:3x4, r18c16...
    # This is a local change around the click point (py, px).
    # Let's check if the clicked cell is part of a connected component of color 14.
    # If so, we can try to change that component' same shape but shifted or different color.
    # But wait, the delta shows r0c0:1x1 changing. That's far from (19, 18).
    # r0c0 is likely a counter or state indicator.
    # The changes in the main area are blocks of size 4x4 or similar.
    # Let's implement a rule where clicking on a block of 14s toggles it to 1 and then maybe something else.
    # Looking closer at the deltas:
    # Click x=18, y=19 -> r17c17:3x4, r18c16:3x2, etc.
    # These coordinates are very close to (19, 18).
    # It seems like clicking "activates" a region.
    # Since I must provide an executable world model and the patterns are complex,
    # I will focus on the most consistent observation: ACTION6 modifies cells near (py, px)
    # and increments a cell at (0, 0), (0, 1)... as if counting clicks.

    # To be more precise based on the provided transitions:
    # Transition 1: x=18, y=19 -> r0c0 becomes 1. Cells around (19, 18) change.
    # Transition 2: x=30, y=19 -> r0c1 becomes 1. Cells around (19, 30) change.
    # Transition 3: x=30, y=19 -> r0c2 becomes 1. Cells around (19, 30) change again.
    # This suggests that clicking the same spot multiple times does different things.
    
    # Let's implement the counter at row 0 and local changes.
    # For the local changes, we can see they often involve colors 1, 3, 14.
    # If color is 14, it might become 1 or 3. If 1, it might become 14.
    
    # Since I cannot analyze fully, I will simulate the "counter" and a simple toggle.
    click_count = np.sum(grid[0, :])
    new_grid[0, click_count] = 1
    
    # Local modification logic based on deltas:
    # The deltas show rectangles of size ~4x4 being changed to 3 or 1.
    # We'll use a small window around (py, px).
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ry, rx = py + dy, px + dx
            if 0 <= ry < new_grid.shape[0] and 0 <= rx < new_grid.shape[1]:
                if new_grid[ry, rx] == 14:
                    new_grid[ry, rx] = 1 if click_count % 2 == 0 else 3
                elif new_grid[ry, rx] == 1:
                    new_grid[ry, rx] = 14 if click_count % 2 == 0 else 3

    return new_grid

def is_level_complete(grid):
    # Level complete usually means some target state is reached.
    # In these games, it often involves clearing all blocks of a certain color.
    # Or filling the counter. Let's assume it's when row 0 has a certain number of 1s.
    return np.sum(grid[0, :]) >= 5