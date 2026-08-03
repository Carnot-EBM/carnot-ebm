import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    y, x = data['y'], data['x']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific regions changes colors.
    # Based on the observed transitions, ACTION6 clicks change blocks of cells.
    # Specifically, it looks like they are target areas that toggle or set values.
    # We need to find the connected components of color 14 and color 5.
    # Let's assume the same time-step logic based on<|channel>thought process.
    # In this case, we can actually just apply the delta if we have a fixed map.
    # But since we only have a few examples, let's induce a general rule.
    # Looking at the deltas, when you click (18, 19), (30, 19), etc., the cells changed are around those coordinates.
    # The cells being changed are often in groups of 2x2, 4x4, or similar.
    # The laughter is that the "r0c0" changes suggest some kind of score or progress tracker.
    # Since I must provide an executable world model, I'll implement a simple version:
    # Clicking a cell toggles its value and surrounding area if it matches certain criteria.
    
    # However, looking closer at the deltas:
    # Click (18, 19) -> r17c17...r22c17 change to 3.
    # Click (30, 19) -> r17c17...r22c17 change to 0/1.
    # This looks like clicking on a block of color 14 (or something else) transforms it into another color.
    # Let's try to find blocks of size 6x6 centered roughly around the click.
    
    # Based on the observed data, let's identify the target regions.
    # Region 1: x=18, y=19. Region 2: x=30, y=19. Region 3: x=42, y=19.
    # These are spaced by 12 units.
    
    # The cells changed in ACTION6 data={'x': 18, 'y': 19} are:
    # r17-22, c17-20 (approx).
    # It seems that when you click near a "structure", it changes state.
    # Specifically, if we click at (x, y), and there is a structure of color 14 nearby, it might be affected.
    
    # Let's implement a rule where clicking toggles colors in a local region.
    # Since I don't have enough data to perfectly map every cell, I will use a simple logic:
    # If action is 6, increment a value at (0,0) and change some pixels based on coordinates.
    
    # For the sake of this specific task, since the deltas are very precise, 
    # let's assume the game is about clearing blocks of color 14.
    
    # Find all connected components of color 14.
    # If the click is inside or near one, change its color.
    
    def get_component(grid, start_node, target_color):
        comp = set()
        stack = [start_node]
        while stack:
            curr = stack.pop()
            if curr not in comp:
                comp.add(curr)
                r, c = curr
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if grid[nr, nc] == target_color:
                            stack.append((nr, nc))
        return comp

    # The observed transitions show that clicking at (18, 19) affects a region around it.
    # Let's check if the clicked cell is part of a structure.
    target_color = 14
    clicked_val = grid[y, x]
    
    # Increment progress tracker at (0,0)
    new_grid[0, 0] += 1
    
    # If we click on color 14 or nearby, change its component to something else.
    # In the first transition, clicking (18, 19) changed cells to color 3.
    # Then clicking (30, 19) changed them back or to something else.
    
    # Let's try this: find all connected components of any color except 0, 5, 9, 10.
    # If the click is within distance 2 of such a component, toggle its color.
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == target_color:
                dist = np.sqrt((r-y)**2 + (c-x)**2)
                if dist < 5:
                    comp = get_component(grid, (r, c), target_color)
                    for cr, cc in comp:
                        new_grid[cr, cc] = 3 if new_grid[cr, cc] == target_color else 14
                    break

    return new_grid

def is_level_complete(grid):
    # The win state isn't provided, but usually it's when certain colors are gone.
    # Based on the game type, maybe when all color 14 is gone?
    return not np.any(grid == 14)