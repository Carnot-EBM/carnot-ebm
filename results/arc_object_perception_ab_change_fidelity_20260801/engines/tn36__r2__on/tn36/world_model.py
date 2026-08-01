import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    # The game seems to be a "click-to-fill" or "toggle" mechanism where clicking on specific areas
    # changes colors of cells in both the clicked area and a distant target area.
    # Based on the observed transitions:
    # ACTION6 x=24, y=41 -> r1c61:3x1, r42c25:5x3
    # ACTION6 x=24, y=44 -> r1c60:3x1, r44c26:5x1, r45c26:5x1, r46c26:5x1
    # ACTION6 x=34, y=41 -> r1c59:3x1, r42c35:5x3
    # ACTION6 x=34, y=44 -> r1c58:3x1, r44c36:5x1, r45c36:5x1, r46c36:5x1
    # ACTION6 x=39, y=41 -> r1c57:3x1, r42c40:5x3
    
    # The clicks are happening at (y, x) = (41, 24), (44, 24), (41, 34), (44, 34), (41, 39).
    # These coordinates correspond to the logical grid indices.
    # Let's analyze the relationship between click coords and changed cells.
    #
    # Click (41, 24): Changed cells are r42c25:5x3 (row 42, col 25-27) and r1c61:3x1 (row 1, col 61).
    # Click (44, 24): Changed cells are r44c26:5x1, r45c26:5x1, r46c26:5x1 (col 26) and r1c60:3x1 (row 1, col 60).
    # Click (41, 34): Changed cells are r42c35:5x3 (row 42, col 35-37) and r1c59:3x1 (row 1, col 59).
    # Click (41, 39): Changed cells are r42c40:5x3 (row 42, {col 40-42}) and r1c57:3x1 (// a bit off)
    #
    # It looks like the click target is to fill in gaps of color 0.
    # The "target" area is row 1, which acts as a progress bar or similar.
    #
    # Let's generalize: clicking on a cell of color 0 that is part of a specific structure
    # changes it to color 5 (the background/wall color), and also fills a cell in row 1.
    #
    # In all cases, the clicked cell (py, px) is color 0.
    #
    # For each action, we look for the connected component of color 0 starting at (py, px).
    #
    # However, the simpler rule is: if you click a cell of color 0, it becomes color 5.
    # If the same object (connected component) of color 0 is fully filled, something happens.
    #
    # Looking at the deltas:
    # Click (41, 24): Changed cells are r42c25:5x3 (row 42, col 25-27) and r1c61:3x1 (row 1, col 61).
    # Wait, the delta says r42c25:5x3. The click was at x=24, y=41.
    # The changed cells are NOT exactly where the click happened.
    # Let's re-examine:
    # ACTION6 data={'x': 24, 'y': 41} -> r42c25:5x3, r1c61:3x1
    # This means clicking at (41, 24) affects row 42, cols 25-27.
    #
    # Actually, looking at the INITIAL GRID, r42 has gaps of color 0.
    # r42: 5x13, 0x7, 5x3, 0x2, 1x3, 0x2, 5x3, 0x2, 1x3, 0x2, 1x3, 0x8, 5x13
    # Row 42 contains several segments of color 0.
    # Segment 1: cols 13-19 (len 7)
    # Segment 2: cols 23-24 (len 2)
    # Segment 3: cols 27-28 (len 2)
    # Segment 4: cols 31-32 (len 2)
    # Segment 5: cols 35-42 (len 8)
    #
    # Click (41, 24): x=24, y=41. This is in a region of color 0.
    # The delta says r42c25:5x3. Wait, row 42 col 25 is not color 0?
    # Let's check INITIAL GRID again.
    # r42: 5x13(0-12), 0x7(13-19), 5x3(20-22), 0x2(23-24), 1x3(25-27), 0x2(28-29), 5x3(30-32), 0x2(33-34), 1x3(35-37), 0x2(38-39), 1x3(40-42), 0x8(43-50), 5x13(51-63).
    # So at r42, c25-27 is color 1.
    # Clicking (41, 24) changes r42c25-27 from color 1 to color 5.
    #
    # Similarly, Click (44, 24): x=24, y=44. Changed cells are r44c26:5x1, r45c26:5x1, r46c26:5x1.
    # In INITIAL GRID:
    # r44: 5x13, 0x8, 5x1, 0x4, 1x1, 0x4, 5x1, 0x4, 1x1, 0x4, 1x1, 0x9, 5x13
    # Col 26 in r44, r45, r46 is color 1.
    #
    # Rule: clicking on a cell of color 0 that "overlaps" or is "near" a block of color 1,
    # converts the corresponding block of color 1 into color 5.
    # And it fills one pixel in row 1 starting from the right (col 61, then 60, etc.).
    #
    # Let's refine: if you click at (py, px), find all connected components of color 1.
    # If any component of color 1 is "associated" with the clicked area, change it to 5.
    #
    # Looking at the coordinates again:
    # Click (41, 24): x=24, y=41. Color 1 block is at r42c25-27.
    # Click (44, 24): x=24, y=44. Color 1 block is at r44-46, c26.
    # Click (41, 34): x=34, y=41. Color 1 block is at r42c35-37.
    # Click (44, 34): x=34, y=44. Color 1 block is at r44-46, c36.
    # Click (41, 39): x=39, y=41. Color 1 block is at r42c40-42.
    #
    # In each case, the color 1 block's centroid or bounding box is very close to the click.
    # Let's use a simple distance rule: find all connected components of color 1.
    # If any component's bbox contains (py, px) or is within a small distance, change it to 5.
    # And for every such component changed, fill one cell in row 1 from col 61 downwards.
    #
    # Row 1 progress bar:
    # Initial: r1: 5x1, 9x61, 5x2. Col 1-61 are color 9.
    # After first click: r1c61 becomes 3.
    # After second click: r1c60 becomes 3.
    # This means we are filling row 1 from right to left with color 3.
    
    new_grid = grid.copy()
    if action == 6:
        px, py = data['x'], data['y']
        
        # Find all connected components of color 1
        visited = np.zeros_like(grid, dtype=bool)
        components = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 1 and not visited[r, c]:
                    comp = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        curr_r, curr_c = q.pop()
                        comp.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and \
                               grid[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    components.append(comp)
        
        # Check which component is closest to the click
        best_comp = None
        min_dist = float('inf')
        for comp in components:
            # Use distance from click to any cell in the component
            for cr, cc in comp:
                dist = abs(cr - py) + abs(cc - px)
                if dist < min_dist:
                    min_dist = dist
                    best_comp = comp
        
        if best_comp and min_dist < 5: # Threshold for "close enough"
            # Change component to color 5
            for cr, cc in best_comp:
                new_grid[cr, cc] = 5
            
            # Update progress bar in row 1 (right to left)
            # Find the rightmost cell in row 1 that is still color 9
            for c in range(61, 0, -1):
                if new_grid[1, c] == 9:
                    new_grid[1, c] = 3
                    break
    return new_grid

def is_level_complete(grid):
    # The level is complete when all blocks of color 1 are gone.
    return not np.any(grid == 1)