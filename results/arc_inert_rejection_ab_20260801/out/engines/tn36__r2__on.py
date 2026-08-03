import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Extract click coordinates
    px, py = data['x'], data['y']
    
    # The game seems to be a "click and clear" or "toggle" mechanism.
    # Based on the observed transitions, clicking at specific locations (e.g., y=41, 44)
    # changes colors of blocks in the same column or related areas.
    # In thiss case, it's specifically changing color 0/1/11 to 5 (the background color).
    # Looking at the<|channel>thought process, the clicks are at x=24, 34, 39 and y=41, 44.
    # These correspond to columns 24, 34, 39.
    # The delta shows that cells at r1c61, r42c25, etc., are changed to 5.
    # The target area for clearing is likely the center region where non-5 values exist.
    # Let's implement a simple rule: if you click a cell, check if it belongs to a connected component of non-5 values.
    # If it clicked cell is part of a group of non-5 values, change those to 5.
    # 
    # However, looking closer at the deltas:
    # ACTION6 data={'x': 24, 'y': 41} -> r1c61:3x1, r42c25:5x3
    # This means clicking at (24, 41) clears something at (61, 1) and (25, 42).
    # Wait, the coordinates in the delta are r<row>c<col>.
    # So r42c25 is row 42, col 25. x=24, y=41? No, logical coords are usually (y, x).
    # In ARC, often x is column, y is row.
    # Click (24, 41): Row 41, Col 24. Delta: r42c25... wait.
    # Let's re-examine: Action6 {'x': 24, 'y': 41}. Delta: r42c25:5x3.
    # Row 42, Col 25 is very close to (41, 24).
    # And r1c61:3x1. Row 1, Col 61 is far away.
    # Looking at the INITIAL grid: r1 has a long run of color 9.
    # The clicks seem to be removing pieces of this "puzzle".
    # Since we don't have enough data to deduce the exact mapping, but we see that clicking
    # non-5 cells generally turns them into 5, and some distant cell also turns into 5.
    # Specifically, it looks like clicking a block in the bottom area clears a corresponding block in the top area.
    
    new_grid = grid.copy()
    
    # Simple rule induced from deltas: if you click a cell (px, py),
    # check for any connected component of non-5 values containing (py, px).
    # Find all such components and clear them? No, just the one clicked.
    # Let's try clearing the connected component of the clicked cell.
    
    if 0 <= py < new_grid.shape[0] and 0 <= px < new_grid.shape[1]:
        val = new_grid[py, px]
        if val != 5:
            # Flood fill to find connected component of same value or any non-5
            stack = [(py, px)]
            visited = set([(py, px)])
            while stack:
                r, c = stack.pop()
                new_grid[r, c] = 5
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
                        if (nr, nc) not in visited and new_grid[nr, nc] != 5:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
    
    return new_grid

def is_level_complete(grid):
    # Level complete when all cells are color 5?
    # Or when the "puzzle" area is cleared.
    # Based on typical ARC games, it's usually when a certain pattern is achieved or removed.
    # Let's check if only color 5 remains.
    return np.all(grid == 5)