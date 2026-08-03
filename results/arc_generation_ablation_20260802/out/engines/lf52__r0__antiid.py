import numpy as np

import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    # The game seems to involve clicking on cells (ACTION6)
    # # We observe that clicks at different x coordinates while y=19 result in
    # # in changes to blocks of colors (3s, 14s, etc.)
    # # and some incremental markers in row 0.
    # # The same click coordinate may trigger multiple state transitions.
    # #<|channel>thought: Clicking a cell triggers a toggle or shift of a color block.
    # # Specifically, if we click a pixel (px, py), it affects a region around it.
    # # Let's simulate the observed deltas.
    
    new_grid = grid.copy()
    
    # Increment marker in row 0 based on current value
    # Find first 0 in row 0
    for c in range(64):
        if new_grid[0, c] == 0:
            new_grid[0, c] = 1
            break
    
    # Logic for specific clicked regions based on observations:
    # Click (18, 19) -> Changes area around r17-r22, c16-c22
    # Click (30, 19) -> Changes area around r17-r22, c24-c34
    # Click (42, 19) -> Changes area around r17-r22, c40-c45
    
    # This is a simplified rule to approximate the observed behavior.
    # If px is near 18, 30, or 42, modify blocks of colors.
    if px == 18:
        # Approximate delta from ACTION6 data={'x': 18, 'y': 19}
        # r17c17:3x4, r18c16:3x2, r18c20:3x2, etc.
        for r in range(17, 23):
            for c in range(16, 23):
                new_grid[r, c] = 3 if (r+c)%2==0 else 14
    elif px == 30:
        # Approximate delta from ACTION6 data={'x': 30, 'y': 19}
        # r17c17:0x4, r18c16:0x1, 1x4...
        for r in range(17, 23):
            for c in range(24, 35):
                new_grid[r, c] = 14 if (r+c)%2==0 else 1
    elif px == 42:
        # Approximate same logic for x=42
        for r in range(17, 23):
            for c in range(40, 46):
                new_grid[r, c] = 14 if (r+c)%2==0 else 1
                
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain condition is met.
    # We assume the level is complete when row 0 has some number of markers.
    return np.sum(grid[0, :] == 1) >= 5

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells in the grid are the same color (all 0s).
    """
    grid = np.array(grid)
    return np.all(grid == 0)
