import numpy as np

def engine(grid, action, data):
    """
    Predict the next grid state based on the action and data.
    
    Rules:
    - ACTION6 (click) fills a 6x6 region centered at the clicked pixel with color 8 (teal).
    - The fill region is defined as: rows [y-3, y+3] and columns [x-3, x+3], clipped to grid bounds.
    - Other actions (1-5) have no effect on the grid.
    """
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        # Define the 6x6 fill region
        rows = np.arange(max(0, py - 3), min(64, py + 4))
        cols = np.arange(max(0, px - 3), min(64, px + 4))
        
        # Create a mask for the fill region
        mask = np.zeros((64, 64), dtype=bool)
        for r in rows:
            for c in cols:
                mask[r, c] = True
        
        # Apply the fill (color 8) to the grid
        grid = grid.copy()
        grid[mask] = 8
    
    return grid

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    
    Win state criteria:
    - The grid must be fully filled with color 4 (yellow) in the top-left 64x64 region.
    - Specifically, the top-left 64x64 region must be entirely color 4.
    """
    # Check if the top-left 64x64 region is entirely color 4
    return np.all(grid[:64, :64] == 4)