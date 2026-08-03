import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where clicking on specific coordinates (ACTION6)
    toggles the state of a 3x3 block centered at that location and potentially 
    affects other blocks or markers.
    Based on the observed transitions:
    - Clicking at (x, y) changes a 3x3 area starting at row y and col x to color 15.
    - It also seems to revert a previous 3x3 block back to its background color (color 5).
    - There's a marker updating in the bottom right corner (row 63).
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px = data['x']
    py = data['y']

    # The observations show ACTION6 creates a 3x3 block of color 15.
    # Let's identify the current active 3x3 block of color 15 if it exists.
    # In the provided deltas, when a new block is created, an old one is removed.
    
    # Find existing 3x3 blocks of color 15
    found_block = None
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                found_block = (r, c)
                break
        if found_block: break

    # Remove previous block if found (set back to background color 5 or original)
    # Looking at the delta: r46c15:5x3 etc suggests reverting to color 5.
    if found_block:
        br, bc = found_block
        new_grid[br:br+3, bc:bc+3] = 5

    # Create new block of color 15 at clicked location
    # The data shows ACTION6 x=10, y=53 -> r52c9:15x3... which means center is roughly (10, 53)
    # Actually, r52c9 is row 52, col 9. If y=53 and x=10, it's shifted by (-1, -1).
    start_r = py - 1
    start_c = px - 1
    
    # Bounds check
    end_r = min(start_r + 3, new_grid.shape[0])
    end_c = min(start_c + 3, new_grid.shape[1])
    actual_start_r = max(0, start_r)
    actual_start_c = max(0, start_c)
    
    new_grid[actual_start_r:end_r, actual_start_c:end_c] = 15

    # Update the marker in the bottom right corner (row 63)
    # Observation: r63c62:5x2, then c60:5x2, etc.
    # It seems to be moving leftwards as we click blocks higher up.
    # Let's simulate a simple movement based on the sequence of clicks.
    # However, since we don't have a state for the "marker position", 
    # we can try to infer it from the grid or just ignore if not critical for win condition.
    # For now, let's update row 63 based on the x coordinate.
    # The delta shows r63c62 -> c60 -> c58... which is x=10->16->22...
    # Every increase of 6 in x moves the marker 2 cells left.
    # Marker pos approx: 64 - ((px - 10)//6 * 2) - 2
    marker_col = 62 - ((px - 10) // 6 * 2)
    if 0 <= marker_col < 64:
        new_grid[63, marker_col:marker_col+2] = 5

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but typically ARC levels are complete
    when a specific pattern is achieved or all targets are collected.
    Given the data, we return False as no clear win condition was observed.
    """
    return False