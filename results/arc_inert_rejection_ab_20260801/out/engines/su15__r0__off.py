import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The game seems to be a "collect" or "clear" puzzle where clicking on specific targets
    # changes their color to 15 and then potentially clears them from the same area.
    # x, y are pixel coordinates which match logical coordinates in this case.
    px, py = data['x'], data['y']
    
    # Based on the observed transitions, clicking at (px, py) creates a 3x3 block of color 15.
    # Create a copy of the grid to actually modify it.
    new_grid = grid.copy()
    
    # Define the target region for the click.
    # In the observed transitions, ACTION6 clicks occur at (10, 53), (16, 47), (22, 41), (28, 35), (34, 29).
    # These are all points that were originally color 0 or 3.
    # Target region is a 3x3 block centered around (py, px).
    # Note: the delta shows r52c9:15x3, r53c9:15x3, r54c9:15x3 for click (10, 53).
    #<|channel>thought
    # Click (10, 53) -> rows 52, 53, 54; cols 9, 10, 11.
    # This means center is (53, 10).
    # Let's check others:
    # Click (16, 47) -> rows 46, 47, 48; cols 15, 16, 17.
    # Center is (47, 16).
    # So it's a 3x3 block at row = py-1 to py+1 and col = px-1 to px+1.
    
    # Apply the change: set target 3x3 area to color 15.
    for r in range(py - 1, py + 2):
        for c in range(px - 1, px + 2):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                new_grid[r, c] = 15
    
    # The observed deltas also show changes in other areas.
    # Specifically, there are changes in the "bottom" region (rows 58-60, cols 3-5)
    # and some cells in the very last row (row 63).
    # It looks like clicking targets clears them and updates a progress bar or score.
    # Let's implement the specific behavior seen in the transitions.
    
    # In transition 1: click (10, 53) -> rows 52-54, cols 9-11 become 15.
    # Also rows 58-60, cols 3-5 become 5.
    # And row 63, col 62 becomes 5.
    
    # Transition 2: click (16, 47) -> rows 46-48, cols 15-17 become 15.
    # Also rows 52-54, cols 9-11 become 5 (clearing previous target).
    # And row 63, col 60 becomes 5.
    
    # This suggests that only one "active" 3x3 block of color 15 exists at a time.
    # When a new one is created, the old one is reset to its background color (color 5).
    # The cells in row 63 are filled from right to left as a counter.
    
    # Find existing blocks of color 15.
    # We need to know what the "background" color for these targets was.
    # Looking at INITIAL GRID, those areas were mostly color 5.
    bg_color = 5
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15:
                new_grid[r, c] = bg_color

    # Now apply the new 3x3 block.
    for r in range(py - 1, py + 2):
        for c in range(px - 1, px + 2):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                new_grid[r, c] = 15
    
    # Update the progress bar in row 63.
    # Count how many times ACTION6 has been called or based on current state.
    # Let's count how many '5's are already in row 63 from the right.
    count = 0
    for c in range(grid.shape[1]-1, -1, -1):
        if grid[63, c] == 5:
            count += 1
        else:
            break
    
    # The delta shows r63c62:5x2 for first click (count=0), then r63c60:5x2 (count=2).
    # This means each click adds two cells of color 5 to the left of existing ones.
    target_col = grid.shape[1] - 2 - (count * 2) # Simplified logic based on observations
    # Actually, let's just look at the delta: 62->60->58->56->54.
    # It moves by 2 columns each time.
    start_col = 62 - (count // 2 if count > 0 else 0) # Not quite.
    # Let's use a simpler approach: find the leftmost '5' in row 63 and put more to its left.
    leftmost_5 = grid.shape[1]
    for c in range(grid.shape[1]):
        if grid[63, c] == 5:
            leftmost_5 = c
            break
    
    if leftmost_5 == grid.shape[1]:
        new_grid[63, 62:64] = 5
    else:
        new_grid[63, max(0, leftmost_5-2):max(0, leftmost_5)] = 5

    return new_grid

def is_level_complete(grid):
    # Level complete when the progress bar in row 63 is filled or targets are gone.
    # Based on observed data, we don't have a win state, but usually it's filling the bar.
    return np.any(grid[63, :] == 5) and np.sum(grid[63, :] == 5) >= 10