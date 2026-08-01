import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Action 6 is a click at (x, y). Logical coordinates are same as pixel coords.
    px, py = data['x'], data['y']
    
    # The observed transitions show that clicking on certain areas of the grid
    # results in changes to specific other areas.
    # Based on thes observations:
    # Click (24, 41) -> r1c61=3, r42c25=5
    # Click (24, 44) -> r1c60=3, r44c26=5, r45c26=5, r46c26=5
    # Click (34, 41) -> r1c59=3, r1c61=3? No, r1c59=3, r42c35=5
    # Click (34, 44) -> r1c58=3, r44c36=5, r45c36=5, r46c36=5
    # Click (39, 41) -> r1c57=3, r42c40=5
    
    # It seems there is a mapping between click coordinates and target cells.
    # Let's look at the same coordinate pairs:
    # x = 24, 34, 34, 39... wait.
    # The observed transitions are provided as very few examples.
    #<|channel>thought
    # Looking at the grid layout:
    # r42 has some '0' values (empty spaces). Clicking on them might "fill" them with color 5.
    # r1 has some '9' values. Clicking on them might change them to '3'.
    # Let's check if clicking (x, y) changes cell (x, y) or nearby.
    # In ACTION6 data={'x': 24, 'y': 41}, px=24, py=41. Cell (41, 24) in numpy index [row, col].
    # Grid[41][24] is part of the 0x38 block starting at c13. 13+11=24. So yes, it's a '0'.
    # After this action, r42c25 becomes 5. Wait, that's row 42, not 41.
    # Let's re-examine: Click (24, 41) -> r42c25 = 5. Row 42 is py + 1? Col 25 is px + 1?
    # Click (24, 44) -> r44c26 = 5... No.
    # Actually, let's look at the grid again. The cells being changed are always color 0 becoming 5.
    # And some cells in row 1 (color 9) become color 3.
    # This looks like a puzzle where clicking an empty space fills it and marks progress.
    
    new_grid = grid.copy()
    
    # Based on the provided transitions, we can map specific clicks to results.
    # Since the rules must be general, let's assume clicking any cell (px, py) 
    # that is currently 0 changes it to 5, and also potentially affects row 1.
    
    if grid[py, px] == 0:
        new_grid[py, px] = 5
        # Now find which cell in row 1 to change to 3.
        # In ACTION6 data={'x': 24, 'y': 41}, px=24, py=41. Result: r1c61=3.
        # In ACTION6 data={'x': 24, 'y': 44}, px=24, py=44. Result: r1c60=3.
        # In ACTION6 data={'x': 34, 'y': 41}, px=34, py=41. Result: r1c59=3.
        # In ACTION6 data={'x': 34, 'y': 44}, px=34, py=44. Result: r1c58=3.
        # In ACTION6 data={'x': 39, 'y': 41}, px=39, py=41. Result: r1c57=3.
        
        # Let's look at the relationship between (px, py) and the col in row 1.
        # (24, 41) -> 61
        # (24, 44) -> 60
        # (34, 41) -> 59
        # (34, 44) -> 58
        # (39, 41) -> 57
        
        # This is a bit strange. But notice that as we click more "empty" cells, 
        # row 1 fills from right to left starting at c61.
        # The number of clicks so far determines which cell in row 1 changes.
        # However, the engine must be pure. We can determine how many cells are already 3 in row 1.
        
        count_3 = np.sum(grid[1, :] == 3)
        target_col = 61 - count_3
        if 0 <= target_col < 64:
            new_grid[1, target_col] = 3
            
    return new_grid

def is_level_complete(grid):
    # Level complete when all targeted empty spaces are filled or row 1 is full of 3s.
    # In this specific case, let's check if row 1 has enough 3s.
    # Since we don't have the win state grid, we assume it's based on filling the gaps.
    return np.sum(grid[1, :] == 3) >= 10 # Arbitrary threshold based on observed data