import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Extract click coordinates
    px, py = data['x'], data['y']
    
    # The game seems to be a "collection" or "toggling" puzzle where clicking on specific 
    # cells (likely targets or triggers) changes the state of the same-shaped object 
    # and other related elements.
    # Based on thes transitions:
    # ACTION6 at (10, 53) -> r52c9:15x3, r53c9:15x3, r54c9:15x3, etc.
    # It looks like the object at (py, px) is being replaced by a 3x3 block of color 15.
    # And some other blocks are also changing.
    # 
    # Let'<|channel>thought
    # Looking at the INITIAL GRID:
    # obj23 is color 0 at bbox=(52, 9, 54, 11). This matches the click at (10, 53).
    # When clicked, it becomes color 15 (a 3x3 block).
    # Simultaneously, the 3x3 block of color 15 at (58, 3, 60, 5) [obj26] becomes color 5.
    # Also, something happens at r63c62:5x2.
    # 
    # The pattern in subsequent clicks:
    # Click at (16, 47) -> r46c15:15x3, r47c15:15x3, r48c15:15x3...
    # These coordinates match the positions of the '3' colored pixels (obj20, etc.)
    # It seems clicking on a target pixel (color 0 or 3) transforms it into a 3x3 block of color 15.
    # And when a new 3x3 block appears, an old one disappears (becomes color 5).
    # 
    # Let's refine the rule:
    # 1. Find if the click is on a "target" cell (color 0 or 3).
    # 2. If so, create a 3x3 block of color 15 centered at (py, px).
    # 3. Remove the existing 3x3 block of color 15 elsewhere.
    # 4. Update some cells at the bottom row (r63).
    # 
    # Wait, looking closer at the deltas:
    # ACTION6 data={'x': 10, 'y': 53} -> r52c9:15x3, r53c9:15x3, r54c9:15x3
    # The center is (53, 10). This matches py=53, px=10.
    # Then r58c3:5x3, r59c3:5x3, r60c3:5x3. This is the 3x3 block at (59, 4) becoming color 5.
    # And r63c62:5x2.
    # 
    # Let's generalize:
    # - Click (px, py) transforms target cell into 3x3 block of color 15.
    # - Existing 3x3 block of color 15 becomes color 5.
    # - A marker moves in the bottom row (r63).
    # 
    # Looking at the sequence of clicks:
    # (10, 53), (16, 47), (22, 41), (28, 35), (34, 29)
    # These are exactly the positions of the targets.
    # Bottom row changes: c62->c60->c58->c56->c54. It moves left by 2 each time.

    new_grid = grid.copy()
    
    # Target check
    if new_grid[py, px] not in [0, 3]:
        return new_grid
    
    # Create 3x3 block of color 15 centered at (py, px)
    for r in range(py-1, py+2):
        for c in range(px-1, px+2):
            if 0 <= r < 64 and 0 <= c < 64:
                new_grid[r, c] = 15
    
    # Remove existing 3x3 blocks of color 15
    # We search for any 3x3 area that is all color 15 AND NOT the one we just created.
    for r in range(1, 63):
        for c in range(1, 63):
            if np.all(new_grid[r-1:r+2, c-1:c+2] == 15):
                # Check if this is the same center as our click
                if r != py or c != px:
                    # Turn it into color 5
                    new_grid[r-1:r+2, c-1:c+2] = 5
    
    # Bottom row marker movement
    # Find current marker (color 5 cells on row 63 that are not part of the background)
    # Actually, looking at INITIAL GRID, r63 is all 0s.
    # After first click: r63c62:5x2.
    # After second: r63c60:5x2.
    # It seems to be a progress bar.
    # Let's find where the '5's are on row 63 and move them left by 2.
    marker_cols = np.where(grid[63] == 5)[0]
    if len(marker_cols) > 0:
        start_col = marker_cols[0]
        end_col = marker_cols[-1]
        new_grid[63, start_col:end_col+1] = grid[63, start_col:end_col+1] # keep? no
        # The delta says "r63c62:5x2", then "r63c60:5x2". This means it replaces.
        # But wait, if we just set new_grid[63, col:col+2] = 5, what happens to old ones?
        # In the deltas, only the NEW positions are mentioned. 
        # This implies the rest of row 63 is reset or stays same.
        # Since INITIAL r63 was all 0, let's assume it resets.
        new_grid[63, :] = 0
        new_grid[63, max(0, start_col-2):max(0, start_col-2)+2] = 5
    else:
        # First click case: r63c62:5x2
        new_grid[63, 62:64] = 5

    return new_grid

def is_level_complete(grid):
    # Win state usually involves collecting all targets or reaching a certain progress.
    # Based on the pattern, maybe when the marker reaches column 0?
    # Or when no more target pixels (color 0 or 3) exist?
    # Let's check for any remaining color 0 or 3 in the main area.
    # But wait, we don't have a win grid. Let's guess based on common ARC patterns.
    if np.any((grid[10:63, 0:64] == 0) | (grid[10:63, 0:64] == 3)):
        return False
    return True