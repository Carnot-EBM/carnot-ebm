import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where clicking on specific areas triggers changes.
    # Based on the observed transitions, it's a "collection" or "toggling" mechanism.
    # Clicking at (x, y) creates a 3x3 block of color 15 and replaces another 3x3 block of color 15.
    # Each click is ACTION6 with coordinates x, y.
    #<|channel>thought
    # Let's analyze the clicks:
    # Click 1: (10, 53) -> r52c9, r53c9, r54c9 (3x3 block starting at c9)
    # Click 2: (16, 47) -> r46c15, r47c15, r48c15 (3x3 block starting at c15)
    # Click 3: (22, 41) -> r40c21, r41c21, r42c21 (3x3 block starting at c21)
    # Click 4: (28, 35) -> r34c27, r35c27, r36c27 (3x3 block starting at {c27})
    # Click 5: (34, 29) -> r28c33, r29c33, r30c33 (3x3 block starting at {c33})
    # In each case, clicking at (x, y) creates a new 3x3 block of color 15.
    # The coordinates x, y are pixel coords. Logical = Pixel.
    # A click at (x, y) seems to center the 3x3 block around (y, x).
    # Let's check:
    # Click 1: x=10, y=53. Block at r52-54, c9-11. Center is (53, 10).
    # Click 2: x=16, y=47. Block at r46-48, c15-17. Center is (47, 16).
    # Click 3: x=22, y=41. Block at r40-42, c21-23. Center is (41, 22).
    # Click 4: x=28, y=35. Block at r34-36, c27-29. Center is (35, 28).
    # Click 5: x=34, y=29. Block at r28-30, c33-35. Center is (30, 34). Wait, r28-30 center is 29.
    # So clicking at (x, y) creates a 3x3 block of color 15 centered at (y, x).
    # Also, the previous 3x3 block of color 15 created by ACTION6 seems to be replaced by color 5 (or original background).
    # Let's check:
    # Transition 1: r58c3, r59c3, r60c3 becomes 5x3. These were 15x3 in INITIAL.
    # Transition 2: r52c9, r53c9, r54c9 becomes 5x3. These were 15x3 from click 1.
    # Transition 3: r46c15, r47c15, r48c15 becomes 5x3. These were 15x3 from click 2.
    # Transition 4: r40c21, r41c21, r42c21 becomes 5x3. These were 15x3 from click 3.
    # Transition 5: r34c27, r35c27, r36c27 becomes 5x3. These were 15x3 from click 4.
    # So the rule is:
    # 1. Find all existing 3x3 blocks of color 15.
    # 2. The most recently created one (or a specific sequence) is replaced by background color 5.
    # 3. A new 3x3 block of color 15 is centered at (y, x).
    # Let's refine this: it seems to be a "cursor" or "token" moving.
    # There are multiple 3x3 blocks of color 15 in INITIAL grid?
    # Initial Grid check: r4-6 c30-32 is 15x3; r58-60 c3-5 is 15x3.
    # Click 1 replaces r58-60 c3-5 and creates r52-54 c9-11.
    # Click 2 replaces r52-54 c9-11 and creates r46-48 c15-17.
    # And so on.
    # Also there's some change at r63c62 etc. This looks like a score counter or progress bar.
    # r63c62 becomes 5x2... wait, the initial r63 was 0x64.
    # Transition 1: r63c62:5x2. (Cells 62, 63 become 5)
    # Transition 2: r63c60:5x2. (Cells 60, 61 become 5)
    # Transition 3: r63c58:5x2. (Cells 58, 59 become 5)
    # Transition 4: r63c56:5x2. (Cells 56, 57 become 5)
    # Transition 5: r63c54:5x2. (Cells 54, 55 become 5)
    # It fills from right to left in pairs of 2.

    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        
        # Find and remove existing "token" blocks of color 15
        # Based on observations, it removes a specific block. Let's find all 3x3 blocks of 15.
        # In this game, the token is the one that was most recently placed or at a certain position.
        # To be simple, let's replace any 3x3 block of 15 that isn't the static one at r4-6 c30-32.
        for r in range(grid.shape[0] - 2):
            for c in range(grid.shape[1] - 2):
                if np.all(grid[r:r+3, c:c+3] == 15):
                    if not (r == 4 and c == 30):
                        new_grid[r:r+3, c:c+3] = 5 # Background color for these areas
        
        # Place new token centered at (py, px)
        # Center (py, px) means top-left is (py-1, px-1)
        tr, tc = py - 1, px - 1
        if 0 <= tr < grid.shape[0]-2 and 0 <= tc < grid.shape[1]-2:
            new_grid[tr:tr+3, tc:tc+3] = 15
            
        # Update progress bar at r63
        # It fills from right to left in pairs of 2.
        # We need to know how many clicks have happened.
        # Let's count current filled cells in r63.
        filled_count = np.sum(grid[63] == 5)
        start_col = 63 - (filled_count + 2)
        if start_col >= 0:
            new_grid[63, max(0, start_col):max(0, start_col)+2] = 5

        return new_grid
    
    return grid

def is_level_complete(grid):
    # Level complete when the progress bar is full or a certain condition is met.
    # Based on observations, it's filling up.
    return np.all(grid[63] == 5)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for 'su15' (a 15-puzzle variant) is typically
    that the grid is sorted in ascending order with the empty cell (0) at the end.
    """
    grid = np.array(grid)
    flat_grid = grid.flatten()
    
    # The target state for a 15-puzzle is usually 1, 2, ..., 15, 0
    # Create a target sequence: 1 to N-1, then 0
    n = flat_grid.size
    target = np.arange(1, n)
    target = np.append(target, 0)
    
    return np.array_equal(flat_grid, target)
