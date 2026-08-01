import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is a 64x64 integer array.
    # Action 6 is a click at (data['x'], data['y']).
    # Action 3 is a directional move (left/right/up/down - though not explicitly labeled in the same way)
    # Based on the observed transitions, ACTION3 seems to be "shift" or "toggle" and ACTION6 is "fill".
    # The game involves modifying blocks of color 10 (gray) and replacing them with patterns of colors 5, 9, 11.
    # Looking at thes deltas, it's clear that a specific region is being modified.
    # In the process, a counter at row 63 increases.
    # The cells in row 63 are acting as a progress bar.
    
    new_grid = grid.copy()
    
    if action == 6:
        # Click fills a 6x5 area (or similar) with color 10.
        # x is column, y is row.
        px, py = data['x'], data['y']
        # It appears the fill happens around the pixel coordinates provided.
        # We need to find the block size. From delta r37c25:10x6...r41c25:10x6, it's rows 37-41 (5 rows) and cols 25-30 (6 cols).
        # Offset from click point (24, 36): px=24, col_start=25, col_end=30; py=36, col_start=37, row_end=41.
        # This means start_row = py + 1, start_col = px + 1.
        # Block size: height=5, width=6.
        start_row = py + 1
        start_col = px + 1
        for r in range(start_row, start_row + 5):
            if 0 <= r < 64:
                new_grid[r, start_col : start_col + 6] = 10
        # Increment progress bar at row 63.
        # Find first color 5 cell in row 63 starting from column 0.
        # For example, if grid[63, 0] is 15, then we are most out of the same block.
        # The same logic applies to other times ACTION6 is clicked.
        # We need to find where the current "head" of the progress bar is.
        # npthought
        current_progress = 0
        while current_progress < 64 and new_grid[63, current_progress] == 15:
            current_progress += 1
        new_grid[63, current_progress] = 15 if current_progress < 64 else 15
        # Wait, looking at INITIAL GRID r63:15x5, so it starts with five 15s.
        # Let's just increment the number of 15s in row 63.
        # # Correcting: let's count how many 15s there are.
        count_15 = np.sum(new_grid[63, :] == 15)
        new_grid[63, count_15] = 15
    
    elif action == 3:
        # Action 3 replaces a block of color 10 with a pattern.
        # It seems to be replacing blocks of 10 that are 5 rows high and 6 columns wide.
        # In the same transition, it's an even larger area is sometimes modified.
        # Looking at ACTION3 (level 0->0): changed cells r37c31:5x2,9x1,5x2...r41c38:10x3.
        # This is replacing a 5x6 block of 10s starting at (37, 31).
        # The logic for the pattern is:
        # Row 0: 5, 5, 9, 5, 5 (wait, 5x2, 9x1, 5x2 -> total 5 cols)
        # Let's look at the delta again: r37c31:5x2,9x1,5x2 (total 5), then r37c37:10x5 (this is separate).
        # Wait, the pattern is actually applied to a 5-column width? Or 6?
        # 5x2, 9x1, 5x2 = 5 columns.
        # Let's re-examine the first ACTION3: r37c37:5x2,9x1,5x2 (cols 37-41) and r37c43:10x5 (cols 43-47).
        # It seems Action 3 "shifts" the pattern from one block to another.
        # Looking closely at the deltas, it replaces color 10 with colors 5, 9, 11.
        # The pattern for rows 37, 38, 39, 40 are slightly different.
        # Row 37: 5, 5, 9, 5, 5 (width 5)
        # Row 38: 5, 11, 9, 9, 5 (width 5)
        # Row 39: 5, 11, 9, 9, 5 (width 5)
        # Row 40: 5, 5, 9, 5, 5 (width 5)
        # Row 41: 5, 5, 5, 10, 10, 10 (this is weird).
        
        # Let's simplify. ACTION3 appears to be a "solve" action that converts blocks of 10 into patterns.
        # Since we don't have coordinates for ACTION3, it must target specific areas or the most recent block.
        # Based on the observed transitions, Action 3 targets the same area as the last ACTION6 click.
        # It seems to replace color 10 with colors 5, 9, and 11 in a 5x6 region.
        
        # Find the most recent 5x6 block of 10s.
        # For simplicity, let's just find any 5x6 block of 10s and apply the pattern.
        # found = False
        # for r in range(64):
        #     for c in range(64 - 6):
        #         if np.all(new_grid[r:r+5, c:c+6] == 10):
        #             # Apply pattern
        #             new_grid[r:r+5, c:c+6] = ...
        #             # Increment progress bar
        #             count_15 = np.sum(new_grid[63, :] == 15)
        #             new_grid[63, count_15] = 15
        #             break
        #             found = True
        #             break
        # Let's implement this logic more generally.
        
        # Based on ACTION3 (level 0->0), it targets blocks at (37, 37), then (37, 31), then (37, 25), then (37, 19).
        # This is a sequence of blocks moving left by 6 columns each time.
        # The block size is 6 cols wide and 5 rows high.
        # Target the most right-most 5x6 block of 10s starting at row 37.
        target_col = -1
        for c in range(63, -1, -1):
            if np.all(new_grid[37:42, c:c+6] == 10 if c+6 <= 64 else False):
                target_col = c
                break
        
        if target_col != -1:
            # Apply pattern to new_grid[37:42, target_col : target_col + 6]
            # Row 37: 5, 5, 9, 5, 5 (cols 0-4)
            # Row 38: 5, 11, 9, 9, 5 (cols 0-4)
            # Row 39: 5, 11, 9, 9, 5 (cols 0-4)
            # Row 40: 5, 5, 9, 5, 5 (cols 0-4)
            # Row 41: 5, 5, 5 (cols 0-2)
            pattern = np.array([
                [5, 5, 9, 5, 5],
                [5, 11, 9, 9, 5],
                [5, 11, 9, 9, 5],
                [5, 5, 9, 5, 5],
                [5, 5, 5, 10, 10] # This is a bit of a guess
            ])
            for r_off in range(5):
                for c_off in range(5):
                    if c_off < pattern.shape[1]:
                        new_grid[37 + r_off, target_col + c_off] = pattern[r_off, c_off]
            
            count_15 = np.sum(new_grid[63, :] == 15)
            new_grid[63, count_15] = 15
    
    elif action == 4:
        # Action 4 seems to be "undo" or "shift". It replaces the pattern with color 10 again?
        # No, it's replacing some cells with 10 and others with patterns.
        # Let's just ignore ACTION 4 for now as it's complex.
        pass

    return new_grid

def is_level_complete(grid):
    # The level is complete when row 63 is filled with 15s (or reaches a certain length).
    # return np.all(grid[63, :] == 15)
    # Based on INITIAL GRID, row 63 starts with five 15s.
    # We don't have WIN STATE grid, but usually ARC games end when a goal is reached.
    # Let's assume completion is when row 63 has more than 60 15s.
    return np.sum(grid[63, :] == 15) >= 60

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for bp35 is that all cells of the same color 
    must be connected (4-connectivity) and each color must form a 
    single contiguous region.
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid)
    
    for color in unique_colors:
        # Find all cells of the current color
        cells = np.argwhere(np.array(grid) == color)
        if len(cells) == 0:
            continue
            
        # Use BFS to find all reachable cells of the same color
        start_node = tuple(cells[0])
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            r, c = tuple(curr)
            # Check 4-connectivity
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        # If the number of visited cells is not equal to the total cells of that color,
        # the color is not connected.
        if len(visited) != len(cells):
            return False
            
    return True
