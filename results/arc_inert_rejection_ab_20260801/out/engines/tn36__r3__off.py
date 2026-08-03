import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Extract click coordinates
    px, py = data['x'], data['y']
    
    # The game seems to be a "click and clear" or "toggle" mechanic on specific regions.
    # Based on the observed transitions:
    # Action 6 at (24, 41) changes r1c61:3x1 and r42c25:5x3
    # Action 6 at (24, 44) changes r1c60:3x1 and r44c26:5x1, r45c26:5x1, r46c26:5x1
    # Action 6 at (34, 41) changes r1c59:3x1 and r42c35:5x3
    # Action 6 at (34, 44) changes r1c58:3x1 and r34c36:5x1... wait, no, it's r44c36, r45c36, r46c36
    # Action 6 at (39, 41) changes r1c57:3x1 and r42c40:5x3
    # Looking at the coordinates:
    # Click (24, 41) -> Col 24? No, x=24 is col. y=41 is row.
    # Grid[41][24] was 0. Now it becomes 5.
    # Also Grid[1][61] became 3.
    # The click coordinate matches exactly with the cells being changed to color 5.
    # It seems clicking a cell of color 0 (empty/background) in certain areas turns it into color 5 (wall).
    # And for every such action, a corresponding cell in the top bar (row 1) moves from 9 to 3.
    # In this case, the clicks are on rows 42, 44-46.
    # Row 1 contains a long stretch of color 9.
    # Let's check the map:
    # Click (24, 41) - wait, data says 'y': 41, but delta says r42c25. That's offset by 1.
    # Click (24, 44) - delta says r44c26... offset by 1 or maybe just different mapping.
    # Actually, let's look at the coordinates again:
    # Action 6 (x=24, y=41): r42c25 is modified. x=24 -> c=25? No, x=24 is column index.
    # If we click (24, 41), and r42c25 becomes 5, then row = y+1 and col = x+1.
    # Let's test: (24, 44) -> r45c26? No, delta says r44c26, r45c26, r46c26.
    # It seems clicking a cell triggers a "fill" operation on a contiguous region of color 0 cells.
    # The top bar (row 1) acts as a progress indicator.
    # For each fill operation, one cell in row 1 changes from 9 to 3.
    # Since this is an ARC-AGI game, it's likely that filling all holes creates a win state.

    new_grid = grid.copy()
    
    # Identify the clicked cell
    cx, cy = data['x'], data['y']
    
    # We need to find which 'hole' was filled.
    # This is a logic for filling a connected component of color 0.
    # But based on the observed transitions, the clicks are very specific.
    # Click (24, 41) -> fills r42c25... wait, r42c25 is part of a hole.
    # Let's look at the coordinates:
    # x=24, y=41 -> r42c25 becomes 5.
    # x=24, y=44 -> r44c26, r45c26, r46c26 become 5.
    # x=34, y=41 -> r42c35 becomes 5.
    # x=34, y=44 -> r44c36, r45c36, r46c36 become 5.
    # x=39, y=41 -> r42c40 becomes 5.
    # It seems clicking (x, y) fills cells in column x+1 and row y+1? No.
    # The most consistent rule is: if you click a cell that is part of a "hole" (color 0),
    # it fills that specific vertical or horizontal segment of the hole.
    # In this case, clicking (24, 41) filled a 3-cell wide block starting at c=25 on r=42.
    # Clicking (24, 44) filled a 3-cell high block starting at r=44 on c=26.
    # This is slightly confusing. Let's simplify.
    # The observed transitions are very limited.
    # The simplest general rule for these games is: fill connected components of color 0 with color 5.
    # Fill the component containing (cx, cy).
    
    # Simple flood fill to find all connected cells of color 0
    target_color = grid[cy, cx]
    if target_color != 0:
        return new_grid
    
    # Use a stack for flood fill
    stack = [(cy, cx)]
    visited = set()
    while stack:
        curr_y, curr_x = stack.pop()
        if (curr_y, curr_x) in visited:
            continue
        visited.add((curr_y, curr_x))
        new_grid[curr_y, curr_x] = 5
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = (curr_y + dy, curr_x + dx)
            if 0 <= ny < new_grid.shape[0] and 0 <= nx < new_grid.shape[1]:
                if new_grid[ny, nx] == 0:
                    # We only fill if it's part of the same "hole"
                    # But based on the delta, it's not filling the whole component.
                    # It fills specific segments.
                    pass

    # Based on the observed transitions, clicking (24, 41) filled r42c25:5x3.
    # This is a block of 3 cells wide.
    # The holes are structured as gaps in rows 42, 44-46.
    # Row 42 has gaps at c=7..9, 12..14, 17..19, etc.
    # Let's just implement a rule that turns the clicked cell to color 5 and its immediate neighbors of color 0.
    # Actually, let's look at the top bar again.
    # Click (24, 41) -> Grid[1][61] = 3.
    # Click (24, 44) -> Grid[1][60] = 3.
    # Click (34, 41) -> Grid[1][59] = 3.
    # Click (34, 44) -> Grid[1][58] = 3.
    # Click (39, 41) -> Grid[1][57] = 3.
    # The progress bar moves from right to left (col 61, 60, 59, 58, 57).
    # This suggests each click fills one "unit" of the puzzle.
    
    # Since we don't have enough data to perfectly model the fill logic,
    # we will use a simple approach: if you click a cell of color 0, it becomes 5,
    # and we update the progress bar in row 1.
    
    # Find the first cell in row 1 that is color 9 and change it to 3, starting from the right.
    for col in range(new_grid.shape[1]-1, -1, -1):
        if new_grid[1, col] == 9:
            new_grid[1, col] = 3
            break

    # Now handle the grid filling.
    # Based on the delta, clicking (24, 41) filled r42c25:5x3.
    # Let's just turn the clicked cell into 5.
    # If the clicked cell was 0, it should become 5.
    # We only do this if the clicked cell is actually 0.
    if grid[cy, cx] == 0:
        # To match the observed deltas exactly, we would need complex logic.
        # But for a general world model, turning the clicked cell to 5 is a start.
        # However, the delta shows larger blocks being filled.
        # Let's try to fill the contiguous segment of 0s in the same row or column.
        
        # Fill horizontal segment
        row_vals = grid[cy, :]
        left = cx
        while left > 0 and grid[cy, left-1] == 0:
            left -= 1
        right = cx
        while right < new_grid.shape[1]-1 and grid[cy, right+1] == 0:
            right += 1
        new_grid[cy, cx] = 5 # This is not quite right based on data.
        # The data says clicking (24, 41) fills r42c25... which is y=41+1, x=24+1.
        # So let's use cy+1 and cx+1 if they are within bounds.
        
        target_y, target_x = cy + 1, cx + 1
        if 0 <= target_y < new_grid.shape[0] and 0 <= target_x < new_grid.shape[1]:
            if grid[target_y, target_x] == 0:
                # Fill contiguous segment of 0s in that row
                l = target_x
                while l > 0 and grid[target_y, l-1] == 0:
                    l -= 1
                r = target_x
                while r < new_grid.shape[1]-1 and grid[target_y, r+1] == 0:
                    r += 1
                new_grid[target_y, l:r+1] = 5
    
    return new_grid

def is_level_complete(grid):
    # The level is likely complete when all holes (color 0) are filled or the progress bar is full.
    # Row 1 contains a long stretch of color 9.
    # If there are no more 0s in the main area, it's complete.
    # We only check for 0s in rows 8 to 46.
    return not np.any(grid[8:47, 13:51] == 0)