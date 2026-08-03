import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    # The observed transitions show that clicking on specific coordinates (x, y)
    # triggers changes in two locations: a target cell at (py, px) and a corresponding
    # marker cell at (1, 63-px) or similar.
    # Based on the observations:
    # x=24, y=41 -> r1c61:3x1, r42c25:5x3
    # x=24, y=44 -> r1c60:3x1, r44c26:5x1, r45c26:5x1, r46c26:5x1
    # x=34, y=41 -> r1c59:3x1, r42c35:5x3
    # x=34, y=44 -> r1c58:3x1, r44c36:5x1, r45c36:5x1, r46c36:5x1
    # x=39, y=41 -> r1c57:3x1, r42c40:5x3
    # Note: The coordinates in data are pixel coords which match logical coords here.
    # Let's analyze the relationship between (px, px) and the changed cells.
    # For x=24, y=41: target is around (41, 24). Changed cell at r1c61. 63-24 = 39? No.
    # 63 - 24 = 39. But it's c61. 63-24+something...
    # Looking at the same pattern:
    # x=24, y=41 -> r1c61. 61 - 24 = 37.
    # x=24, y=44 -> r1c60. 60 - 24 = 36.
    # x=34, y=41 -> r1c59. 59 - 34 = 25.
    # x=34, y=44 -> r1c60? No, r1c58. 58 - 34 = 24.
    # x=39, y=41 -> r1c57. 57 - 39 = 18.
    # Actually, let's look at the coordinates again.
    # The marker cells are in row 1. Row 1 is a long line of color 9.
    # Let's see if clicking (px, py) changes grid[py, px] to 5.
    # For x=24, y=41: r42c25 is changed to 5. (41+1, 24+1).
    # For x=24, y=44: r44-46 c26 is changed to 5. (44, 24+2)? No.
    # Wait, the delta says "r42c25:5x3". This means starting at col 25, 3 cells become 5.
    # So for x=24, y=41: (42, 25), (42, 26), (42, 27) become 5.
    # For x=24, y=44: (44, 26), (45, 26), (46, 26) become 5.
    # It seems the click coordinates (px, py) are slightly offset or refer to a specific object.
    # The marker cell in row 1 is also changing from 9 to 3.
    # Let's implement a simple rule: if action 6, change grid[py, px] and its neighbors to 5,
    # and change a corresponding cell in row 1 to 3.
    #
    # Looking closer at the data:
    # x=24, y=41 -> r1c61:3x1, r42c25:5x3
    # x=24, y=44 -> r1c60:3x1, r44c26:5x1, r45c26:5x1, r46c26:5x1
    # x=34, y=41 -> r1c59:3x1, r42c35:5x3
    # x=34, y=44 -> r1c58:3x1, r44c36:5x1, r45c36:5x1, r46c36:5x1
    # x=39, y=41 -> r1c57:3x1, r42c40:5x3
    # The marker cells are (1, 61), (1, 60), (1, 59), (1, 58), (1, 57).
    # These correspond to the clicks in order.
    # Let's assume clicking on a cell that is not color 5 changes it and some neighbors to 5,
    # and moves a "cursor" in row 1 from right to left.
    
    new_grid = grid.copy()
    if action == 6:
        # Find current cursor position in row 1 (the first '3' or something)
        # Row 1 has colors [5, 9, ..., 9, 5, 5]
        # We can find the last index of color 9 that was changed to 3.
        # In the initial grid, row 1 is 5x1, 9x61, 5x2.
        # Indices 1 to 61 are color 9.
        # Index 62 is color 5.
        # For x=24, y=41: r1c61 becomes 3.
        # For x=24, y=44: r1c60 becomes 3.
        # For x=34, y=41: r1c59 becomes 3.
        # For x=34, y=44: r1c58 becomes 3.
        # For x=39, y=41: r1c57 becomes 3.
        # The marker cells are being filled from right to left starting at col 61.
        
        # Let's identify which cell in row 1 should be changed to 3.
        # Count how many 3s already exist in row 1.
        num_3s = np.sum(grid[1, :] == 3)
        marker_col = 61 - num_3s
        new_grid[1, marker_col] = 3
        
        # Now handle the target area change.
        # Based on the observations, clicking (px, py) changes a small region to color 5.
        # We can simply set grid[py, px] and its immediate neighbors to 5 if it was not 5.
        # If we click on an object of color 0, 1, 4, or 11, we "clear" it by turning it into 5.
        # Let's try to find the connected component of the clicked cell.
        # From the data:
        # x=24, y=41 -> r42c25:5x3 (cells (42, 25), (42, 26), (42, 27))
        # x=24, y=44 -> r44-46 c26 (cells (44, 26), (45, 26), (46, 26))
        # la
        # For x=24, y=41: Click (24, 41). Target cells are (42, 25..27).
        # For x=24, y=44: Click (24, 44). Target cells are (44..46, 26).
        # For x=24, y=41 is near (41, 24) and (42, 25).
        # For x=24, y=44 is near (44, 24) and (44, 26).
        # Let's just set a small region around (py, px) to 5.
        # The most accurate way is to find the connected component of non-5 cells.
        # We can actually see that clicking on a cell changes its entire "block" or "line" to 5.
        # For x=24, y=41: r42c25:5x3 means a horizontal line of 3 cells.
        # For x=24, y=44: r44-46 c26 means a vertical line of 3 cells.
        # This looks like it's clearing blocks of color 0, 1, 4, 11.
        # Let's implement this: if grid[py, px] is not 5, find all connected cells of the same color as grid[py, px],
        # and turn them into 5.
        
        target_color = grid[py, px]
        if target_color != 5:
            # Simple flood fill to clear the block
            stack = [(py, px)]
            visited = set()
            while stack:
                curr_y, curr_x = stack.pop()
                if (curr_y, curr_x) in visited: continue
                visited.add((curr_y, curr_x))
                if 0 <= curr_y < 64 and 0 <= curr_x < 64 and grid[curr_y, curr_x] == target_color:
                    new_grid[curr_y, curr_x] = 5
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        stack.append((curr_y + dy, curr_x + dx))
            
    return new_grid

def is_level_complete(grid):
    # The win state is not provided, but typically it's when all "targets" are cleared.
    # In this game, targets seem to be blocks of colors other than 5 and 9.
    # Let's assume the level is complete when no cells of color 0, 1, 4, or 11 remain.
    # Check if any cell has a color that is not 5, 9, or 3.
    # return np.any((grid != 5) & (grid != 9) & (grid != 3))
    # We can actually see from the initial grid that there are many such cells.
    # For example, row 8 contains color 0.
    # return False # Default for now as we cannot determine the win condition.
    # Based on common ARC patterns, it's usually about clearing something.
    # Let's try:
    # return not np.any((grid != 5) & (grid != 9) & (grid != 3))
    # return True if everything is just background (5), marker line (9/3), and maybe some others.
    return not np.any((grid != 5) & (grid != 9) & (grid != 3))