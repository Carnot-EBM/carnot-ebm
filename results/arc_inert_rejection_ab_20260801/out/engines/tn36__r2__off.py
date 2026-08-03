import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # The observed transitions show that clicking on a specific cell (x, y)
    # triggers changes in two distant locations: the top row (r1) and a target area
    # in the bottom half of the grid.
    # Based on thes deltas:
    # Click (24, 41) -> r1c61=3, r42c25=5
    # Click (24, 44) -> r1c60=3, r44c26=5, r45c26=5, r46c26=5
    # Click (34, 41) -> r1c59=3, r1c61=3? No, r1c59=3, r42c35=5
    # Click (34, 44) -> r1c58=3, r44c36=5, r45c36=5, r46c36=5
    # Click (39, 41) -> r1c57=3, r42c40=5
    
    # It seems there is a mapping between click coordinates (x, y) and the same cells being changed to color 5.
    # Let's look at the target area changes:
    # (24, 41) -> r42c25 = 5 (span of 3)
    # (24, 44) -> r44-46 c26 = 5 (span of 1 each row)
    # (24, 41) corresponds to x=24, y=41. The change happens at row 42, col 25.
    # (24, 44) corresponds to x=24, y=44. The change happens at rows 44, 45, 46, col 26.
    # (24, 41) -> r42, c25. (y+1, x+1)? No.
    # (34, 41) -> r42, c35. (y+1, x+1).
    # (39, 41) -> r42, c40. (y+1, x+1).
    # (24, 44) -> r44, 45, 46, c26? Wait, if click is (x, y), target cell is (y+1, x+1).
    # Let's check: Click (24, 44) -> Target cells are (44, 26), (45, 26), (46, 26).
    # This looks like it fills a gap in the grid pattern.
    
    # Based on the observed deltas, we can implement a simple rule:
    # If clicking on a cell that is color 0 or 1, it changes to color 5.
    # The action is essentially "filling" holes of color 0 or 1.
    # The same time, some cells in row 1 are changed to color 3.
    # Row 1 cells change from 9 to 3.
    # (24, 41) -> r1c61=3. (64 - 1 - 2 = 61).
    # (24, 44) -> r1c60=3. (64 - 1 - 3 = 60).
    # (34, 41) -> r1c59=3. (64 - 1 - 4 = 59).
    # (34, 44) -> r1c58=3. (64 - 1 - 5 = 58).
    # (39, 41) -> r1c57=3. (64 - 1 - 6 = 57).
    
    # Let's simplify: the click target (y, x) is modified to color 5 if it was not 5.
    # If we click on a cell that "belongs" to a gap, it fills the gap.
    # The same time, a counter in row 1 moves leftwards.
    # Since this is an ARC-AGI task, and the observed data is very limited,
    # the most likely rule is that clicking on any non-5 cell changes it to 5,
    # and some other cells change as well.
    
    new_grid = grid.copy()
    if action == 6:
        px, py = data['x'], data['y']
        # Fill the clicked cell and its connected components of non-5 colors?
        # Or just fill the specific hole being clicked?
        # Target area filling based on deltas:
        # Click (24, 41) -> r42c25:5x3. This means (42, 25), (42, 26), (42, 27) become 5.
        # Click (24, 44) -> r44c26:5x1, r45c26:5x1, r46c26:5x1.
        # (44, 26), (45, 26), (46, 26) become 5.
        # Target cells are always changing from something else to 5.
        # (x, y) click coordinates map to target cells.
        # Let's assume clicking a cell changes it and any adjacent non-5 cells of same color to 5.
        # Use a simple flood fill or similar logic for the "hole" being filled.
        
        target_color = grid[py, px]
        if target_color != 5:
            # Simple flood fill to change connected components of target_color to 5
            stack = [(py, px)]
            visited = set()
            while stack:
                curr_y, curr_x = stack.pop()
                if (curr_y, curr_x) in visited: continue
                visited.add((curr_y, curr_x))
                if not (0 <= curr_y < 64 and 0 <= curr_x < 64): continue
                if grid[curr_y, curr_x] == target_color:
                    new_grid[curr_y, curr_x] = 5
                    stack.append((curr_y + 1, curr_x))
                    stack.append((curr_y - 1, curr_x))
                    stack.append((curr_y, curr_x + 1))
                    stack.append((curr_y, curr_x)) # Wait, this is a bug.
                    stack.append((curr_y, curr_x - 1))
        
        # The counter in row 1 also changes. We need to find the rightmost cell that is color 9 and change it to 3.
    return new_grid

def engine(grid, action, data):
    # Refined engine based on observed deltas
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # Fill the clicked hole (flood fill of non-5 colors)
    target_color = grid[py, px]
    if target_color != 5:
        stack = [(py, px)]
        visited = set()
        while stack:
            curr_y, curr_x = stack.pop()
            # Check bounds
            if not (0 <= curr_y < 64 and 0 <= curr_x < 64): continue
            if (curr_y, curr_x) in visited: continue
            visited.add((curr_y, curr_x))
            if grid[curr_y, curr_x] == target_color:
                new_grid[curr_y, curr_x] = 5
                stack.append((curr_y + 1, curr_x))
                stack.append((curr_y - 1, curr_x))
                stack.append((curr_y, curr_x + 1))
                stack.append((curr_y, curr_x - 1))
    
    # Update row 1 counter: find rightmost cell that is color 9 and change it to 3.
    for c in range(63, -1, -1):
        if new_grid[1, c] == 9:
            new_grid[1, c] = 3
            break
    
    return new_grid

def is_level_complete(grid):
    # The win state is usually when all holes are filled or a specific condition is met.
    # In this case, the most likely win state is no non-5 cells except for the border/background.
    # Let's check if any cells of color 0, 1, 4, 11 are present in the grid.
    # For simplicity, let's assume the level is complete when there are no more "holes" (color 0, 1, 4, 11).
    # Check if only colors 5, 3, 9 are left.
    for r in range(64):
        for c in range(64):
            if grid[r, c] in [0, 1, 4, 11]:
                return False
    return True