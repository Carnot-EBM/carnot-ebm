import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    new_grid = grid.copy()
    
    # The game seems to be a puzzle where clicking on specific areas changes colors of blocks.
    # Based on the observed transitions, ACTION6 (click) at (px, py) targets a 
    # specific region and its surrounding cells.
    # We need to find if the click coordinates correspond to a certain "button" or "trigger".
    # 
    # Looking at the same coordinate clicks in the observations:
    # Transition 0->0: Click (18, 19) -> Changes color of some cells around y=17-22, x=16-22.
    #<|channel>thought
    # Transition 0->0: Click (30, 19) -> Changes color of some cells around y=17-22, x=24-35.
    # Transition 0->0: Click (30, 19) again? No, the delta shows it's shifting.
    # Let's look at the r0c0, r0c1... increments. This suggests a counter is being kept.
    # Actually, looking closely at the deltas:
    # Action 6 at (18, 19) affects rows 17-22.
    # Action 6 at (30, 19) affects rows 17-22.
    # Action 6 at (42, 19) affects rows 17-22.
    # It seems clicking on a "button" (color 1 or 14 blocks) toggles their state between 1 and 14.
    # The coordinates provided in data are logical coords (px, py).
    # In the INITIAL GRID, row 19 has values like 1x4, 0x2, 14x4...
    # Row 19, col 11-14 is color 1. Col 17-20 is color 14.
    # If you click (18, 19), px=18, py=19. Grid[19, 18] is color 14.
    # The delta shows r17c17:3x4, etc. This means it's changing a block of cells.
    
    # Let's implement a simple toggle rule for the clicked cell if it's part of a connected component of colors 1/14.
    # But wait, the deltas show changes to multiple areas including r0c0.
    # r0c0:1x1, then r0c1:1x1... this looks like a progress marker.
    
    # Given the complexity and limited samples, let's try to find the "button" at (py, px)
    # and toggle its group.
    
    target_val = grid[py, px]
    if target_val == 0 or target_val == 10:
        return new_grid

    # Find connected component of same value
    component = []
    stack = [(py, px)]
    visited = set([(py, px)])
    while stack:
        r, c = stack.pop()
        component.append((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 64 and 0 <= nc < 64 and (nr, nc) not in visited and grid[nr, nc] == target_val:
                visited.add((nr, nc))
                stack.append((nr, nc))
    
    # Toggle between 1 and 14
    toggle_map = {1: 14, 14: 1}
    if target_val in toggle_map:
        new_val = toggle_map[target_val]
        for r, c in component:
            new_grid[r, c] = new_val
            
    # The progress marker at row 0 is likely based on how many buttons are correctly toggled.
    # However, we don't have enough data to know the win condition or the exact logic for r0.
    # Let's just implement the button toggle.
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves all "buttons" being a certain color or a specific pattern.
    # In this game, maybe it's when row 0 has a certain number of cells filled?
    # Or simply check if any cell in row 0 is non-zero (as seen in deltas).
    # But that happens after one click.
    # Let's assume complete if most blocks are color 14 or some other goal.
    # Without more data, return False unless a clear signal exists.
    return np.any(grid[0, :] != 0) and np.sum(grid[0, :]) > 5 # Placeholder