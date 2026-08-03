import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where clicking on specific coordinates (ACTION6)
    toggles the state of a 3x3 block and affects other blocks or markers.
    Based on the observed transitions:
    - Clicking at (x, y) changes a 3x3 area centered around (y, x).
    - The value in that 3x3 area toggles between color 15 and color 5 (or similar background colors).
    - There's a pattern of 'cascading' effects where previous clicks are reverted or modified.
    - A marker moves along the bottom row (r63).
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # In ARC-AGI logical coords, usually y is row and x is col.
    # Based on ACTION6 data={'x': 10, 'y': 53} -> r52c9:15x3, r53c9:15x3, r54c9:15x3
    # This implies center is roughly (py, px). 
    # Let's use the exact offsets from the delta:
    # For x=10, y=53: rows 52, 53, 54; cols 9, 10, 11.
    # Center would be (53, 10). The block is [py-1:py+2, px-1:px+2].
    
    row_start, row_end = py - 1, py + 2
    col_start, col_end = px - 1, px + 2
    
    # Clip to boundaries
    r_s, r_e = max(0, row_start), min(64, row_end)
    c_s, c_e = max(0, col_start), min(64, col_end)
    
    # Toggle logic for the clicked 3x3 area
    # If it was background (5 or 4), make it 15. If it was 15, make it background.
    for r in range(r_s, r_e):
        for c in range(c_s, c_e):
            if new_grid[r, c] == 15:
                new_grid[r, c] = 5 if grid[10, 0] == 5 else 4
            else:
                new_grid[r, c] = 15

    # The observed transitions show that clicking a new point often reverts 
    # the previous click's block back to its original state.
    # This suggests a "single active block" or "path" mechanic.
    # However, since we don't have the full history of clicks in the engine call,
    # and the delta shows specific cells changing, let's look at the pattern:
    # Click 1 (10, 53) -> Block A becomes 15.
    # Click 2 (16, 47) -> Block B becomes 15, Block A becomes 5/3.
    # This is hard to implement without state. But wait, the deltas are provided.
    # Let's check for existing blocks of color 15 and revert them.
    
    for r in range(64):
        for c in range(64):
            if new_grid[r, c] == 15:
                # If this cell is not part of the current clicked block, revert it?
                if not (r_s <= r < r_e and c_s <= c < c_e):
                    # The observed data shows some complex reversion (some become 3).
                    # For simplicity, we revert to background.
                    new_grid[r, c] = 5 if grid[10, 0] == 5 else 4

    # Marker movement on row 63:
    # x=10 -> c62:5x2 (cols 62, 63)
    # x=16 -> c60:5x2 (cols 60, 61)
    # x=22 -> c58:5x2 (cols 58, 59)
    # Pattern: col = 64 - (px // 5 * 2)? No.
    # px: 10->62, 16->60, 22->58, 28->56, 34->54
    # This is a linear relationship: col_start = 64 - ((px - 10)//6 * 2 + 2)
    # Let's simplify: the marker moves left as px increases.
    marker_col = 64 - ((px - 10) // 6 * 2 + 2)
    if 0 <= marker_col < 63:
        new_grid[63, :] = 0 # Clear bottom row or set to background
        new_grid[63, marker_col : marker_col + 2] = 5

    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing blocks.
    # Without a WIN STATE grid provided, we assume it's complete if 
    # the marker reaches a certain position or all targets are hit.
    # Based on typical ARC patterns, let's check for a specific condition.
    # Since no win state was given, return False unless a known pattern is met.
    return False