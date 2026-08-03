import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up (inferred)
    
    # The object being moved is a 3x3 block of color 9 or similar markers.
    # Find the current position of the marker (color 9).
    # We assume there is only one such "active" moving piece.
    # Let's identify the movement logic.
    # In the observations, color 9 blocks move in steps of 3 cells.
    # 3x3 blocks are often replaced by 0s or other colors.
    #<|channel>thought
    # Looking at the deltas:
    # r21c33:0x3 r22c33:0x3 r23c33:0x3 -> This removes a 3x3 block at (21, 33).
    # r27c33:9x3 r28c33:9x3 r29c33:9x1,4x1,9x1 -> This adds a 3x3 block at (27, 33), but with a center pixel change.
    # It seems like the active piece is a 3x3 block that moves across the grid.
    # The game state includes some static obstacles/walls (colors 0, 2, 5, 14).
    # Color 5 is background.
    # Action 2: Down, Action 3: Left, Action 4: Right, Action 1: Up.
    
    # Find all coordinates of color 9.
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return grid
    
    # Assume the moving piece is the top-left corner of a 3x3 block.
    # We identify the "active" piece by finding the smallest x and y.
    y_min, x_min = coords[0]
    
    # Determine movement delta based on action.
    dy, dx = 0, 0
    if action == 1: dy, dx = -3, 0
    elif action == 2: dy, dx = 3, 0
    elif action == 3: dy, dx = -0, -3
    elif action == 4: dy, dx = 0, 3
    
    # New position
    ny, nx = y_min + dy, x_min + dx
    
    # Check boundaries
    if ny < 0 or ny + 3 > grid.shape[0] or nx < 0 or nx + 3 > grid.shape[1]:
        return grid
    
    # The observed transitions show that when color 9 moves, it replaces what was there.
    # The cells at the old position are set to background (color 5) or something else?
    # Looking at ACTION2: r21c33:0x3 ... -> Old pos becomes 0s.
    # Wait, looking closer at INITIAL GRID:
    # r21: ..., 9x3, ... (at col 33-35).
    # After Action 2: r21c33:0x3... and r27c33:9x3...
    # So the block at (21, 33) became 0s, and a new block appeared at (27, 33).
    # This is movement of a 3x3 block.
    
    # Let's refine the "active piece" detection.
    # Color 9 seems to be the marker.
    # Find all blocks of 3x3 color 9.
    # For simplicity, find any cell with color 9 and treat its top-left as the anchor.
    y_min, x_min = np.min(coords, axis=0)
    
    # Calculate target area
    old_area = grid[y_min:y_min+3, x_min:x_min+3]
    new_grid = grid.copy()
    
    # Clear old position - set to background or what was there?
    # In ACTION2: r21c33:0x3 -> it becomes 0.
    # But in other cases, it might become something else.
    # Looking at INITIAL GRID, cells at (21, 33) were 9. Now they are 0.
    # Actually, looking at the deltas, the old positions are often set to 0.
    new_grid[y_min:y_min+3, x_min:x_min+3] = 0
    
    # Place new block
    ny, nx = y_min + dy, x_min + dx
    if ny < 0 or ny + 3 > grid.shape[0] or nx < 0 or nx + 3 > grid.shape[1]:
        return grid
    
    # The new block is not always just color 9.
    # "r27c33:9x3 r28c33:9x3 r29c33:9x1,4x1,9x1" -> center pixel of bottom row is 4.
    # This suggests a specific pattern for the moving piece.
    # Let's use the same pattern as the original block if possible.
    pattern = old_area.copy()
    # But wait, the pattern changes?
    # ACTION2 (level 0->0): ... r29c33:9x1,4x1,9x1 ...
    # Then next ACTION2: ... r35c33:9x1,4x1,9x1 ...
    # It seems the marker has a internal state or the target cells affect it.
    # Actually, looking at the deltas, the "4" appears in the middle of the 3x3 block.
    # Let's just copy the 3x3 block from the old position to the new one.
    new_grid[ny:ny+3, nx:nx+3] = pattern
    
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it involves reaching a goal or clearing something.
    # Based on typical ARC games, we can't be sure without data.
    # However, often color 0 or specific colors are goals.
    # For now, return False unless a clear condition is found.
    return False