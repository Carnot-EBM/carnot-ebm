import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle where an object (color 9 or similar)
    # moves across a set of predefined slots/paths.
    # The "object" seems to be a 3x3 block of color 9.
    # Action 2: Down
    # Action 3: Left
    # Action 4: Right
    # Action 1: Up
    # Action 5: Not seen but likely Up/Down/Left/Right
    # Let's identify the current position of the 3x3 block of color 9.
    
    # Find the center of the 3x3 block of color 9
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return grid.copy()
    
    # We assume it's a 3x3 block. If there are multiple blocks, we take the average.
    center_y, center_x = np.mean(coords, axis=0).astype(int)
    
    # Movement vectors for actions
    # ACTION 1: Up?
    # ACTION 2: Down
    # ACTION 3: Left
    # ACTION 4: Right
    # ACTION 5: ?
    # ACTION 6: Click
    
    move_map = {
        1: ( -1 * 6, 0), # Likely distance is 6 units (since slots are 3+3 wide)
        2: ( 6, 0),
        3: (0, -6),
        4: (0, 6),
    }
    
    if action not in move_map:
        return grid.copy()
    
    dy, dx = move_map[action]
    
    # New potential position
    new_y, new_x = center_y + dy, center_x + dx
    
    # Check if movement is valid (within bounds and boundaries/walls)
    # We assume walls are color 5 (the background) or specific colors that other cells are 0 or 2.
    # a "slot" is a 3x3 area of some color (usually 0 or 2).
    # a "path" consists of these slots.
    # a<|channel>thought
    # Let's refine the movement logic based on the observed deltas.
    # In the first transition: r21c33:0x3 r22c33:0x3 r23c33:0x3 (old pos) -> r27c33:9x3 r28c33:9x3 r29c33:9x3 (new pos)
    # The block moves from (21-23, 33-35) to (27-29, 33-35). Delta Y = 6.
    # Action 2 = Down.
    # Transition 2: (27-29, 33-35) -> (33-35, 33-35). Delta Y = 6.
    # Action 2 = Down.
    # Transition 3: (33-35, 33-35) -> (33-35, 27-29). Delta X = -6.
    # Action 3 = Left.
    # Transition 4: (33-35, 27-29) -> (33-35, 21-23). Delta X = -6.
    # Action 3 = Left.
    # Transition 5: (33-35, 21-23) -> (39-41, 21-23). Delta Y = 6.
    # Action 2 = Down.
    # Transition 6: (39-41, 21-23) -> (39-41, 27-29). Delta X = 6.
    # Action 4 = Right.
    # Transition 7: (39-41, 27-29) -> (39-41, 33-35). Delta X = 6.
    # Action 4 = Right.
    # Transition 8: (39-41, 33-35) -> (45-47, 33-35). Delta Y = 6.
    # Action 2 = Down.
    
    # The block of color 9 moves in steps of 6 units.
    # It can only move into a "slot" that is not blocked by walls (color 5).
    # Let's implement this movement logic.
    
    new_grid = grid.copy()
    
    # Current position of the 3x3 block of color 9
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    y_min, x_min = np.min(coords, axis=0)
    
    # Target top-left corner
    target_y, target_x = y_min + dy, x_min + dx
    
    # Check if the target 3x3 area is "passable".
    # In this game, it seems any cell that isn't color 5 is part of the path.
    # We check if all cells in the target 3x3 are NOT color 5.
    if target_y < 0 or target_x < 0 or target_y + 3 > 64 or target_x + 3 > 64:
        return new_grid
    
    # The observed deltas also show a change at r63cXX. This looks like a progress bar or counter.
    # Let's track how many moves were made.
    # Transition 1: r63c55:0x1 (was something else?)
    # Initial grid r63: 6x56, 0x8. So c56..63 are 0.
    # After Action 2: r63c55 becomes 0? No, wait.
    # Initial r63: col 0-55=6, col 56-63=0.
    # Trans 1: r63c55:0x1 -> col 55 becomes 0. Now 0-54=6, 55-63=0.
    # Trans 2: r63c54:0x1 -> col 54 becomes 0. Now 0-53=6, 54-63=0.
    # Trans 3: r63c52:0x2 -> col 52,53 become 0. Now 0-51=6, 52-63=0.
    # This is a countdown/progress bar at the bottom row.
    
    # Check if target area contains any walls (color 5)
    target_area = grid[target_y : target_y + 3, target_x : target_x + 3]
    if np.any(target_area == 5):
        return new_grid
    
    # Move the block
    new_grid[y_min : y_min + 3, x_min : x_min + 3] = grid[y_min : y_min + 3, x_min : x_min + 3].copy() # wait, this is wrong
    # We need to restore the original colors of the slot we leave.
    # But since we don't know what was there, and it seems slots are just placeholders...
    # Let's look at the deltas again.
    # Transition 1: r21c33:0x3 ... (old pos becomes 0). New pos becomes 9.
    # So old cells become 0? No, let's check initial grid.
    # Initial r21c33: 9x3. After Action 2: r21c33:0x3.
    # So color 9 moves, and the previous position becomes color 0.
    
    # Correct movement logic:
    # 1. Identify current 3x3 block of 9s.
    # 2. Clear current 3x3 block (set to 0).
    # 3. Set target 3x3 block to 9.
    # 4. Update progress bar at row 63.
    
    new_grid[y_min : y_min + 3, x_min : x_min + 3] = 0
    new_grid[target_y : target_y + 3, target_x : target_x + 3] = 9
    
    # Progress bar update: find the rightmost cell in row 63 that is color 6 and set it to 0.
    row_63 = new_grid[63, :]
    idx = np.where(row_63 == 6)[0]
    if idx.size > 0:
        last_idx = idx[-1]
        new_grid[63, last_idx] = 0
        
    return new_grid

def is_level_complete(grid):
    # Level complete when the object reaches a certain position or the progress bar is empty.
    # In observed data, we don't see a win state, but usually it's reaching the goal.
    # Let's assume it's when the block of 9s reaches the final slot (45-47, 33-35).
    coords = np.argwhere(grid == 9)
    if coords.size == 0: return False
    y_min, x_min = np.min(coords, axis=0)
    return y_min == 45 and x_min == 33