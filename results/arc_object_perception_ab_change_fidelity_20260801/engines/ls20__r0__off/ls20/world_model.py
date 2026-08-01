import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right, ACTION6: Click
    
    # Identify current position of the "player" or active object.
    # The player seems to be represented by color 3 at rows 61-62 and columns 14-20 in the observations.
    # Let's find all cells of color 3 in that region.
    # Initial state has color 3 at r61c14 and r62c14.
    # After ACTION3 (Left), it moves to c15? No, wait.
    # Wait, looking at the same transition: r61c14:3x1, r62c14:3x1 becomes r61c15:3x1, r62c15:3x1.
    # That is moving RIGHT. But ACTION3 is usually LEFT.
    # Let's re-examine.
    # Transition 1: ACTION3 -> r61c14:3x1, r62c14:3x1 changes to r61c15:3x1, r62c15:3x1.
    # Transition 2: ACTION3 -> r61c15:3x1, r62c15:3x1 changes to r61c16:3x1, r62c16:3x1. (Wait, this is not what happened).
    # Let's look at the transitions again carefully.
    # Transition 1: ACTION3 (0->0) changed cells = r45c24:12x5,3x5 ... r61c14:3x1 r62c14:3x1.
    # This means color 3 was at c14 and now it's NOT there.
    # The new state has color 3 at some other place? No, the delta only shows CHANGED cells.
    # If a cell becomes color 3, it's in the delta. If it leaves, it's in the delta.
    # In "r61c14:3x1", it means column 14 of row 61 became value 3.
    # Wait, if the initial grid had color 3 at r61c14, then "r61c14:3x1" would mean it STAYED color 3 or became color 3.
    # Actually, usually ARC deltas show NEW values.
    # Initial Grid: r61: ..., 3x1, 11x41, ... -> col 14 is 3.
    # Delta 1: r61c14:3x1. That means it stays 3? Or maybe the player is not 3.
    # Let's look at ACTION1 (Up): r61c16:3x1, r62c16:3x1. This means the player moved to c16.
    # Let's re-read: "ACTION3 (level 0->0): changed cells = ... r61c14:3x1 r62c14:3x1".
    # If current was at c15 and moves to c14, delta shows r61c14:3x1 and r61c15:<something else>.
    # Looking at the sequence:
    # Init: c14 is 3.
    # Action 3: r61c14:3x1... wait, if it was already 3, this delta is weird.
    # Unless the player is NOT color 3.
    # Let's look at the colors in that area: 3, 11, 5, 8.
    # Initial Grid r61: ..., 3x1, 11x41, ... -> col 14 is 3.
    # Delta 1 (Action 3): r61c14:3x1. This is very strange.
    # Wait! The deltas are NEW values.
    # In Transition 1: ACTION3 -> r61c14:3x1. It means cell (61, 14) becomes 3.
    # But it was already 3.
    # Let's look at all "r61cx" in deltas:
    # T1 (A3): r61c14:3x1
    # T2 (A3): r61c15:3x1
    # T3 (A1): r61c16:3x1
    # T4 (A1): r61c17:3x1
    # T5 (A1): r57c3:5x2... no r61 here.
    # T6 (A1): r61c18:3x1
    # T7 (A4): r61c19:3x1
    # T8 (A4): r61c20:3x1
    # This looks like a player moving across columns 14, 15, 16, 17, 18, 19, 20.
    # And the actions were A3, A3, A1, A1, A1, A1, A4, A4.
    # This is not standard movement.
    # However, there's another pattern: The "blocks" of colors are shifting.
    # Look at rows 40-49 and cols 19-24.
    # In ACTION1 (Up), blocks move from r45->r40, then r40->r35, etc.
    # It seems action moves a block of pixels.
    # Let's simplify: Action 1 = Up, Action 2 = Down, Action 3 = Left, Action 4 = Right.
    # We need to find what object is moving.
    # Looking at the deltas, it's a 5-column wide block (e.g., c19:12x5) that shifts vertically or horizontally.
    # For Action 1 (Up): The block at r45c19 (size 5x5 approx) moves up by 5 rows each time.
    # Transition T3: Block at r45c19 moves to r40c19.
    # Transition T4: Block at r40c19 moves to r35c19.
    # Transition T5: Block at r35c19 moves to r30c19.
    # Transition T6: Block at r30c19 moves to r25c19.
    # This is exactly "Move block UP by 5 units".
    # Similarly for ACTION4 (Right):
    # Transition T7: Block at r25c19 moves to r25c24.
    # Transition T8: Block at r25c24 moves to r25c29? No, delta says r25c24:3x5,12x5.
    # Wait, the blocks are shifting.
    # Let's implement a simple shift for the identified moving blocks.
    
    new_grid = grid.copy()
    
    # Find the current position of the moving block (color 12 or 9)
    # The block seems to be in rows 25-49 and cols 19-28.
    # Let's find any cell with color 12 or 9 in that region.
    coords = np.argwhere((grid >= 9) & (grid <= 12))
    if coords.size == 0:
        return new_grid
    
    # Get the bounding box of this object
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    # Object dimensions
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    # Determine movement based on action
    dr, dc = 0, 0
    if action == 1: dr = -5 # Up
    elif action == 2: dr = 5  # Down
    elif action == 3: dc = -5 # Left
    elif action == 4: dc = 5  # Right
    
    # Apply movement
    new_min_r, new_min_c = min_r + dr, min_c + dc
    new_max_r, new_max_c = max_r + dr, max_c + dc
    
    # Boundary check
    if new_min_r < 0 or new_max_r >= grid.shape[0] or \
       new_min_c < 0 or new_max_c >= grid.shape[1]:
        return grid # Block same position if out of bounds
    
    # Clear old position and set new position
    # We only move the pixels that are not color 4 (the background)
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r, c] != 4:
                new_grid[r, c] = 4
                new_grid[r + dr, c + dc] = grid[r, c]
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing something.
    # In this game, there's no win state provided, but typically it's when the object reaches a certain area.
    # Let's assume complete if the block moves to a specific region or matches a pattern.
    # Based on common ARC patterns, let's check if any part of the block is in the top-left or similar.
    coords = np.argwhere((grid >= 9) & (grid <= 12))
    if coords.size == 0: return False
    return coords.min(axis=0)[0] < 5 # Example condition: reached the top