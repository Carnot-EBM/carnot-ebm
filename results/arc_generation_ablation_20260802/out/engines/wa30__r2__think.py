import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the action and observed transitions.
    The game 'wa30' appears to be a puzzle involving moving or modifying colors of blocks.
    Based on the observations, it seems that some specific regions are modified by actions.
    """
    # The provided transitions are very limited and sparse.
    # Wethought they look like patterns of changes in<|channel>thought
    # However, looking at thes same sequence of events:
    # Action 4 (Right/Down?) moves something across columns 32, 36, 40, 44.
    # Action 1 (Up/Left?) shifts blocks of color 14 (yellowish) upwards.
    # Action 3 (Down/Right?) modifies cells in rows 24-27.
    # Action 5 (Left/Up?) resets or clears certain areas.
    
    # Since we don'tt know the full rules, we will implement a basic logic 
    # that mimics the observed delta changes if possible, but since the 
    # deltas are relative to current positions, we can actually see a 
    # 0x4 block being moved.
    
    # To make this an executable world model, we need to identify the "player" object.
    # Color 0 is often used as background or empty space.
    # Color 1 is the main filler.
    # Colors 4, 9, 14, 2, 7 are objects.
    
    # Let's assume ACTION 4 is 'Move Right'.
    # Let's assume ACTION 1 is 'Move Up'.
    # Let's assume ACTION 3 is 'Move Down'.
    # Let's assume ACTION 5 is 'Move Left'.
    
    # The transitions show blocks of color 14 moving and changing colors.
    # Based on the observations:
    # r32c32:1x4,14x3,0x1 -> This looks like a sliding window of values.
    # Action 4 moves something from col 32 to 36 to 40 to 44.
    # Action 1 shifts things up (r32-35 becomes r28-31 then r24-27).
    # Action 3 modifies rows 24-27.
    
    # Given the extreme sparsity and specific coordinates, it's likely a grid puzzle.
    # We will implement a simplified version where we track an object's position.
    
    new_grid = grid.copy()
    
    # Find the "player" or "active block". In this case, let's look for color 0.
    # Color 0 seems to be the gap/hole that moves.
    holes = np.argwhere(grid == 0)
    if holes.size == 0:
        return new_grid

    # Let's assume there is one primary hole at (y, x)
    y, x = holes[0]
    
    # Based on observed transitions:
    # ACTION 4: Move Right (x increases by 4)
    # ACTION 1: Move Up (y decreases by 4)
    # ACTION 3: Move Down (y increases by 4)
    # ACTION 5: Move Left (x decreases by 4)
    
    # This matches the coordinate shifts in the deltas (c32 -> c36 -> c40 -> c44).
    
    if action == 4: # Right
        nx, ny = x + 4, y
    elif action == 1: # Up
        nx, ny = x, y - 4
    elif action == 3: # Down
        nx, ny = x + 4, y # Wait, Action 3 delta was r24-27... let's re-evaluate.
    elif action == 5: # Left
        nx, ny = x - 4, y
    else:
        return new_grid

    # The actual logic seems to be swapping the hole with a block of colors.
    # We need to handle the "block" movement carefully.
    # In ARC games, usually you move an object or a gap.
    # Let's try to implement a simple swap for the hole and its neighbor.
    
    # However, the observed transitions are very specific about blocks of size 4x4.
    # Let's refine:
    if action == 4: # Move Right
        # Shift a 4x4 area? No, the deltas show r32c32:1x4, 14x3, 0x1.
        # This means at (32, 32), it becomes color 1 for 4 cells, then 14 for 3, then 0 for 1.
        # Original state r32: 1x32, 14x3, 0x1, 1x28.
        # So col 32-34 is 14, col 35 is 0.
        # After ACTION 4: col 32-35 is 1, col 36-38 is 14, col 39 is 0.
        # The "hole" (color 0) moved from x=35 to x=39.
        dx, dy = 4, 0
    elif action == 1: # Move Up
        # Hole was at y=35, x=47 (approx). Moved to y=31, x=47 then y=27, x=47.
        dx, dy = 0, -4
    elif action == 3: # Move Down
        dx, dy = 0, 4
    elif action == 5: # Move Left
        dx, dy = -4, 0
    else:
        return new_grid

    # To implement this movement:
    # Find the hole (0), move it by (dy, dx), and shift colors in between.
    # Since we don't have a full map of blocks, we will just swap the hole with the target cell.
    # But the deltas show a block of color 14 moving.
    
    # Let's find all holes.
    holes = np.argwhere(grid == 0)
    for h_y, h_x in holes:
        target_y, target_x = h_y + dy, h_x + dx
        if 0 <= target_y < 64 and 0 <= target_x < 64:
            # Swap hole with target
            new_grid[h_y, h_x] = grid[target_y, target_x]
            new_grid[target_y, target_x] = 0
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a specific configuration or clearing objects.
    # In the absence of a WIN STATE example, we check if any object reached the bottom-right.
    # The last delta shows r63c56:4x1 (color 4 at row 63).
    # This suggests that getting color 4 to the bottom row might be the goal.
    return np.any(grid[63, :] == 4)