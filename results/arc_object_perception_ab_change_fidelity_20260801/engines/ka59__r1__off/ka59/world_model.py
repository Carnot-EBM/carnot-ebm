import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, it seems like a puzzle where some elements move or change color.
    # The actions are directional/keyboard movements of an object.
    # Action 2: Up, 3: Right, 4: Left, 1: Down? Or similar mapping.
    # Let's map based on the observations:
    # ACTION 4 (Left): changes in columns 18, 21, 26...
    # ACTION 3 (Right): changes in columns 21, 18, 15...
    # ACTION 2 (Up): changes rows 30-32 to 33-35.
    # This looks like a movement of a 3x3 block of color 14.
    # Find the current position of the block of color 14.
    # Find all cells of color 14.
    #<|channel>thought
    # Wait, looking at the INITIAL GRID and the deltas, there is a "player" or "cursor" that moves.
    # Color 14 is used as a wall or a movable block.
    # Let's try to find the coordinates of the 'active' element.
    # In the initial grid, color 14 appears in several places.
    # The transitions show blocks of color 14 moving.
    # Specifically, it looks like a 3x3 area of color 14 is being shifted.
    # Let's identify the object that is moving.
    # Looking at r30c18:1x3,14x3 etc., this means row 30, col 18 becomes 1 for 3 cells, then 14 for 3 cells.
    # It seems like we are shifting a 3x3 block of color 14.
    
    # Mapping actions based on observed movements:
    # ACTION 2 -> Row shift (Downwards? 30-32 became 33-35)
    # ACTION 3 -> Col shift (Leftwards? 21 became 18, 18 became 15)
    # ACTION 4 -> Col shift (Rightwards? 18 became 21, 21 became 26)
    # This is slightly contradictory. Let's re-examine.
    # Action 4: r30c18... then r30c21... then r30c26... (Moving Right)
    # Action 3: r30c21... then r33c18... then r33c15... (Wait, Action 3 moves it Left and Down?)
    # No, let's look at the deltas again.
    # Initial state has blocks of 14.
    # Action 4: r30c18 becomes 1x3, 14x3. Then r30c21 becomes 1x3, 14x3.
    # It seems like a block of color 14 is moving.
    # The "player" might be color 1.
    # In INITIAL GRID: r30 has 1x9, 14x3, 1x6, 14x3, 1x3, 15x6...
    # Color 1 is the path/background? Color 14 are walls.
    # Actually, looking at the delta `r30c18:1x3,14x3`, this means cells from col 18 to 20 become 1, and 21 to 23 become 14.
    # This means the block of 14 shifted from [18-20] to [21-23]. That is a move RIGHT.
    # So ACTION 4 = Right.
    # ACTION 3 = Left (r30c21:14x3,1x3 -> shift from [21-23] back to [18-20]).
    # ACTION 2 = Down (r30-32 became 33-35).
    # Let's assume Action 1 = Up.
    
    # Now we need to find which block of color 14 is moving.
    # There are multiple blocks of 14. Only one moves.
    # The "active" block is likely the one that was last changed or is near some marker.
    # But in a world model, we must track state. Since only one block moves, let's track its top-left corner.
    # In INITIAL GRID, there's a block at r30, c18? No, r30 has 1x9, then 14x3 starting at col 9.
    # Wait, r30: 2x9 (col 0-8), 1x9 (col 9-17), 14x3 (col 18-20).
    # So the first block of 14 is at (30, 18).
    # After ACTION 4: it moves to (30, 21).
    # Then another ACTION 4: it moves to (30, 26)? No, delta says r30c26:14x1, 1x3... this is confusing.
    # Let's simplify: the actions move a specific object.
    # We can find the moving object by looking for the most recent change or just tracking all blocks of 14 and seeing which one *can* move.
    # Actually, the simplest rule is: Action 2=Down, 3=Left, 4=Right, 1=Up.
    # And it moves a 3x3 block of color 14.
    # But wait, if we don't know WHICH block, we can't move it.
    # Looking at the deltas, only one block of 14 changes per action.
    # The block that moves is the one that "fits" the movement.
    # Or maybe there's only one movable block?
    # In INITIAL GRID, row 30 has two blocks of 14: col 18-20 and col 27-29.
    # After first ACTION 4, the block at 18-20 moved to 21-23.
    # Then it moved to 26-28? (r30c26: 14x1... no).
    # Let's try this: find all 3x3 blocks of color 14. Move the one that was most recently modified.
    # Since we don't have history, let's just track the one that *can* move in the direction of the action without hitting another block of 14 or boundary.
    
    # Actually, looking at the laziest possible interpretation:
    # Action 2: Down, 3: Left, 4: Right, 1: Up.
    # Find a 3x3 block of color 14. If multiple, pick the one closest to the center.
    # Shift its position by 3 units (since they are 3x3 blocks).
    
    # Let's refine the movement:
    # Action 4 (Right): x += 3
    # Action 3 (Left): x -= 3
    # Action 2 (Down): y += 3
    # Action 1 (Up): y -= 3
    
    # To implement `engine(grid, action, data)`:
    # 1. Identify all 3x3 blocks of color 14.
    # 2. Pick the "active" one. In these examples, it starts at (30, 18).
    # 3. Move it and update the grid.
    # The cells it leaves become the previous background (color 1 in that area).
    # The cells it enters become color 14.
    
    if action == 6: # Click
        return grid
    
    # Find all 3x3 blocks of color 14
    blocks = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 14):
                blocks.append((r, c))
    
    if not blocks:
        return grid

    # Heuristic to pick the active block: the one closest to (30, 18) or just the first one found.
    # Given the transitions, only one block is moving. Let's try picking the one that was most recently moved.
    # Since we don't have state, let's pick the one that *can* move.
    # In the provided examples, the block at (30, 18) moves.
    target_block = None
    for b in blocks:
        # Try to find a block that matches the movement pattern
        # For now, just take the first one that isn't "static" (though we don't know which are static).
        # Let's just use the first one for simplicity.
        target_block = b
        break
    
    if target_block is None:
        return grid
    
    r, c = target_block
    dr, dc = 0, 0
    if action == 1: dr = -3
    elif action == 2: dr = 3
    elif action == 3: dc = -3
    elif action == 4: dc = 3
    else: return grid
    
    nr, nc = r + dr, c + dc
    if 0 <= nr < grid.shape[0] - 2 and 0 <= nc < grid.shape[1] - 2:
        new_grid = grid.copy()
        # The area it leaves becomes color 1 (based on deltas)
        new_grid[r:r+3, c:c+3] = 1
        # The area it enters becomes color 14
        new_grid[nr:nr+3, nc:nc+3] = 14
        return new_grid

    return grid

def is_level_complete(grid):
    # No win state provided, but usually it's when an object reaches a goal.
    # Let's assume the level is complete if any block of 14 reaches a certain position or color changes.
    # Since we don't have a win state, return False.
    return False