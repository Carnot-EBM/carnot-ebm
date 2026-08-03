import numpy as np

def engine(grid, action, data):
    # The provided observed transitions are very sparse and limited.
    # Based on the same pattern of changes in the same areas, it's a grid-based puzzle.
    # We induce that Action 4 moves an object or modifies a specific region.
    # The observed deltas for Action 4 consistently modify rows 32-35 and columns 32-44.
    #<|channel>thought
    # Let's look at the initial grid again.
    # Rows 32-35, Col 32-35 are color 14.
    # Row 63, Col 56 is color 4.
    # Action 4 shifts something to the right by 4 units.
    # Find current position of "the player" (color 0)
    # Find target positions (color 4).
    # In the INITIAL GRID, row 63 col 56 is color 4.
    # After some ACTION 4s, r63c56 becomes 4x1? No, wait.
    # Looking at the delta: r63c56:4x1 means cell (63, 56) becomes value 4. It was already 4.
    # Wait, let's re-read the run length encoding.
    # Initial Grid r63: 7x57, 4x7 -> cells 0-56 are 7, cells 57-63 are 4.
    # Delta r63c56: 4x1 -> cell (63, 56) becomes 4.
    # This looks like a puzzle where you move a block (color 0) and it affects other things.
    
    # Since we have very little data, we implement a simple movement model for the 'player' (color 0).
    # Color 0 is the only one that seems to be moving or acting as a cursor.
    # Let's assume Action 1=Up, 2=Down, 3=Left, 4=Right, 5=Action/Interact.
    
    new_grid = grid.copy()
    
    # Find player position (value 0)
    player_pos = np.where(grid == 0)
    if len(player_pos[0]) == 0:
        return new_grid
    
    py, px = player_pos[0][0], player_pos[1][0]
    
    dy, dx = 0, 0
    if action == 1: dy = -1
    elif action == 2: dy = 1
    elif action == 3: dx = -1
    elif action == 4: dx = 1
    
    if action in [1, 2, 3, 4]:
        ny, nx = py + dy, px + dx
        if 0 <= ny < 64 and 0 <= nx < 64:
            # Simple collision check: can move if target is not color 1 (wall)
            if grid[ny, nx] != 1:
                # Swap player with the cell they move into
                val_at_target = grid[ny, nx]
                new_grid[py, px] = val_at_target
                new_grid[ny, nx] = 0
                
    return new_grid

def is_level_complete(grid):
    # Level complete when all targets are reached or a specific condition is met.
    # In many ARC games, this means removing all of a certain color or reaching a goal.
    # Based on the provided data, we don't have a win state.
    # We'll assume it's complete if there are no more 'targets' (color 4) left in some area.
    # Or simply return False as we haven't seen a win state.
    return False