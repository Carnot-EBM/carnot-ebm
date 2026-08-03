import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where certain colors are used as walls or background.
    # Based on the observed transitions, it's a a movement/collection puzzle.
    # Action 4 is likely 'Right', Action 2 is 'Down', Action 1 is 'Left'.
    # Let's assume standard directional movements for an object.
    # a simple model based on the<|channel>thought process of the same thought trace.
    
    # In this specific dataset, we see changes in cells that are color 0 and 7.
    # Color 0 might be a target or empty space.
    # Color 7 is often replaced by 5 or vice versa.
    # 
    # Looking at the ACTION4 delta:
    # r48c15:3x5, r48c22:0x5... This looks like a shift of a block of size 5.
    # It replaces 0 with 3 and 3 with 0.
    # 
    # However, the provided observations are very sparse.
    # Let's implement a basic engine that handles these shifts if they can be identified.
    
    # The most consistent pattern is that action 4 (right) moves something to the right,
    # action 1 (left) moves it to the left, and action 2 (down) moves it down.
    # Action 6 is click.
    
    # Based on the observed transitions, there is a "player" or "cursor" moving.
    # The cell r63c63 was initially 4. Then it became r63c62, then r63c61, etc.
    # This suggests color 4 is the player.
    
    new_grid = grid.copy()
    
    # Find player position
    player_pos = np.where(grid == 4)
    if len(player_pos[0]) == 0:
        return new_grid
    
    py, px = player_pos[0][0], player_pos[1][0]
    
    # Map actions to movement
    # ACTION 1: Left, ACTION 2: Down, ACTION 3: Up, ACTION 4: Right
    movements = {
        1: (0, -1),
        2: (1, 0),
        3: (-1, 0),
        4: (0, 1)
    }
    
    if action in movements:
        dy, dx = movements[action]
        ny, nx = py + dy, px + dx
        
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            # Check for walls (color 2 is likely a wall based on INITIAL GRID)
            if grid[ny, nx] != 2:
                # Move player
                new_grid[py, px] = grid[py, px] if grid[py, px] != 4 else 0 # This is tricky
                # We need to know what was at the target position.
                # In this game, color 0 might be background.
                # Let's assume the player swaps positions with whatever is there.
                
                # Based on observed deltas, when the player moves, other things move too.
                # Action 4 shifts blocks of colors 0 and 3.
                # The block shift happens at rows 48-60.
                # When ACTION 4 occurs, r48c15 becomes 3x5, r48c22 becomes 0x5.
                # This is a shift of a 5-wide block from col 22 to col 15? No, that's left.
                # Wait, r48c15:3x5 means cells [15, 19] become 3. r48c22:0x5 means [22, 26] become 0.
                # That looks like a movement of a block of size 5 from x=22 to x=15.
                # But action 4 is usually 'Right'.
                
                # Let's re-examine:
                # Initial: r48c15:3x5 (Wait, INITIAL GRID says r48:3x15, 0x5, 3x44)
                # So initially cols 15-19 are color 0.
                # After ACTION 4: r48c15:3x5 and r48c22:0x5.
                # Now cols 15-19 are color 3 and cols 22-26 are color 0.
                # The block of color 3 moved from [22, 26] to [15, 19].
                # This is a LEFT move. So Action 4 = Left?
                # Let's check ACTION 1: "changed cells ... r52c30:7x4".
                # This doesn't look like the same block shift.
                
                # Given the complexity and sparsity, let's implement the player movement
                # as the primary mechanism and assume it triggers these shifts.
                
                new_grid[py, px] = grid[ny, nx] if grid[ny, nx] != 4 else 0 # Simplified
                new_grid[ny, nx] = 4
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves collecting something or reaching a goal.
    # In many ARC games, this means all targets (color X) are gone or in place.
    # Without a win state example, we return False unless a specific condition is met.
    return False