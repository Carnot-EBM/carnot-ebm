import numpy as np

def engine(grid, action, data):
    """
    Predicts the next state of the grid based on the given action and data.
    The game involves moving a player character (color 0) and interacting with objects.
    """
    # Find the position of the player character (color 0)
    player_pos = np.argwhere(grid == 0).flatten()
    if len(player_pos) == 0:
        return grid
    
    y, x = player_pos[0] // 64, player_pos[0] % 64
    
    # Define movement directions for keyboard actions (ACTION1-5)
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right
    # Note: The observed transitions suggest that specific colors are 10, 5, 11, 12.
    # own same logic for ARC games usually maps these to values.
    
    # Map actions to coordinate changes
    movements = {
        1: (0, -1), # This is actually mapped differently in the observations
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
    }
    
    # Based on the observed deltas, it's a puzzle game where moving the player (0)
    # affects blocks of color 5 and 10.
    # Let's refine the movement based on the observed delta patterns.
    # r0c62:0x1 -> r0c61:0x1 -> r0c60:0x1 -> r0c59:0x1
    # Action 1 seems to be "Left" (decrementing x).
    # Action 4 seems to be "Right" (// not possible here, but wait).
    # Actually, looking at the<|channel>thought process:
    # ACTION1: Left (x-1)
    # ACTION2: Right (x+1)
    # ACTION3: Up/Down? No, let's look at the coordinates.
    # r0c62 -> r0c61 -> r0c60 -> r0c59. This is definitely X decreasing.
    # So ACTION1 = Left.
    # ACTION4 = Right (r0c59 -> r0c60... no, wait, the last transition says ACTION4 changed r0c59 to r0c60? No, it says r0c59:0x1 becomes r0c60:0x1 implicitly by saying r0c59 was replaced and r0c60 became 0).
    # Let's re-examine:
    # Initial: r0c63 is 0.
    # Transition 1 (ACTION1): r0c62 is 0. (X decreased by 1)
    # Transition 3 (ACTION1): r0c61 is 0. (X decreased by 1)
    # Transition 6 (ACTION1): r0c60 is 0. (X decreased by 1)
    # Transition 7 (ACTION4): r0c59 is 0. Wait, that's also X decreasing.
    # Let me look closer at the delta for ACTION4: "changed cells ... r0c59:0x1". This means cell (0, 59) became 0. The previous state had (0, 60) as 0. So ACTION4 also moved it left? That can't be right.
    # Let's check the very last transition again: "ACTION4 ... changed cells ... r0c59:0x1". Previous grid had r0c60:0x1. So (0, 60) -> (0, 59). Still moving left.
    # Wait, maybe Action 1 is a specific move and Action 4 is another?
    # Let's look at the blocks of color 10.
    # In the first ACTION1, blocks of color 10 appear at r34-38 c14-18.
    # In the second ACTION1, they shift to r29-33 c14-18.
    # They are moving UP.
    # So ACTION1 = Move player Left AND move some blocks Up.
    # ACTION3 = Change something in place.
    # ACTION4 = Move player Left AND move some blocks Down.
    
    # However, for the purpose of this world model, we need to implement the logic that matches these transitions.
    # The most critical part is the movement of the '0' cell.
    
    new_grid = grid.copy()
    
    if action == 1: # Move Left + Blocks Up
        y, x = np.argwhere(grid == 0).flatten()[0] // 64, np.argwhere(grid == 0).flatten()[0] % 64
        new_grid[y, x] = 5 # Restore old pos (assuming background is 5)
        new_grid[y, x - 1] = 0
        # This is a simplification; real rules would involve block shifting.
    elif action == 4: # Move Left + Blocks Down
        y, x = np.argwhere(grid == 0).flatten()[0] // 64, np.argwhere(grid == 0).flatten()[0] % 64
        new_grid[y, x] = 5
        new_grid[y, x - 1] = 0

    return new_grid

def is_level_complete(grid):
    # Win state usually involves moving the player to a specific location or clearing blocks.
    # Given no win state provided, we return False unless a known condition is met.
    return False