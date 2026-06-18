import numpy as np

def engine(grid, action, data):
    """
    Simulates the game logic for 'ka59'.
    
    Logic Induction:
    1. The grid contains a 'Player' (color 14) and 'Blocks' (color 1).
    2. The 'Player' moves in 3x3 blocks. When the player moves, the 3x3 area
       they occupy toggles between state 1 (Block) and 14 (Player).
       Specifically, the center of the 3x3 becomes 0 (Empty) if it was 1,
       and the surrounding 8 cells toggle 1<->14.
       Actually, looking at the deltas:
       - Action 1 (Up): The 3x3 block at (r, c) moves to (r-3, c).
         The cells at (r, c) become 1 (Block) or 0 (Empty).
         The cells at (r-3, c) become 14 (Player) or 0 (Empty).
         The pattern is a 3x3 toggle.
       - Action 2 (Down): The 3x3 block at (r, c) moves to (r+3, c).
    3. There is a 'Score' or 'Counter' at the bottom right (63, 63) and (63, 62) etc.
       It counts down from 4 to 0.
    4. The game ends when the counter reaches 0? Or when the grid is full of blocks?
       The win state is likely when all blocks are collected (turned to 0 or 14?).
       Actually, the counter goes to 0.
    
    Let's refine the movement rule:
    The player is a 3x3 entity.
    Action 1 (Up): Move the 3x3 entity up by 3 rows.
    Action 2 (Down): Move the 3x3 entity down by 3 rows.
    Action 3 (Left): Move the 3x3 entity left by 3 cols.
    Action 4 (Right): Move the 3x3 entity right by 3 cols.
    
    The 3x3 entity consists of:
    - Center: 0 (Empty)
    - Surrounding 8 cells: 14 (Player)
    
    When moving, the old position's 3x3 area reverts to 1 (Block) or 0 (Empty) based on what was there before?
    No, the deltas show:
    - Old position cells (1) become 14 (Player) in the new position? No.
    - Let's look at Action 1 (Up) from row 27 to 24.
      Old rows 27-29, Cols 18-20.
      New rows 24-26, Cols 18-20.
      
      Deltas for Action 1 (Up):
      (24, 18, 1, 14) -> (24, 19, 1, 14) -> (24, 20, 1, 14)
      (25, 18, 1, 14) -> (25, 19, 1, 0)  -> (25, 20, 1, 14)
      (26, 18, 1, 14) -> (26, 19, 1, 14) -> (26, 20, 1, 14)
      
      (27, 18, 14, 1) -> (27, 19, 14, 1) -> (27, 20, 14, 1)
      (28, 18, 14, 1) -> (28, 19, 14, 0) -> (28, 20, 14, 1)
      (29, 18, 14, 1) -> (29, 19, 14, 1) -> (29, 20, 14, 1)
      
      It seems the 3x3 block at (24, 18) becomes the Player (14) with center 0.
      The 3x3 block at (27, 18) becomes the Block (1) with center 0? No, center is 0 in both.
      Wait, (25, 19) became 0. (28, 19) became 0.
      So the center of the 3x3 is always 0.
      The surrounding 8 cells toggle between 1 and 14.
      
      Rule:
      1. Identify the current 3x3 Player block. It is a 3x3 area where the center is 0 and the rest are 14.
      2. Determine the target 3x3 area based on the action.
      3. For the target area:
         - Set the center to 0.
         - Set the surrounding 8 cells to 14.
      4. For the old area:
         - Set the center to 0.
         - Set the surrounding 8 cells to 1.
      5. Decrement the counter at (63, 63) if it is > 0.
    
    Counter Logic:
    - The counter is at (63, 63) and (63, 62) etc.
    - It starts at 4 and goes to 0.
    - It seems to decrement by 1 for each move?
    - Or maybe it's a score of collected blocks?
    - The counter cells are 4 (Yellow).
    - When the counter reaches 0, the level might be complete.
    
    Let's implement the 3x3 toggle and counter decrement.
    """
    grid = grid.copy()
    
    # Find the player block (3x3 area with center 0 and rest 14)
    # We can scan for 0s and check the surrounding 8 cells.
    player_r, player_c = -1, -1
    H, W = grid.shape
    
    # Find the center of the player block
    # The player block is 3x3. Center is (r, c).
    # We look for a 0 that is surrounded by 14s.
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if grid[r, c] == 0:
                # Check if it's a player block
                is_player = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        if grid[r + dr, c + dc] != 14:
                            is_player = False
                            break
                    if not is_player:
                        break
                if is_player:
                    player_r, player_c = r, c
                    break
        if player_r != -1:
            break
            
    if player_r == -1:
        # No player found, maybe game over or initial state?
        # If no player, just decrement counter if present.
        pass
    else:
        # Determine the move direction
        # Action 1: Up, 2: Down, 3: Left, 4: Right
        # We need to map action to delta.
        # Based on the data:
        # Action 1: Up (row decreases)
        # Action 2: Down (row increases)
        # Action 3: Left (col decreases)
        # Action 4: Right (col increases)
        
        dr, dc = 0, 0
        if action == 1:
            dr = -3
        elif action == 2:
            dr = 3
        elif action == 3:
            dc = -3
        elif action == 4:
            dc = 3
        else:
            # Other actions (5, 6, 7) might be clicks or special.
            # For now, assume no movement.
            pass
            
        if dr != 0 or dc != 0:
            # Calculate new center
            new_r, new_c = player_r + dr, player_c + dc
            
            # Check bounds
            if 0 <= new_r < H and 0 <= new_c < W:
                # Apply the move
                # Old position: revert to 1s
                for r in range(player_r - 1, player_r + 2):
                    for c in range(player_c - 1, player_c + 2):
                        if 0 <= r < H and 0 <= c < W:
                            if r == player_r and c == player_c:
                                grid[r, c] = 0
                            else:
                                grid[r, c] = 1
                
                # New position: set to 14s
                for r in range(new_r - 1, new_r + 2):
                    for c in range(new_c - 1, new_c + 2):
                        if 0 <= r < H and 0 <= c < W:
                            if r == new_r and c == new_c:
                                grid[r, c] = 0
                            else:
                                grid[r, c] = 14

    # Decrement counter
    # The counter is at the bottom right.
    # It seems to be a single cell or a few cells.
    # From the data, (63, 63) goes from 4 to 0.
    # Let's find the counter cell.
    # It's likely the cell with value 4 at the bottom right.
    # We can scan for 4s.
    counter_cells = []
    for r in range(H - 5, H):
        for c in range(W - 5, W):
            if grid[r, c] == 4:
                counter_cells.append((r, c))
                
    if counter_cells:
        # Decrement all counter cells
        for r, c in counter_cells:
            if grid[r, c] > 0:
                grid[r, c] -= 1
                
    return grid

def is_level_complete(grid):
    """
    Check if the level is complete.
    The level is complete when the counter reaches 0.
    """
    H, W = grid.shape
    # Check if there are any 4s left
    if np.any(grid == 4):
        return False
    return True