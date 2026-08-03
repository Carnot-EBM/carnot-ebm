import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Inducing rules from observed transitions:
    # The game seems to be a puzzle where an object (represented by color 9 or 4)
    # moves through a maze of corridors (color 0 and 2).
    # The same time, there's some counter/indicator on the bottom row (r63), which changes.
    # Action 2: Down
    # Action 3: Left
    # Action 3: Left
    # Action 4: Right
    # Action 4: Right
    # Action 2: Down
    # This sequence matches the movement of the object (color 9/4) in the coordinates provided in the umapped deltas.
    # Based on the observations, the same time, the la own row (//C63) is<|channel>thoughttto be a
    # a progress indicator.
    
    # Find current position of the "player" (the one moving)
    # We look for the cell that is not background (5) and not part of the static map.
    # Let's identify the player as the block of size 3x3 typically.
    # # In the initial grid, r21-23 c33-35 was color 9.
    # Initial state: r21c33:9x3, r22c33:9x2,4x1, r23c33:9x3.
    # After ACTION2 (Down): it moves to r27-29 c33-35.
    # After ACTION3 (Left): it moves to r33c27-29, then r33c21-23.
    # Wait, the observed transitions show the player starting at r21-23 c33-35.
    # Action 2 (Down) -> r27-29 c33-35.
    # Action 3 (Left) -> r33c27-29? No, looking closer:
    # Transition 1: ACTION2 -> r21-23 c33 becomes 0, r27-29 c33 becomes 9.
    # Transition 2: ACTION2 -> r27-29 c33 becomes 0, r33-35 c33 becomes 9.
    # Transition 3: ACTION3 -> r33-35 c33 becomes 0, r33-35 c27 becomes 9.
    # Transition 4: ACTION3 -> r33-35 c27 becomes 0, r33-35 c21 becomes 9.
    # Transition 5: ACTION2 -> r33-35 c21 becomes 0, r39-41 c21 becomes 9.
    # Transition 6: ACTION4 -> r39-41 c21 becomes 0, r39-41 c27 becomes 9.
    # Transition 7: ACTION4 -> r39-41 c27 becomes 0, r39-41 c33 becomes 9.
    # Transition 8: ACTION2 -> r39-41 c33 becomes 0, r45-47 c33 becomes 9.
    
    # The player is a 3x3 block. Let's find its top-left corner (r, c).
    # It's the region that contains color 9 or 4.
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                player_pos = (r, c)
                break
        if player_pos: break

    if player_pos is None:
        return grid.copy()

    new_grid = grid.copy()
    r, c = player_pos
    
    # Define movement offsets based on action
    # Action 2: Down (moves by 6 rows)
    # Action 3: Left (moves by 6 columns)
    # Action 4: Right (moves by 6 columns)
    move_map = {
        2: (6, 0),
        3: (0, -6),
        4: (0, 6)
    }
    
    if action in move_map:
        dr, dc = move_map[action]
        nr, nc = r + dr, c + dc
        
        # Check if target position is valid (within bounds and not blocked by color 5)
        # The "maze" consists of colors 0, 2, 9, 4. Color 5 is the wall.
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Check if the 3x3 block can fit at (nr, nc)
            can_move = True
            for ir in range(3):
                for ic in range(3):
                    if not (0 <= nr+ir < grid.shape[0] and 0 <= nc+ic < grid.shape[1]) or \
                       grid[nr+ir, nc+ic] == 5:
                        can_move = False
                        break
                if not can_move: break
            
            if can_move:
                # Clear old position
                for ir in range(3):
                    for ic in range(3):
                        new_grid[r+ir, c+ic] = 0 # Assuming corridors are 0
                
                # Set new position
                # We need to preserve the internal structure of the player block (the 4 cell)
                # In initial state: r22c35 was color 4. That's offset (1, 2).
                player_block = np.zeros((3, 3), dtype=int)
                for ir in range(3):
                    for ic in range(3):
                        if grid[r+ir, c+ic] == 9: player_block[ir, ic] = 9
                        elif grid[r+ir, c+ic] == 4: player_block[ir, ic] = 4
                
                for ir in range(3):
                    for ic in range(3):
                        new_grid[nr+ir, nc+ic] = player_block[ir, ic]

    # Update progress indicator on row 63
    # The observed deltas show r63 cells changing from some value to 0.
    # This happens every time a move is successful.
    # Let's find the rightmost non-zero cell in r63 and set it to 0.
    for col in range(grid.shape[1]-1, -1, -1):
        if grid[63, col] != 0:
            # Only if we actually moved
            if action in move_map:
                # We need to check if movement happened. If so, clear one cell.
                # But wait, the deltas are specific: r63c55, then c54, then c52 (2 cells), etc.
                # It looks like the number of cells cleared depends on the distance or just 1 per move?
                # Transition 1: Action 2 -> r63c55:0x1 (1 cell)
                # Transition 2: Action 2 -> r63c54:0x1 (1 cell)
                # Transition 3: Action 3 -> r63c52:0x2 (2 cells)
                # Transition 4: Action 3 -> r63c51:0x1 (1 cell)
                # Transition 5: Action 2 -> r63c50:0x1 (1 cell)
                # Transition 6: Action 4 -> r63c49:0x1 (1 cell)
                # Transition 7: Action 4 -> r63c47:0x2 (2 cells)
                # Transition 8: Action 2 -> r63c46:0x1 (1 cell)
                # This is slightly inconsistent. Let's just clear one cell from the right if moved.
                new_grid[63, col] = 0
                break

    return new_grid

def is_level_complete(grid):
    # Level complete when player reaches a target or progress bar is empty?
    # In this game, we don't have a win state grid, but typically it's reaching color 14.
    # Looking at initial grid: r45-47 c33-35 is color 14.
    # The last transition moves the player to r45-47 c33-35.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9 and any(grid[r+ir, c+ic] == 14 for ir in range(-1, 2) for ic in range(-1, 2) if 0<=r+ir<grid.shape[0] and 0<=c+ic<grid.shape[1]):
                return True
    # Or simply check if the player block overlaps with the goal (color 14).
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                player_pos = (r, c)
                break
        if player_pos: break
    
    if player_pos:
        r, c = player_pos
        for ir in range(3):
            for ic in range(3):
                if 0 <= r+ir < grid.shape[0] and 0 <= c+ic < grid.shape[1]:
                    if grid[r+ir, c+ic] == 14:
                        return True
    return False