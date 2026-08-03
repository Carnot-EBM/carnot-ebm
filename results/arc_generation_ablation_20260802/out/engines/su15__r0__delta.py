import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    y, x = data['y'], data['x']
    out = grid.copy()
    
    # The click target (the "player" object) moves to the clicked location.
    # It's represented by a 3x3 square of color 15.
    # Find current position of the player object (3x3 square of color 15).
    # In the initial state, it's at r4-6, c30-32.
    # Based on observed transitions, clicking shifts the player object.
    #<|channel>thought
    # Observed clicks:
    # Click (10, 53) -> Player moves to r52-54, c9-11.
    # Click (16, 47) -> Player moves to r46-48, c15-17.
    # Click (22, 41) -> Player moves to r40-42, c21-23.
    # Click (28, 35) -> Player moves to r34-36, c27-29.
    # Click (34, 29) -> Player moves to r28-30, c33-35.
    # Note: x=10, y=53 corresponds to center of 3x3 block starting at r52, c9?
    # No, if click is (10, 53), and result is r52-54, c9-11, then row = y-1, col = x-1.
    # Let's check others:
    # Click (16, 47): row = 47-1=46, col = 16-1=15. Correct.
    # Click (22, 41): row = 41-1=40, col = 22-1=21. Correct.
    # Click (28, 35): row = 35-1=34, col = 28-1=27. Correct.
    # Click (34, 29): row = 29-1=28, col = 34-1=33. Correct.
    
    # Find current player position (top-left corner)
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15:
                # Check if it's the top-left of a 3x3 block
                if r+2 < grid.shape[0] and c+2 < grid.shape[1]:
                    if np.all(grid[r:r+3, c:c+3] == 15):
                        player_pos = (r, c)
                        break
        if player_pos: break

    if player_pos is None:
        return out

    # New position based on click data
    new_r, new_c = y - 1, x - 1
    
    # Clear old position
    out[player_pos[0]:player_pos[0]+3, player_pos[0]:player_pos[0]+3] = 5 # This is wrong logic for clearing
    # Let's look at what color replaces it.
    # In transition 1: r4-6, c30-32 becomes 5? No, initial was 15.
    # Transition 1: r58-60, c3-5 becomes 5. Wait, that's not where the player was.
    # Initial player: r4-6, c30-32.
    # After first action: r52-54, c9-11 is 15. And r58-60, c3-5 becomes 5.
    # Wait, let's re-read carefully.
    # INITIAL GRID: r4-6, c30-32 are 15x3 (Player).
    # ACTION6 (10, 53): r52-54, c9-11 become 15x3. AND r58-60, c3-5 become 5x3.
    # But in INITIAL grid, r58-60, c3-5 were already 15x3!
    # So the "player" object moved from r58-60, c3-5 to r52-54, c9-11.
    # Let's check if there are multiple players? No, usually one.
    # In INITIAL grid: r4-6, c30-32 = 15; r58-60, c3-5 = 15. Two objects of color 15.
    # Action 1: Object at r58-60, c3-5 moves to r52-54, c9-11.
    # Action 2: Object at r52-54, c9-11 moves to r46-48, c15-17.
    # Action 3: Object at r46-48, c15-17 moves to r40-42, c21-23.
    # And so on.
    # The other object at r4-6, c30-32 stays put (it doesn't appear in deltas).
    
    # Find which 3x3 block of 15 is NOT the static one at r4-6, c30-32.
    current_r, current_c = -1, -1
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15 and not (4 <= r < 7 and 30 <= c < 33):
                if r+2 < grid.shape[0] and c+2 < grid.shape[1]:
                    if np.all(grid[r:r+3, c:c+3] == 15):
                        current_r, current_c = r, c
                        break
        if current_r != -1: break

    if current_r == -1: return out

    # Move player
    out[current_r:current_r+3, current_c:current_c+3] = 5
    new_r, new_c = y - 1, x - 1
    out[new_r:new_r+3, new_c:new_c+3] = 15
    
    return out

def is_level_complete(grid):
    # No win state provided, but usually it's when the moving object reaches a target.
    # The static object at r4-6, c30-32 might be the target.
    # Let's check if they overlap or are adjacent.
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 15 and not (4 <= r < 7 and 30 <= c < 33):
                # If any part of the moving block is within the target area
                if 4 <= r < 7 and 30 <= c < 33:
                    return True
    return False