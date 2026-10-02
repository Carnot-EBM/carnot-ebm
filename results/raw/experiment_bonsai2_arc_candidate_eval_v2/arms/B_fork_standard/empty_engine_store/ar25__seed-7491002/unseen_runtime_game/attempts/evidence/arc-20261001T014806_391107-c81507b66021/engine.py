import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track player position on the right wall (column 63)
    # The player appears as color 5 at column 63 in some row
    player_row = None
    for r in range(g.shape[0]):
        if g[r, 63] == 5:
            player_row = r
            break
    
    # If no player found yet, check initial state - player might be implicit
    # Looking at transitions, ACTION3 moves player down and changes blocks
    # ACTION5 moves player up
    # ACTION1 moves left (into the field)
    # ACTION2 moves right (back to wall?)
    # ACTION7 swaps colors of adjacent block groups
    
    # Let's identify the movable block structures
    # From the data, there seem to be rectangular blocks made of colors 5 and 4
    # that can move around within the play area
    
    # Based on observed patterns:
    # - There are "block" objects composed of color 5 (with internal 0s) and color 4
    # - They appear to be 9x3 or similar sized rectangles
    # - Actions 1-5 seem to control movement/positioning
    # - Action 7 seems to swap/toggle colors between two block groups
    
    # Simplified model based on observations:
    
    # Identify all non-background cells in the play area (cols 0-62, excluding row 63)
    # Background is color 9, walls are 10 (col 30-32), right border is 11 (col 63)
    
    if action == 3:
        # Move down / shift blocks down
        if player_row is not None:
            g[player_row, 63] = 9  # Remove old player position
            new_row = min(player_row + 1, 62)
            g[new_row, 63] = 5     # Place player at new position
        else:
            # No player yet, place at top
            g[0, 63] = 5
            
    elif action == 5:
        # Move up / shift blocks up
        if player_row is not None:
            g[player_row, 63] = 9
            new_row = max(player_row - 1, 0)
            g[new_row, 63] = 5
        else:
            g[0, 63] = 5

    elif action == 7:
        # Swap colors between the two main block groups
        # Find all cells with color 5 and color 4 in the play area
        mask_5 = (g == 5) & (np.arange(g.shape[0])[:, None] < 63)
        mask_4 = (g == 4) & (np.arange(g.shape[0])[:, None] < 63)
        
        # Temporarily mark them
        temp_val = 99
        g[mask_5] = temp_val
        g[mask_4] = 5
        g[g == temp_val] = 4

    elif action == 1:
        # Move left - this seems to reposition blocks
        # From observations, it shifts the entire block configuration
        pass  # Complex movement, simplified as no-op for now
        
    elif action == 2:
        # Move right
        pass
        
    elif action == 6:
        # Click action - from data, clicking at (53,47) had no effect
        if data is not None:
            px, py = data.get('x', 0), data.get('y', 0)
            # Pixel coords map directly to logical coords (pixel = logical*1)
            x, y = px, py
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                # No observable change in provided transitions
                pass
    
    return g


def is_level_complete(grid):
    # Win condition: based on observed patterns, likely when all blocks are 
    # properly aligned or collected. Without explicit win state grid, we check
    # if the play area has been cleared of movable objects (only background remains)
    
    # Check if there are any color 5 or 4 cells remaining in the play area
    # (excluding the bottom row which is always 5, and column 63 which may have player)
    play_area = grid[:63, :63]
    
    # If no movable blocks remain in the play area, level might be complete
    has_blocks_5 = np.any(play_area == 5)
    has_blocks_4 = np.any(play_area == 4)
    
    # Level complete when both block types are gone from play area
    return not (has_blocks_5 or has_blocks_4)