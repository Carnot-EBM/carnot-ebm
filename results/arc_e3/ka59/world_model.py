import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    World model for ARC-AGI game 'ka59'.
    The grid contains two moving frames (color 14) of size 3x3 with a hole in the center.
    Action 3 moves them left by 3 pixels; Action 4 moves them right by 3 pixels.
    Row 63 acts as a progress bar/timer, decreasing from right to left on most actions.
    """
    next_grid = grid.copy()
    h, w = next_grid.shape

    # Identify all color 14 blocks (frames). A frame is a 3x3 block with a hole at its center.
    # We look for the top-left corner of these 3x3 structures.
    frames = []
    visited = set()
    for r in range(h - 2):
        for c in range(w - 2):
            if (r, c) not in visited:
                # Check if it's a 3x3 frame of color 14
                is_frame = True
                for dr in range(3):
                    for dc in range(3):
                        if dr == 1 and dc == 1:
                            continue # Hole in the middle
                        if next_grid[r + dr, c + dc] != 14:
                            is_frame = False
                            break
                    if not is_frame: break
                
                if is_frame:
                    frames.append([r, c])
                    for dr in range(3):
                        for dc in range(3):
                            visited.add((r + dr, c + dc))

    # Movement logic based on Action 3 (Left) and Action 4 (Right)
    move_dist = 0
    if action == 3:
        move_dist = -3
    elif action == 4:
        move_dist = 3

    if move_dist != 0:
        for fr, fc in frames:
            # Clear current frame position
            for dr in range(3):
                for dc in range(3):
                    if dr == 1 and dc == 1: continue
                    next_grid[fr + dr, fc + dc] = grid[fr + dr, fc + dc] if grid[fr+dr, fc+dc] != 14 else 2 # Simplified clear
            
            # New position
            nc = fc + move_dist
            if 0 <= nc <= w - 3:
                # Draw new frame
                for dr in range(3):
                    for dc in range(3):
                        if dr == 1 and dc == 1: continue
                        next_grid[fr + dr, nc + dc] = 14
                # Restore the hole's original color at the new center
                next_grid[fr + 1, nc + 1] = grid[fr + 1, fc + 1]
            else:
                # If it goes out of bounds, we just leave it (or handle as needed)
                pass

    # Progress bar logic on row 63
    # The progress bar decreases from right to left. Find the rightmost cell that is still color 4.
    if h > 63:
        rightmost_4 = -1
        for c in range(w - 1, -1, -1):
            if next_grid[63, c] == 4:
                rightmost_4 = c
                break
        
        # Most actions trigger a decrease in the progress bar.
        # Based on observations, Action 2 doesn't always change r63, but others do.
        if action != 2 and rightmost_4 != -1:
            next_grid[63, rightmost_4] = 0

    return next_grid

def is_level_complete(grid):
    """
    The level is complete when the progress bar at row 63 is fully depleted (all zeros).
    """
    if grid.shape[0] <= 63:
        return False
    return np.all(grid[63, :] == 0)

import numpy as np

def is_level_complete(grid):
    """
    Determines if the grid for ARC-AGI-3 game 'ka59' is in a win state.
    The general win condition for this completion task is that the grid 
    is entirely filled with a single non-zero color.
    """
    if grid.size == 0:
        return False
    
    # Get the color of the first cell
    first_color = grid[0, 0]
    
    # The level is complete if all cells match the first cell and the color is not the background (0)
    return first_color != 0 and np.all(grid == first_color)
