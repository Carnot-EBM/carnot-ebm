import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # ACTION4 is a rightward shift of the 'empty' space/object and the corresponding 
    # layout elements. Based on the observed transitions, it seems like a puzzle where 
    # moving the empty space (color 0) moves the blocks.
    # The game state consists of a background (color 10), background-top (color 4),
    # and several blocks of different colors (0, 8, 14).
    
    # In this specific level, ACTION4 shifts everything in the "playable area" 
    # within the same row or region.
    # Find the current position of the color 0 object.
    # zero_pos = np.where(grid == 0)
    # However, looking at thes deltas, ACTION4 consistently shifts the entire pattern of non-background cells 
    # (// This is part of a logic induced from the<|channel>thought process)
    
    # We need to identify the bounding box of the "puzzle" content.
    # Let's define the playable area as the region containing colors other than 10 and 4.
    # 
    # The transitions show that when ACTION4 is called, the x-coordinates of all 
    # block positions shift by +3 pixels/units.
    # Shift amount: 3 units.
    # Shift direction: Rightward.
    #
    # Looking at the observed transitions:
    # r14c11:10x3 r14c26:0x3 ... -> r14c14:10x3 r14c29:0x3
    # The empty space (color 0) moves from c11..13 to c14..17... no, wait.
    # The delta shows r14c11 becomes 10, and r14c26 becomes 0.
    # Wait, let's re-examine:
    # Initial: r14: 10x11, 0x15, 10x38  (col 11 to 25 is color 0)
    # Action 4 first delta: r14c11:10x3, r14c26:0x3 (Wait, this is a partial update?)
    # No, it says "changed cells".
    # Let's look at the coordinates of the '0' blocks in each transition.
    # Transition 1: r14c26:0x3 (and others). This means col 26-28 became 0.
    # Transition 2: r14c29:0x3. Col 29-31 became 0.
    # Transition 3: r14c32:0x3. Col 32-34 became 0.
    # It seems ACTION4 shifts the entire puzzle content right by 3 units.
    
    if action == 4:
        new_grid = grid.copy()
        # Identify all cells that are not background (color 10 or 4)
        # We only shift things that aren't the static top bar (color 4)
        mask = (grid != 10) & (grid != 4)
        
        # To avoid shifting the same cell multiple times, we work on a copy
        # Shift everything in the mask to the right by 3
        # But wait, the deltas show color 10 filling in the gaps and colors moving.
        # Let's simply shift the non-background elements.
        
        # Find coordinates of all "puzzle" pieces
        coords = np.argwhere(mask)
        for r, c in coords:
            val = grid[r, c]
            # The boundaries of the board are 64x64.
            # If it moves off board, it disappears.
            if c + 3 < 64:
                new_grid[r, c + 3] = val
            # Fill original position with background if it wasn't shifted into
            # However, this is tricky because multiple cells might move to one spot.
            # A better way: create a blank grid of color 10 and place shifted items.
    
    # Re-evaluating based on the deltas:
    # ACTION4 shifts the entire pattern of blocks (including the 'empty' space 0) 
    # relative to the background 10.
    
    # Correct logic for ACTION4:
    # 1. Identify the region containing the puzzle (everything not color 10 or 4).
    # 2. Shift that whole content right by 3 units.
    # 3. Cells that were part of the puzzle but now shift out are replaced by 10.
    # 4. The top bar (color 4) remains static.
    
    if action == 4:
        # Create a mask of what needs to be shifted
        # In this level, everything except color 10 and color 4 is "puzzle"
        # But wait, the empty space (color 0) IS part of the puzzle.
        # So we shift anything that isn't color 10 or color 4? 
        # No, looking at INITIAL GRID, r14 has 0x15. That's the puzzle.
        # Let's just shift all cells in rows 14-63 that aren't color 10.
        # Actually, let's shift ALL cells in rows 14-63, and then restore the background if needed.
        # Or even simpler: shift any cell that is NOT color 10 AND NOT color 4.
        # Wait, the deltas show color 0 moving too. Color 0 is not 10 and not 4.
        
        new_grid = grid.copy()
        # We must process from right to left to avoid overwriting.
        for r in range(14, 64):
            row = grid[r, :].copy()
            # Shift row by 3
            shifted_row = np.full(64, 10, dtype=int)
            for c in range(64 - 3):
                if row[c] != 10:
                    shifted_row[c + 3] = row[c]
            # Special case: The top bar (color 4) is at r=0. It doesn't move.
            new_grid[r, :] = shifted_row
        return new_grid

    return grid

def is_level_complete(grid):
    # Win state usually involves blocks reaching a target or a specific configuration.
    # Given no win state provided, we return False unless a known condition is met.
    # In many ARC games, completing means all blocks of one color are gone or aligned.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for cn04 is that all cells in the grid are the same color.
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    
    # Check if all elements are the same
    return np.all(grid == grid[0, 0])
