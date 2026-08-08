import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the same logic observed in 'cn04'.
    """
    # Action 4 is a move right operation for certain objects.
    if action == 4:
        # Find the "player" or movable object (color 0 or color 12)
        # In this game, it seems like there's a specific rectangle/block that moves.
        # Let's identify the rectangular blocks of non-background color 10.
        # We need to find the boundaries of the primary movable block.
        # The background is color 10.
        mask = (grid != 10)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return grid
        
        # For simplicity, we assume the only thing moving is the contiguous block of color 0 or 12.
        # # This is not too* complex.
        # Let's look at the others.
        # It's just shifting a block of cells.
        # Shift all cells by 3 columns to the right.
        # New grid
        new_grid = grid.copy()
        
        # Identify the region that behaves as the 'movable' entity.
        # In the observed transitions, ACTION 4 shifts everything except the background.
        # shifted_mask = np.roll(mask, 3, axis=1)
        # But wait, ACTION 4 in the data shows very specific column changes.
        # r14c11:10x3 r14c26:0x3 -> this means cols 11-13 become 10 and cols 26-28 become 0.
        # The movable object is likely the rectangle of color 0 (bbox=(14, 11, 28, 25)).
        # Width is 15? No, bbox x0=11, x1=25. width = 15.
        # bbox y0=14, y1=28. height = 15.
        # Let's find the rectangular block of color 0 or 12.
        # target_color = 0 # Start with a<|channel>thought
        # Find all non-background cells.
        # Find the bounding box of the movable object.
        # Find its current position.
        # Find the same shape as the laout//
        # In observed transitions, it's shifting by 3 columns.
        # shift = 3
        # For every cell that was not background (10), we move it to col + 3.
        # Fill original positions with background (10).
        
        # Correct logic for ACTION 4: Shift everything except the top bar (row 0) 
        # and potentially some static elements. But looking at deltas, only specific rows are affected.
        # The "movable" part is from row 14 to 31.
        # We need to identify which pixels belong to the 'entity'.
        # This entity consists of any pixel in rows [14, 63] that isn't background (10).
        # a few exceptions exist but let's try this general rule.
        
        # Let's refine: find all connected components of color != 10.
        # Only shift those that aren't the top bar or own fixed positions.
        # Actually, look at r0c16:0x1... then r0c17:0x1. Row 0 is also changing!
        # It seems like there's a cursor moving on row 0, and a block moving below.
        
        # Find the current position of the 'cursor' on row 0.
        # Cursor is likely the cell that is NOT color 4 or 10.
        cursor_col = np.where(grid[0] != 4)[0] # Wait, grid[0] has colors 10, 4, 10.
        # In INITIAL GRID: r0:10x16, 4x32, 10x16. The "gap" is where it moves.
        # Let's just use the delta logic: ACTION 4 shifts everything by 3 columns right.
        
        new_grid = grid.copy()
        
        # Shift non-background cells in rows [14, 49] by 3 cols to the right.
        # We must be careful not to shift things that are static (like the walls).
        # But the deltas show only certain objects move.
        # Specifically, the object starting at col 11 (color 0) moves to 14, 17, ...
        # And other objects also move.
        
        # Simple rule: find all pixels of a specific movable shape and shift them.
        # Or even simpler: any pixel in row >= 14 that isn't color 10 and is part of a 'block'.
        # Actually, let's just implement the observed behavior: shift ALL non-10 pixels in rows [14, 49] by 3 units.
        # Wait, some pixels stay put? No, r14c11 becomes 10, r14c26 becomes 0... wait.
        # The block was from c11 to c25. Now it's from c14 to c28.
        # That's exactly a shift of +3.
        
        # Let's apply this to everything except the top bar.
        for r in range(14, 64):
            row = grid[r].copy()
            mask_non_bg = (row != 10)
            new_grid[r][:] = 10 # Reset row to background
            # Shift mask
            shifted_indices = np.where(mask_non_bg)[0] + 3
            valid_indices = shifted_indices[shifted_indices < 64]
            original_indices = np.where(mask_non_bg)[0]
            # We only move if they are not "static" objects? 
            # In the deltas, color 8 and 14 also move!
            # So just shift all non-10 cells by 3.
            for old_idx, new_idx in zip(original_indices, shifted_indices):
                if new_idx < 64:
                    new_grid[r][new_idx] = row[old_idx]

        # Row 0 cursor movement:
        # Find the cell on row 0 that is NOT 4 or 10.
        cursor_pos = np.where((grid[0] != 4) & (grid[0] != 10))[0]
        if cursor_pos.size > 0:
            c = cursor_pos[0]
            new_grid[0, c] = 10 # clear old
            if c + 1 < 64:
                new_grid[0, c+1] = grid[0, c] # move it
        else:
            # If no cursor, maybe it's a specific starting point.
            pass
            
        return new_grid

    elif action == 3:
        # Action 3 seems to be "move left". Shift everything back by 3?
        # Let's check deltas for ACTION 3: r29c38:0x3 r29c47:8x3...
        # This looks like shifting non-10 pixels in rows [14, 49] by -3 columns.
        new_grid = grid.copy()
        for r in range(14, 64):
            row = grid[r].copy()
            mask_non_bg = (row != 10)
            new_grid[r][:] = 10
            shifted_indices = np.where(mask_non_bg)[0] - 3
            original_indices = np.where(mask_non_bg)[0]
            for old_idx, new_idx in zip(original_indices, shifted_indices):
                if new_idx >= 0:
                    new_grid[r][new_idx] = row[old_idx]
        return new_grid

    elif action == 6:
        # Action 6 is a click. data={'x': px, 'y': py}.
        # In the delta, it seems to "clear" or "change" a large area of cells to color 12.
        # bbox=(14, 38, 28, 52) became color 12.
        # The click was at (44, 30). This is roughly the center of that block.
        # It looks like it converts the movable object at that location to color 12.
        new_grid = grid.copy()
        px, py = data['x'], data['y']
        # Find connected component containing (py, px) and change its color to 12?
        # Or just find all non-background pixels within some radius?
        # Let's try finding the same shape as the current "movable block".
        mask = (grid != 10)
        # Simple approach: any pixel in rows [14, 49] that is not background 
        # and is close to (py, px) becomes 12.
        for r in range(14, 64):
            for c in range(64):
                if grid[r, c] != 10:
                    # If it's part of the object we clicked on...
                    # For now, let's just use a distance threshold or bounding box.
                    if abs(r - py) < 15 and abs(c - px) < 15:
                        new_grid[r, c] = 12
        return new_grid

    elif action == 5:
        # Action 5 is the level completion / transition.
        # It doesn't matter what engine returns for ACTION 5 if it's only used to check win state.
        # But the prompt says "the completing action also re-lays out the board for the next level".
        # We can return a dummy grid since we don't have the rules for the next level layout.
        return grid

    return grid

def is_level_complete(grid):
    """
    Determines if the current grid represents a win state.
    A win occurs when the movable block (now color 12) reaches a certain position
    or fulfills a condition.
    In the WIN TRANSITION, the block was at col 38-52 and had color 12.
    Also, there were some other objects of color 0/8.
    Looking at the GRID BEFORE COMPLETING ACTION, the object of color 12 is present.
    Let's assume the goal is to move the 'player' object to the target area and change its color.
    """
    # The most distinct feature of the winning grid is the presence of color 12 in the target region.
    if np.any(grid == 12):
        # Check if the block of color 12 is in the right-hand side (e.g., x > 30).
        coords = np.argwhere(grid == 12)
        if coords.size > 0:
            if np.mean(coords[:, 1]) > 30:
                return True
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells of the same color (excluding background 0)
    must be connected (4-connectivity) and form a single contiguous region.
    """
    grid = np.array(grid)
    colors = np.unique(grid)
    colors = colors[colors != 0]
    
    if len(colors) == 0:
        return False
        
    for color in colors:
        # Find all cells of this color
        cells = np.argwhere(grid == color)
        if len(cells) == 0:
            return False
        
        # BFS to check connectivity
        start_node = tuple(cells[0])
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            r, c = tuple(curr)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        if len(visited) != len(cells):
            return False
            
    return True
