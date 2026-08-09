import numpy as np

def engine(grid, action, data):
    """
    World model for ARC-AGI game 'm0r0'.
    The game involves moving two 5x5 blocks of color 10 within a central area (color 5),
    merging them into one block, and growing boundary markers (color 0).
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    # Identify all color 10 blocks
    blocks = []
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if new_grid[r, c] == 10 and not visited[r, c]:
                block_cells = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    curr_r, curr_c = stack.pop()
                    block_cells.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and \
                           new_grid[nr, nc] == 10 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                rs = [cell[0] for cell in block_cells]
                cs = [cell[1] for cell in block_cells]
                blocks.append({'y0': min(rs), 'x0': min(cs), 'y1': max(rs), 'x1': max(cs)})

    # Sort blocks by their leftmost column to identify B1 (leftmost) and B2 (rightmost)
    blocks.sort(key=lambda b: b['x0'])

    def move_block(block, dy, dx):
        """Moves a block's cells on the grid."""
        # Clear old position
        for r in range(block['y0'], block['y1'] + 1):
            for c in range(block['x0'], block['x1'] + 1):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 5 # Return to background color
        
        # Calculate new boundaries with constraints
        ny0 = max(9, block['y0'] + dy)
        ny1 = ny0 + (block['y1'] - block['y0'])
        nx0 = max(9, min(w - 6, block['x0'] + dx)) # Simplified boundary for x
        nx1 = nx0 + (block['x1'] - block['x0'])
        
        # Ensure it doesn't go out of bounds
        ny1 = min(h - 6, ny1)
        nx1 = min(w - 1, nx1)
        
        # Set new position
        for r in range(ny0, ny1 + 1):
            for c in range(nx0, nx1 + 1):
                new_grid[r, c] = 10
        return {'y0': ny0, 'x0': nx0, 'y1': ny1, 'x1': nx1}

    if action == 1:
        # Both blocks move UP by 5. P1 moves left, P2 moves right.
        for b in blocks:
            move_block(b, -5, 0)
        # Grow color 0 markers
        p1_len = np.sum(new_grid[0, :] == 0)
        new_grid[0, w - p1_len - 1] = 0
        p2_len = np.sum(new_grid[h-1, :] == 0)
        new_grid[h-1, p2_len] = 0

    elif action == 3:
        # Rightmost block moves RIGHT by 5.
        if blocks:
            move_block(blocks[-1], 0, 5)

    elif action == 4:
        # Leftmost block moves RIGHT by 5, Rightmost block moves LEFT by 5.
        # If only one block exists, it stays put (moves both ways).
        if len(blocks) >= 2:
            move_block(blocks[0], 0, 5)
            move_block(blocks[-1], 0, -5)
        elif len(blocks) == 1:
            pass # Stays in place as per logic derived from merged state
            
        # Grow color 0 markers
        p1_len = np.sum(new_grid[0, :] == 0)
        new_grid[0, w - p1_len - 1] = 0
        p2_len = np.sum(new_grid[h-1, :] == 0)
        new_grid[h-1, p2_len] = 0

    return new_grid

def is_level_complete(grid):
    """
    Level is complete when the two blocks of color 10 have merged into a single 
    larger block and boundary markers reach a certain size.
    Based on observed data, merging them to a width of 10 seems key.
    """
    # Count connected components of color 10
    visited = np.zeros(grid.shape, dtype=bool)
    num_blocks = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 10 and not visited[r, c]:
                num_blocks += 1
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    curr_r, curr_c = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and \
                           grid[nr, nc] == 10 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    
    # Win condition: blocks merged into one AND markers have grown sufficiently
    p1_len = np.sum(grid[0, :] == 0)
    p2_len = np.sum(grid[-1, :] == 0)
    
    return num_blocks == 1 and p1_len >= 7 and p2_len >= 7